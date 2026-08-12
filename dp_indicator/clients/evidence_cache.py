"""Evidence Cache Client - PMID/DOI-based metadata and abstract caching.

Provides a unified interface to fetch full abstracts and metadata by evidence ID,
with SQLite caching (7-day TTL) to avoid repeated API calls.

Supports:
- PMID -> NCBI E-utilities (esummary + efetch)
- EPMC ID -> EuropePMC REST API
- DOI -> EuropePMC / Crossref lookup

Cache layers:
  Layer 1: In-memory dict (per-session)
  Layer 2: SQLite cache (7-day TTL, shared across runs)
  Layer 3: External API (NCBI / EuropePMC)
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Cache TTL: 7 days (literature metadata doesn't change)
_CACHE_TTL_SECONDS = 7 * 24 * 3600


class EvidenceCacheClient:
    """Cache and fetch evidence metadata by PMID/EPMC ID/DOI."""

    def __init__(self, cache_dir: str = "data/cache/evidence",
                 email: str = "bohrium-agent@dp.tech"):
        os.makedirs(cache_dir, exist_ok=True)
        self._db_path = os.path.join(cache_dir, "evidence_cache.db")
        self._email = email
        self._http_client: Optional[httpx.AsyncClient] = None
        self._memory_cache: dict[str, dict] = {}
        self._init_db()

    def _init_db(self):
        """Initialize SQLite cache table."""
        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evidence_cache (
                cache_key TEXT PRIMARY KEY,
                data_json TEXT NOT NULL,
                cached_at REAL NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(15, connect=10),
                follow_redirects=True,
                headers={"User-Agent": "BohrClaw-EvidenceCache/1.0"},
            )
        return self._http_client

    async def close(self):
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    # ── Cache helpers ──

    def _cache_get(self, key: str) -> Optional[dict]:
        # Layer 1: memory
        if key in self._memory_cache:
            return self._memory_cache[key]
        # Layer 2: SQLite
        conn = sqlite3.connect(self._db_path)
        row = conn.execute(
            "SELECT data_json, cached_at FROM evidence_cache WHERE cache_key = ?",
            (key,)
        ).fetchone()
        conn.close()
        if row is None:
            return None
        data_json, cached_at = row
        if time.time() - cached_at > _CACHE_TTL_SECONDS:
            return None  # expired
        data = json.loads(data_json)
        self._memory_cache[key] = data  # promote to memory
        return data

    def _cache_set(self, key: str, data: dict):
        self._memory_cache[key] = data
        conn = sqlite3.connect(self._db_path)
        conn.execute(
            "INSERT OR REPLACE INTO evidence_cache (cache_key, data_json, cached_at) VALUES (?, ?, ?)",
            (key, json.dumps(data, ensure_ascii=False), time.time())
        )
        conn.commit()
        conn.close()

    # ── Public API ──

    async def fetch_by_id(self, evidence_id: str) -> Optional[dict]:
        """Fetch full metadata + abstract by evidence ID.

        Routes by ID prefix:
          - PMID:12345 -> NCBI E-utilities
          - EPMC:12345 -> EuropePMC REST API
          - DOI:10.xxx -> EuropePMC DOI search

        Returns:
            {
                "id": str,
                "title": str,
                "abstract": str (full, untruncated),
                "authors": [str],
                "journal": str,
                "year": int,
                "doi": str,
                "pmid": str,
                "pmcid": str,
                "source": str (which API returned this)
            }
            or None if not found.
        """
        cache_key = evidence_id.lower()
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        data = None
        eid_upper = evidence_id.upper()
        if eid_upper.startswith("PMID:"):
            pmid = evidence_id.split(":", 1)[1]
            data = await self._fetch_ncbi(pmid)
        elif eid_upper.startswith("EPMC:"):
            epmc_id = evidence_id.split(":", 1)[1]
            data = await self._fetch_europepmc(epmc_id)
        elif eid_upper.startswith("DOI:"):
            doi = evidence_id.split(":", 1)[1]
            data = await self._fetch_by_doi(doi)
        else:
            # Try PMID first (numeric IDs)
            if evidence_id.isdigit():
                data = await self._fetch_ncbi(evidence_id)
            else:
                # Try as DOI
                data = await self._fetch_by_doi(evidence_id)

        if data is not None:
            self._cache_set(cache_key, data)
        return data

    async def fetch_batch(self, evidence_ids: list[str]) -> dict[str, dict]:
        """Fetch metadata for multiple IDs. Returns {evidence_id: metadata_dict}.

        Uses NCBI batch API for PMIDs (efficient), individual calls for others.
        """
        # Deduplicate
        unique_ids = list(set(evidence_ids))
        results: dict[str, dict] = {}
        missing: list[str] = []

        # Check cache for all
        for eid in unique_ids:
            cached = self._cache_get(eid.lower())
            if cached is not None:
                results[eid] = cached
            else:
                missing.append(eid)

        if not missing:
            return results

        # Split by type
        pmids = [e.split(":", 1)[1] for e in missing if e.upper().startswith("PMID:")]
        pmids += [e for e in missing if e.isdigit()]
        non_pmid = [e for e in missing if not e.upper().startswith("PMID:") and not e.isdigit()]

        # Batch fetch PMIDs via NCBI esummary
        if pmids:
            batch_results = await self._fetch_ncbi_batch(pmids)
            for pmid, data in batch_results.items():
                eid = f"PMID:{pmid}" if f"PMID:{pmid}" in missing else pmid
                results[eid] = data
                self._cache_set(eid.lower(), data)

        # Fetch non-PMID individually
        for eid in non_pmid:
            data = await self.fetch_by_id(eid)
            if data:
                results[eid] = data

        return results

    # ── NCBI E-utilities ──

    async def _fetch_ncbi(self, pmid: str) -> Optional[dict]:
        """Fetch single PMID via NCBI E-utilities."""
        client = await self._get_client()
        try:
            # esummary for metadata
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={pmid}&retmode=json&email={self._email}"
            resp = await client.get(url, timeout=10)
            if resp.status_code != 200:
                return None
            data = resp.json()
            result = data.get("result", {})
            if str(pmid) not in result:
                return None
            entry = result[str(pmid)]
            if entry.get("error"):
                return None

            # efetch for abstract
            abstract = await self._fetch_ncbi_abstract(pmid)

            authors = [a.get("name", "") for a in entry.get("authors", [])[:10]]
            return {
                "id": f"PMID:{pmid}",
                "title": entry.get("title", ""),
                "abstract": abstract or "",
                "authors": authors,
                "journal": entry.get("fulljournalname", entry.get("source", "")),
                "year": int(entry.get("pubdate", "0000")[:4]) if entry.get("pubdate") else 0,
                "doi": next((aid.get("value", "") for aid in entry.get("articleids", []) if aid.get("idtype") == "doi"), ""),
                "pmid": pmid,
                "pmcid": next((aid.get("value", "") for aid in entry.get("articleids", []) if aid.get("idtype") == "pmc"), ""),
                "source": "ncbi_esummary",
            }
        except Exception as e:
            logger.debug(f"NCBI fetch failed for PMID {pmid}: {e}")
            return None

    async def _fetch_ncbi_abstract(self, pmid: str) -> str:
        """Fetch abstract via efetch."""
        client = await self._get_client()
        try:
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&rettype=abstract&retmode=text&email={self._email}"
            resp = await client.get(url, timeout=10)
            if resp.status_code == 200:
                text = resp.text.strip()
                # Clean up: remove PMID line, keep abstract
                lines = [l for l in text.split("\n") if not l.startswith(f"{pmid}.") and l.strip()]
                return "\n".join(lines)[:5000]
        except Exception as e:
            logger.debug(f"NCBI abstract fetch failed for PMID {pmid}: {e}")
        return ""

    async def _fetch_ncbi_batch(self, pmids: list[str]) -> dict[str, dict]:
        """Batch fetch multiple PMIDs via NCBI esummary."""
        if not pmids:
            return {}
        client = await self._get_client()
        results: dict[str, dict] = {}

        # NCBI allows up to 200 IDs per request
        for i in range(0, len(pmids), 200):
            batch = pmids[i:i+200]
            ids_str = ",".join(batch)
            try:
                url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={ids_str}&retmode=json&email={self._email}"
                resp = await client.get(url, timeout=15)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                result = data.get("result", {})
                uids = result.get("uids", [])
                for uid in uids:
                    entry = result.get(uid, {})
                    if entry.get("error"):
                        continue
                    authors = [a.get("name", "") for a in entry.get("authors", [])[:10]]
                    results[uid] = {
                        "id": f"PMID:{uid}",
                        "title": entry.get("title", ""),
                        "abstract": "",  # batch esummary doesn't include abstract
                        "authors": authors,
                        "journal": entry.get("fulljournalname", entry.get("source", "")),
                        "year": int(entry.get("pubdate", "0000")[:4]) if entry.get("pubdate") else 0,
                        "doi": next((aid.get("value", "") for aid in entry.get("articleids", []) if aid.get("idtype") == "doi"), ""),
                        "pmid": uid,
                        "pmcid": next((aid.get("value", "") for aid in entry.get("articleids", []) if aid.get("idtype") == "pmc"), ""),
                        "source": "ncbi_esummary_batch",
                    }
            except Exception as e:
                logger.warning(f"NCBI batch fetch failed: {e}")

        # Fetch abstracts individually for batch results (parallel with concurrency limit)
        sem = asyncio.Semaphore(4)
        async def _fetch_abs(pmid: str):
            async with sem:
                abs_text = await self._fetch_ncbi_abstract(pmid)
                if pmid in results:
                    results[pmid]["abstract"] = abs_text

        await asyncio.gather(*[_fetch_abs(p) for p in pmids if p in results], return_exceptions=True)
        return results

    # ── EuropePMC ──

    async def _fetch_europepmc(self, epmc_id: str) -> Optional[dict]:
        """Fetch via EuropePMC REST API."""
        client = await self._get_client()
        try:
            url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=SRC_ID:{epmc_id}&format=json&resultType=core"
            resp = await client.get(url, timeout=10)
            if resp.status_code != 200:
                return None
            data = resp.json()
            hits = data.get("resultList", {}).get("result", [])
            if not hits:
                return None
            entry = hits[0]
            return {
                "id": f"EPMC:{epmc_id}",
                "title": entry.get("title", ""),
                "abstract": entry.get("abstractText", entry.get("abstract", "")) or "",
                "authors": [a.get("fullName", "") for a in entry.get("authorList", {}).get("author", [])[:10]],
                "journal": entry.get("journalTitle", ""),
                "year": int(entry.get("pubYear", "0") or "0"),
                "doi": entry.get("doi", ""),
                "pmid": entry.get("pmid", ""),
                "pmcid": entry.get("pmcid", ""),
                "source": "europepmc",
            }
        except Exception as e:
            logger.debug(f"EuropePMC fetch failed for {epmc_id}: {e}")
            return None

    async def _fetch_by_doi(self, doi: str) -> Optional[dict]:
        """Fetch via DOI using EuropePMC search."""
        client = await self._get_client()
        try:
            url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:{doi}&format=json&resultType=core"
            resp = await client.get(url, timeout=10)
            if resp.status_code != 200:
                return None
            data = resp.json()
            hits = data.get("resultList", {}).get("result", [])
            if not hits:
                return None
            entry = hits[0]
            return {
                "id": f"DOI:{doi}",
                "title": entry.get("title", ""),
                "abstract": entry.get("abstractText", entry.get("abstract", "")) or "",
                "authors": [a.get("fullName", "") for a in entry.get("authorList", {}).get("author", [])[:10]],
                "journal": entry.get("journalTitle", ""),
                "year": int(entry.get("pubYear", "0") or "0"),
                "doi": doi,
                "pmid": entry.get("pmid", ""),
                "pmcid": entry.get("pmcid", ""),
                "source": "europepmc_doi",
            }
        except Exception as e:
            logger.debug(f"EuropePMC DOI fetch failed for {doi}: {e}")
            return None
