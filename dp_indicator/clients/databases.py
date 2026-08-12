from __future__ import annotations
import asyncio
import httpx
import json as _json
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from .base import BaseClient

# Module-level shared httpx clients per (base_url, timeout) tuple (connection pooling + rate limiting)
_shared_clients: dict[tuple[str, float], httpx.AsyncClient] = {}

# Retry config for external DB calls
DB_MAX_RETRIES = 2
DB_RETRY_DELAY = 2.0  # seconds
DB_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
DB_RETRY_EXCEPTIONS = (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError, httpx.CloseError)


async def _get_http_client(base_url: str, timeout: float = 30) -> httpx.AsyncClient:
    """Get or create a shared httpx.AsyncClient for the given base URL and timeout."""
    cache_key = (base_url, timeout)
    if cache_key not in _shared_clients or _shared_clients[cache_key].is_closed:
        _shared_clients[cache_key] = httpx.AsyncClient(
            base_url=base_url, timeout=timeout,
            limits=httpx.Limits(max_connections=10))
    return _shared_clients[cache_key]


async def _request_with_retry(client: httpx.AsyncClient, method: str, url: str, **kwargs) -> httpx.Response:
    """Execute HTTP request with exponential backoff retry for transient errors."""
    last_error = None
    for attempt in range(DB_MAX_RETRIES + 1):
        try:
            resp = await client.request(method, url, **kwargs)
            if resp.status_code in DB_RETRY_STATUS_CODES and attempt < DB_MAX_RETRIES:
                wait = DB_RETRY_DELAY * (2 ** attempt)
                retry_after = resp.headers.get("retry-after")
                if retry_after:
                    try:
                        wait = max(float(retry_after), wait)
                    except ValueError:
                        pass
                print(f"  [DB] HTTP {resp.status_code}, retry {attempt+1}/{DB_MAX_RETRIES} in {wait:.1f}s", flush=True)
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except DB_RETRY_EXCEPTIONS as e:
            last_error = e
            if attempt >= DB_MAX_RETRIES:
                raise
            wait = DB_RETRY_DELAY * (2 ** attempt)
            print(f"  [DB] {type(e).__name__}, retry {attempt+1}/{DB_MAX_RETRIES} in {wait:.1f}s", flush=True)
            await asyncio.sleep(wait)
    raise last_error


async def _close_all_clients():
    for client in list(_shared_clients.values()):
        if not client.is_closed:
            await client.aclose()
    _shared_clients.clear()

def shutdown():
    """Synchronous cleanup of shared HTTP client pool.

    Safe to call from sync code (e.g., CLI main) after asyncio.run().
    Creates a temporary event loop to close shared clients.
    """
    if not _shared_clients:
        return
    try:
        asyncio.run(_close_all_clients())
    except RuntimeError:
        # Event loop already closed, ignore
        pass

class PubMedClient(BaseClient):
    def __init__(self, cache_dir: str = "data/cache", api_key: str = None):
        super().__init__("https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
                         rate_limit=10 if api_key else 3, cache_dir=cache_dir)
        self.api_key = api_key

    @staticmethod
    def _extract_year(pub_date: str) -> int | None:
        """Extract 4-digit year from pub_date string."""
        if not pub_date:
            return None
        import re
        m = re.search(r"\b(19|20)\d{2}\b", pub_date)
        return int(m.group(0)) if m else None

    def _parse_pubmed_xml(self, xml_text: str) -> dict[str, dict]:
        """Parse PubMed efetch XML and return a dict mapping PMID -> article data."""
        articles: dict[str, dict] = {}
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            print(f"  [DB] PubMed XML parse error: {e}, returning empty", flush=True)
            return articles
        for article in root.findall("PubmedArticle"):
            medline = article.find("MedlineCitation")
            if medline is None:
                continue
            pmid_elem = medline.find("PMID")
            if pmid_elem is None:
                continue
            pmid = str(pmid_elem.text or "").strip()
            if not pmid:
                continue

            article_elem = medline.find("Article")
            if article_elem is None:
                continue

            # Title
            title_elem = article_elem.find("ArticleTitle")
            title = "".join(title_elem.itertext()).strip() if title_elem is not None else ""

            # Abstract (may have multiple AbstractText children with Labels)
            abstract_parts = []
            abstract_elem = article_elem.find("Abstract")
            if abstract_elem is not None:
                for abs_text in abstract_elem.findall("AbstractText"):
                    label = abs_text.get("Label", "")
                    text = "".join(abs_text.itertext()).strip()
                    if text:
                        if label:
                            abstract_parts.append(f"{label}: {text}")
                        else:
                            abstract_parts.append(text)
            abstract = "\n".join(abstract_parts)

            # Publication types
            pub_types = []
            pub_type_list = article_elem.find("PublicationTypeList")
            if pub_type_list is not None:
                for pt in pub_type_list.findall("PublicationType"):
                    text = "".join(pt.itertext()).strip()
                    if text:
                        pub_types.append(text)

            # Publication date (prefer ArticleDate, fallback to Journal/JournalIssue/PubDate)
            pub_date = ""
            article_date = article_elem.find("ArticleDate")
            if article_date is not None:
                year = article_date.find("Year")
                month = article_date.find("Month")
                day = article_date.find("Day")
                parts = [p.text for p in (year, month, day) if p is not None and p.text]
                if parts:
                    pub_date = "-".join(parts)
            if not pub_date:
                journal = article_elem.find("Journal")
                if journal is not None:
                    ji = journal.find("JournalIssue")
                    if ji is not None:
                        pd = ji.find("PubDate")
                        if pd is not None:
                            pd_parts = []
                            for tag in ("Year", "Month", "Day"):
                                t = pd.find(tag)
                                if t is not None and t.text:
                                    pd_parts.append(t.text)
                            if pd_parts:
                                pub_date = " ".join(pd_parts)
                            else:
                                medline_date = pd.find("MedlineDate")
                                if medline_date is not None and medline_date.text:
                                    pub_date = medline_date.text.strip()

            # Authors
            authors = []
            author_list = article_elem.find("AuthorList")
            if author_list is not None:
                for author in author_list.findall("Author"):
                    last = author.find("LastName")
                    fore = author.find("ForeName")
                    initials = author.find("Initials")
                    if last is not None and last.text:
                        name = last.text
                        if fore is not None and fore.text:
                            name = f"{fore.text} {last.text}"
                        elif initials is not None and initials.text:
                            name = f"{initials.text} {last.text}"
                        authors.append(name)

            # Journal info
            journal_title = ""
            journal_iso = ""
            volume = ""
            issue = ""
            pages = ""
            journal = article_elem.find("Journal")
            if journal is not None:
                jt = journal.find("Title")
                if jt is not None:
                    journal_title = "".join(jt.itertext()).strip()
                jiso = journal.find("ISOAbbreviation")
                if jiso is not None:
                    journal_iso = "".join(jiso.itertext()).strip()
                ji = journal.find("JournalIssue")
                if ji is not None:
                    vol = ji.find("Volume")
                    if vol is not None and vol.text:
                        volume = vol.text
                    iss = ji.find("Issue")
                    if iss is not None and iss.text:
                        issue = iss.text

            # Pagination
            pagination = article_elem.find("Pagination")
            if pagination is not None:
                mp = pagination.find("MedlinePgn")
                if mp is not None and mp.text:
                    pages = mp.text

            # DOI
            doi = ""
            for eloc in article_elem.findall("ELocationID"):
                if eloc.get("EIdType") == "doi":
                    doi = "".join(eloc.itertext()).strip()
                    break

            articles[pmid] = {
                "title": title,
                "abstract": abstract,
                "pub_types": pub_types,
                "pub_date": pub_date,
                "authors": authors,
                "first_author": authors[0] if authors else "",
                "journal": journal_title,
                "journal_iso": journal_iso,
                "volume": volume,
                "issue": issue,
                "pages": pages,
                "doi": doi,
            }
        return articles

    async def search(self, target: str, cutoff: str = None, max_results: int = 30) -> list[dict]:
        await self._wait_rate()
        params = {
            "db": "pubmed", "term": target, "retmax": max_results,
            "retmode": "json", "sort": "relevance",
        }
        if self.api_key:
            params["api_key"] = self.api_key
        key = self._cache_key(params)
        cached = self._get_cached(key)
        if cached:
            return _json.loads(cached)

        results = []
        client = await _get_http_client(self.base_url, timeout=30)

        # Step 1: esearch to get PMID list
        resp = await _request_with_retry(
            client, "GET", f"{self.base_url}/esearch.fcgi", params=params)
        resp.raise_for_status()
        data = resp.json()
        ids = data.get("esearchresult", {}).get("idlist", [])
        if not ids:
            self._set_cached(key, _json.dumps(results))
            return results

        # Step 2: efetch in batches of 50 to get full XML abstracts (smaller batches = more stable)
        batch_size = 50
        articles_map: dict[str, dict] = {}
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            await self._wait_rate()
            efetch_params = {
                "db": "pubmed",
                "id": ",".join(batch_ids),
                "retmode": "xml",
                **({"api_key": self.api_key} if self.api_key else {}),
            }
            fetch_resp = await _request_with_retry(
                client, "GET", f"{self.base_url}/efetch.fcgi", params=efetch_params)
            fetch_resp.raise_for_status()
            batch_articles = self._parse_pubmed_xml(fetch_resp.text)
            articles_map.update(batch_articles)

        # Step 3: Build results in original esearch order
        for pmid in ids:
            article = articles_map.get(pmid)
            if article is None:
                # Fallback: minimal entry if efetch missed this PMID
                results.append({
                    "evidence_id": f"PMID:{pmid}",
                    "source_db": "pubmed",
                    "evidence_type": "literature",
                    "title": "",
                    "abstract_snippet": "",
                    "publication_date": "",
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "source_client": "PubMedClient",
                    "query_params": {"term": target, "retmax": max_results},
                    "raw_id": pmid,
                    "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "source_metadata": {"source_type": "literature"},
                })
                continue

            pub_types_str = " ".join(article["pub_types"]).lower()
            if "randomized controlled trial" in pub_types_str:
                ev_type = "RCT_human"
            elif "review" in pub_types_str:
                ev_type = "review"
            else:
                ev_type = "literature"

            # Build source_metadata
            source_metadata = {
                "source_type": "literature",
                "authors": article.get("authors", []),
                "first_author": article.get("first_author", ""),
                "year": self._extract_year(article["pub_date"]),
                "journal": article.get("journal", ""),
                "journal_short": article.get("journal_iso", ""),
                "volume": article.get("volume", ""),
                "issue": article.get("issue", ""),
                "pages": article.get("pages", ""),
                "doi": article.get("doi", ""),
                "pmid": pmid,
                "pmcid": "",
            }

            results.append({
                "evidence_id": f"PMID:{pmid}",
                "source_db": "pubmed",
                "evidence_type": ev_type,
                "title": article["title"],
                "abstract_snippet": article["abstract"],
                "publication_date": article["pub_date"],
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "source_client": "PubMedClient",
                "query_params": {"term": target, "retmax": max_results},
                "raw_id": pmid,
                "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "source_metadata": source_metadata,
            })

        self._set_cached(key, _json.dumps(results))
        return results

class EuropePMCClient(BaseClient):
    def __init__(self, cache_dir: str = "data/cache"):
        super().__init__("https://www.ebi.ac.uk/europepmc/webservices/rest",
                         rate_limit=5, cache_dir=cache_dir)
    async def search(self, target: str, cutoff: str = None, max_results: int = 30) -> list[dict]:
        await self._wait_rate()
        params = {"query": target, "format": "json", "resultType": "core", "pageSize": max_results}
        key = self._cache_key(params)
        cached = self._get_cached(key)
        if cached:
            return _json.loads(cached)
        client = await _get_http_client(self.base_url, timeout=30)
        resp = await _request_with_retry(
            client, "GET", f"{self.base_url}/search", params=params)
        resp.raise_for_status()
        data = resp.json()
        results = []
        for r in data.get("resultList", {}).get("result", []):
            pmid = r.get("pmid", "")
            pmcid = r.get("pmcid", "")
            raw_id = pmid if pmid else pmcid
            source_url = f"https://europepmc.org/article/{r.get('source','')}/{r.get('id','')}"
            pub_types = str(r.get("pubType", [])).lower()
            title = r.get("title", "")
            abstract = r.get("abstractText", "")
            combined = (title + " " + abstract).lower()
            if "randomized controlled trial" in combined:
                ev_type = "RCT_human"
            elif "review" in pub_types:
                ev_type = "review"
            else:
                ev_type = "literature"
            # Parse author string
            authors = []
            author_str = r.get("authorString", "")
            if author_str:
                authors = [a.strip() for a in author_str.split(",") if a.strip()]

            # Extract year
            pub_year = None
            fp_date = r.get("firstPublicationDate", "")
            if fp_date:
                import re
                m = re.search(r"\b(19|20)\d{2}\b", fp_date)
                if m:
                    pub_year = int(m.group(0))

            source_metadata = {
                "source_type": "literature",
                "authors": authors,
                "first_author": r.get("firstAuthor", authors[0] if authors else ""),
                "year": pub_year,
                "journal": r.get("journalTitle", ""),
                "journal_short": r.get("journalAbbreviation", ""),
                "volume": r.get("volume", ""),
                "issue": r.get("issue", ""),
                "pages": r.get("pageInfo", ""),
                "doi": r.get("doi", ""),
                "pmid": pmid,
                "pmcid": pmcid,
            }

            results.append({
                "evidence_id": f"EPMC:{raw_id}",
                "source_db": "europe_pmc",
                "evidence_type": ev_type,
                "title": title,
                "abstract_snippet": abstract,
                "publication_date": r.get("firstPublicationDate", ""),
                "url": source_url,
                "source_client": "EuropePMCClient",
                "query_params": {"query": target, "pageSize": max_results},
                "raw_id": raw_id,
                "source_url": source_url,
                "source_metadata": source_metadata,
            })
        self._set_cached(key, _json.dumps(results))
        return results

class ChEMBLClient(BaseClient):
    def __init__(self, cache_dir: str = "data/cache"):
        super().__init__("https://www.ebi.ac.uk/chembl/api/data",
                         rate_limit=5, cache_dir=cache_dir)
    async def search(self, target: str, cutoff: str = None, **kwargs) -> list[dict]:
        await self._wait_rate()
        params = {"query": target, "only_active": "true", "limit": 20}
        key = self._cache_key(params)
        cached = self._get_cached(key)
        if cached:
            return _json.loads(cached)
        client = await _get_http_client(self.base_url, timeout=60)
        resp = await _request_with_retry(
            client, "GET", f"{self.base_url}/target.json", params=params)
        resp.raise_for_status()
        data = resp.json()
        results = []
        for t in data.get("targets", [])[:10]:
            # ChEMBL API v3 returns snake_case field names
            chembl_id = t.get("target_chembl_id", "")
            pref_name = t.get("pref_name", "")
            description = t.get("description", "")
            source_url = f"https://www.ebi.ac.uk/chembl/target/{chembl_id}"
            results.append({
                "evidence_id": f"ChEMBL:{chembl_id}",
                "source_db": "chembl",
                "evidence_type": "in_vitro",
                "title": pref_name,
                "abstract_snippet": description,
                "url": source_url,
                "source_client": "ChEMBLClient",
                "query_params": params,
                "raw_id": chembl_id,
                "source_url": source_url,
                "source_metadata": {
                    "source_type": "database",
                    "database_name": "ChEMBL",
                    "record_id": chembl_id,
                    "record_url": source_url,
                    "confidence_note": "Curated bioactivity data",
                },
            })
        self._set_cached(key, _json.dumps(results))
        return results

class KEGGClient(BaseClient):
    def __init__(self, cache_dir: str = "data/cache"):
        super().__init__("https://rest.kegg.jp",
                         rate_limit=5, cache_dir=cache_dir)
    async def search(self, target: str, cutoff: str = None, **kwargs) -> list[dict]:
        await self._wait_rate()
        key = self._cache_key({"target": target})
        cached = self._get_cached(key)
        if cached:
            return _json.loads(cached)
        client = await _get_http_client(self.base_url, timeout=30)
        resp = await _request_with_retry(
            client, "GET", f"{self.base_url}/find/genes/{target}")
        resp.raise_for_status()
        lines = resp.text.strip().split("\n")
        results = []
        target_lower = target.lower()
        for line in lines[:10]:
            parts = line.split("\t")
            if len(parts) >= 2:
                kegg_id = parts[0]
                desc = parts[1]
                # Filter: only keep entries related to the target gene
                if target_lower not in desc.lower() and target_lower not in kegg_id.lower():
                    continue
                source_url = f"https://www.kegg.jp/entry/{kegg_id}"
                results.append({
                    "evidence_id": f"KEGG:{kegg_id}",
                    "source_db": "kegg",
                    "evidence_type": "expert_curation",
                    "title": desc,
                    "abstract_snippet": "",
                    "url": source_url,
                    "source_client": "KEGGClient",
                    "query_params": {"target": target},
                    "raw_id": kegg_id,
                    "source_url": source_url,
                    "source_metadata": {
                        "source_type": "database",
                        "database_name": "KEGG",
                        "record_id": kegg_id,
                        "record_url": source_url,
                        "confidence_note": "KEGG pathway/gene annotation",
                    },
                })
        self._set_cached(key, _json.dumps(results))
        return results

class UniProtClient(BaseClient):
    def __init__(self, cache_dir: str = "data/cache"):
        super().__init__("https://rest.uniprot.org/uniprotkb",
                         rate_limit=5, cache_dir=cache_dir)
    async def search(self, target: str, **kwargs) -> list[dict]:
        await self._wait_rate()
        key = self._cache_key({"target": target})
        cached = self._get_cached(key)
        if cached:
            return _json.loads(cached)
        client = await _get_http_client(self.base_url, timeout=30)
        resp = await _request_with_retry(
            client, "GET", f"{self.base_url}/search", params={
                "query": f"gene:{target} AND reviewed:true",
                "format": "json", "size": 5,
            })
        resp.raise_for_status()
        data = resp.json()
        results = []
        for entry in data.get("results", [])[:5]:
            accession = entry.get("primaryAccession", "")
            source_url = f"https://www.uniprot.org/uniprotkb/{accession}/entry"
            comments = entry.get("comments", [])
            snippet = ""
            if comments:
                texts = comments[0].get("texts", [{}])
                if texts:
                    snippet = texts[0].get("value", "")
            results.append({
                "evidence_id": f"UniProt:{accession}",
                "source_db": "uniprot",
                "evidence_type": "expert_curation",
                "title": entry.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", ""),
                "abstract_snippet": snippet,
                "url": source_url,
                "source_client": "UniProtClient",
                "query_params": {"gene": target, "reviewed": "true"},
                "raw_id": accession,
                "source_url": source_url,
                "source_metadata": {
                    "source_type": "database",
                    "database_name": "UniProt",
                    "record_id": accession,
                    "record_url": source_url,
                    "confidence_note": "Manually reviewed protein annotation",
                },
            })
        self._set_cached(key, _json.dumps(results))
        return results

class OpenTargetsClient(BaseClient):
    def __init__(self, cache_dir: str = "data/cache"):
        super().__init__("https://api.platform.opentargets.org/api/v4/graphql",
                         rate_limit=2, cache_dir=cache_dir)
    async def search(self, target: str, ensembl_id: str = None, **kwargs) -> list[dict]:
        await self._wait_rate()
        client = await _get_http_client(self.base_url, timeout=60)
        if not ensembl_id:
            search_query = f'{{ search(queryString: "{target}", entityNames: ["target"], page: {{index: 0, size: 1}}) {{ total hits {{ id name entity }} }} }}'
            search_resp = await _request_with_retry(
                client, "POST", self.base_url, json={"query": search_query})
            search_resp.raise_for_status()
            search_data = search_resp.json()
            hits = search_data.get("data", {}).get("search", {}).get("hits", [])
            if not hits:
                return []
            ensembl_id = hits[0].get("id", "")
            if not ensembl_id:
                return []
        query = f'{{ target(ensemblId: "{ensembl_id}") {{ id approvedSymbol associatedDiseases {{ rows {{ disease {{ id name }} score }} }} }} }}'
        key = self._cache_key({"query": query})
        cached = self._get_cached(key)
        if cached:
            return _json.loads(cached)
        resp = await _request_with_retry(
            client, "POST", self.base_url, json={"query": query})
        resp.raise_for_status()
        data = resp.json()
        results = []
        target_data = data.get("data", {}).get("target", {}) or {}
        target_id = target_data.get("id", "")
        symbol = target_data.get("approvedSymbol", "")
        for row in target_data.get("associatedDiseases", {}).get("rows", []):
            disease = row.get("disease", {})
            disease_id = disease.get("id", "")
            source_url = f"https://platform.opentargets.org/target/{target_id}"
            results.append({
                "evidence_id": f"OT:{target_id}:{disease_id}",
                "source_db": "opentargets",
                "evidence_type": "database_association",
                "title": f"{symbol} → {disease.get('name','')}",
                "abstract_snippet": f"Association score: {row.get('score', 0):.3f}",
                "url": source_url,
                "source_client": "OpenTargetsClient",
                "query_params": {"approvedSymbol": target, "ensemblId": ensembl_id},
                "raw_id": target_id,
                "source_url": source_url,
                "source_metadata": {
                    "source_type": "database",
                    "database_name": "OpenTargets",
                    "record_id": f"{target_id}:{disease_id}",
                    "record_url": source_url,
                    "confidence_note": f"Association score: {row.get('score', 0):.3f}",
                },
            })
        self._set_cached(key, _json.dumps(results))
        return results

class GWASCatalogClient(BaseClient):
    def __init__(self, cache_dir: str = "data/cache"):
        super().__init__("https://www.ebi.ac.uk/gwas/rest/api",
                         rate_limit=5, cache_dir=cache_dir)
    async def search(self, target: str, disease: str = None, max_results: int = 20, **kwargs) -> list[dict]:
        await self._wait_rate()
        if disease:
            params = {"diseaseTrait": disease, "size": max_results}
        else:
            return []
        key = self._cache_key(params)
        cached = self._get_cached(key)
        if cached:
            return _json.loads(cached)
        client = await _get_http_client(self.base_url, timeout=30)
        resp = await _request_with_retry(
            client, "GET", f"{self.base_url}/associations", params=params)
        resp.raise_for_status()
        data = resp.json()
        results = []
        for a in data.get("_embedded", {}).get("associations", [])[:10]:
            accession = a.get("accessionId", "")
            source_url = f"https://www.ebi.ac.uk/gwas/studies/{accession}"
            results.append({
                "evidence_id": f"GWAS:{accession}",
                "source_db": "gwas",
                "evidence_type": "gwas",
                "title": a.get("diseaseTrait", ""),
                "abstract_snippet": a.get("initialSampleSize", ""),
                "url": source_url,
                "source_client": "GWASCatalogClient",
                "query_params": params,
                "raw_id": accession,
                "source_url": source_url,
                "source_metadata": {
                    "source_type": "database",
                    "database_name": "GWAS Catalog",
                    "record_id": accession,
                    "record_url": source_url,
                    "confidence_note": a.get("initialSampleSize", ""),
                },
            })
        self._set_cached(key, _json.dumps(results))
        return results

class ClinicalTrialsClient(BaseClient):
    def __init__(self, cache_dir: str = "data/cache"):
        super().__init__("https://clinicaltrials.gov/api/v2",
                         rate_limit=5, cache_dir=cache_dir)
    async def search(self, target: str, cutoff: str = None, disease: str = None, **kwargs) -> list[dict]:
        await self._wait_rate()
        if disease:
            params = {"query.cond": disease, "pageSize": 20}
        else:
            params = {"query.intr": target, "pageSize": 20}
        key = self._cache_key(params)
        cached = self._get_cached(key)
        if cached:
            return _json.loads(cached)
        client = await _get_http_client(self.base_url, timeout=30)
        resp = await _request_with_retry(
            client, "GET", f"{self.base_url}/studies", params=params)
        resp.raise_for_status()
        data = resp.json()
        results = []
        for s in data.get("studies", [])[:10]:
            protocol = s.get("protocolSection", {})
            nct_id = protocol.get("identificationModule", {}).get("nctId", "")
            source_url = f"https://clinicaltrials.gov/study/{nct_id}"
            design = protocol.get("designModule", {})
            allocation = design.get("allocation", "").upper()
            if allocation == "RANDOMIZED":
                ev_type = "RCT_human"
            else:
                ev_type = "clinical_trial"
            results.append({
                "evidence_id": f"CT:{nct_id}",
                "source_db": "clinicaltrials",
                "evidence_type": ev_type,
                "title": protocol.get("identificationModule", {}).get("briefTitle", ""),
                "abstract_snippet": protocol.get("descriptionModule", {}).get("briefSummary", ""),
                "publication_date": protocol.get("statusModule", {}).get("studyFirstSubmitDate", ""),
                "url": source_url,
                "source_client": "ClinicalTrialsClient",
                "query_params": params,
                "raw_id": nct_id,
                "source_url": source_url,
                "source_metadata": {
                    "source_type": "database",
                    "database_name": "ClinicalTrials.gov",
                    "record_id": nct_id,
                    "record_url": source_url,
                    "study_phase": protocol.get("designModule", {}).get("phases", [""])[0] if protocol.get("designModule", {}).get("phases") else "",
                    "confidence_note": f"Allocation: {design.get('allocation', 'N/A')}",
                },
            })
        self._set_cached(key, _json.dumps(results))
        return results

class BioRxivClient(BaseClient):
    def __init__(self, cache_dir: str = "data/cache"):
        super().__init__("https://api.biorxiv.org",
                         rate_limit=3, cache_dir=cache_dir)
    async def search(self, target: str, cutoff: str = None, **kwargs) -> list[dict]:
        # BioRxiv retrieval is disabled because the configured endpoint is unavailable.
        return []

class MONDOClient(BaseClient):
    def __init__(self, cache_dir: str = "data/cache"):
        super().__init__("https://www.ebi.ac.uk/ols4/api",
                         rate_limit=5, cache_dir=cache_dir)
    async def search(self, target: str, **kwargs) -> list[dict]:
        await self._wait_rate()
        key = self._cache_key({"target": target})
        cached = self._get_cached(key)
        if cached:
            return _json.loads(cached)
        client = await _get_http_client(self.base_url, timeout=30)
        resp = await _request_with_retry(
            client, "GET", f"{self.base_url}/search", params={
                "q": target, "ontology": "mondo", "size": 20,
            })
        resp.raise_for_status()
        data = resp.json()
        results = []
        for doc in data.get("response", {}).get("docs", [])[:10]:
            iri = doc.get("iri", "")
            source_url = iri if iri else ""
            results.append({
                "evidence_id": f"MONDO:{iri}",
                "source_db": "mondo",
                "evidence_type": "expert_curation",
                "title": doc.get("label", ""),
                "abstract_snippet": doc.get("description", [""])[0] if doc.get("description") else "",
                "url": source_url,
                "source_client": "MONDOClient",
                "query_params": {"q": target, "ontology": "mondo"},
                "raw_id": iri,
                "source_url": source_url,
                "source_metadata": {
                    "source_type": "database",
                    "database_name": "MONDO",
                    "record_id": iri,
                    "record_url": source_url,
                    "confidence_note": "Disease ontology term",
                },
            })
        self._set_cached(key, _json.dumps(results))
        return results
