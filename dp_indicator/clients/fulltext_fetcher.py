"""Full-text fetcher with multi-strategy access and paywall fallback.

Strategies (in order):
1. Europe PMC full-text XML (open access subset)
2. PMC OA web page (HTML extraction)
3. Unpaywall API → OA PDF URL → download + extract
4. Semantic Scholar API → PDF URL → download + extract
5. Fallback: extended abstract (from DB API) with paywall flag

For paywalled papers, we:
- Record the paywall encounter with DOI/PMID for audit
- Use the best available abstract (from PubMed/EuropePMC API)
- Flag the evidence as "paywalled" so downstream agents know the limitation
- Optionally try author preprint/server lookup
"""
from __future__ import annotations

import asyncio
import re
import time
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class FullTextFetcher:
    """Multi-strategy full-text fetcher with graceful paywall handling."""

    # Noise indicators for filtering non-content HTML responses
    _NOISE_INDICATORS = [
        "Access Denied", "access denied", "Log in", "Sign in",
        "captcha", "CAPTCHA", "403 Forbidden", "404 Not Found",
        "Service Unavailable", "Server Error", "Please enable JavaScript",
        "redirecting", "Redirecting",
    ]

    def __init__(self, cache_dir: str = "data/cache/fulltext",
                 email: str = "bohrium-agent@dp.tech",
                 max_pdf_size_mb: int = 20,
                 pdf_timeout: int = 30):
        import os
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.email = email
        self.max_pdf_size_bytes = max_pdf_size_mb * 1024 * 1024
        self.pdf_timeout = pdf_timeout
        self._http_client: Optional[httpx.AsyncClient] = None
        # Track paywall encounters for audit
        self.paywall_log: list[dict] = []

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.pdf_timeout, connect=10),
                follow_redirects=True,
                limits=httpx.Limits(max_connections=5),
                headers={
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/pdf,application/xml,*/*",
                },
            )
        return self._http_client

    async def close(self):
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    # ── Public API ──

    async def fetch(self, ev: dict) -> dict:
        """Fetch full text for an evidence item.

        Returns dict with:
            - text: str (extracted text or extended abstract)
            - source: str (full_text_xml | full_text_html | full_text_pdf | extended_abstract | abstract_only)
            - paywalled: bool
            - pdf_url: str | None (URL where PDF was found, if applicable)
            - error: str | None
        """
        sm = ev.get("source_metadata", {})
        pmcid = sm.get("pmcid", "")
        pmid = sm.get("pmid", "")
        doi = sm.get("doi", "")

        # Strategy 1: Europe PMC full-text XML (best quality)
        if pmcid and pmcid.startswith("PMC"):
            result = await self._try_europepmc_xml(pmcid)
            if result["text"] and len(result["text"]) > 1000:
                return result

        # Strategy 2: PMC OA web page
        if pmcid and pmcid.startswith("PMC"):
            result = await self._try_pmc_html(pmcid)
            if result["text"] and len(result["text"]) > 1000:
                return result

        # Strategy 3: Unpaywall → PDF
        if doi:
            pdf_url = await self._try_unpaywall(doi)
            if pdf_url:
                text = await self._download_and_extract_pdf(pdf_url)
                if text and len(text) > 1000:
                    return {
                        "text": text, "source": "full_text_pdf",
                        "paywalled": False, "pdf_url": pdf_url, "error": None,
                    }

        # Strategy 4: Semantic Scholar → PDF
        if doi or pmid:
            pdf_url = await self._try_s2(doi, pmid)
            if pdf_url:
                text = await self._download_and_extract_pdf(pdf_url)
                if text and len(text) > 1000:
                    return {
                        "text": text, "source": "full_text_pdf",
                        "paywalled": False, "pdf_url": pdf_url, "error": None,
                    }

        # Strategy 5: Fallback — extended abstract with paywall flag
        abstract = ev.get("abstract_snippet", "")
        title = ev.get("title", "")
        combined = f"{title}\n\n{abstract}" if abstract else title

        # Determine if paywalled (we tried but failed)
        is_paywalled = bool(doi or pmcid)
        if is_paywalled:
            self.paywall_log.append({
                "evidence_id": ev.get("evidence_id", ""),
                "doi": doi,
                "pmid": pmid,
                "pmcid": pmcid,
                "reason": "all_strategies_failed",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })

        if len(combined) > 100:
            return {
                "text": combined, "source": "extended_abstract",
                "paywalled": is_paywalled, "pdf_url": None,
                "error": "paywall_fallback" if is_paywalled else None,
            }
        return {
            "text": "[No content available]", "source": "abstract_only",
            "paywalled": is_paywalled, "pdf_url": None,
            "error": "no_abstract",
        }

    # ── Strategy implementations ──

    async def _try_europepmc_xml(self, pmcid: str) -> dict:
        """Strategy 1: Europe PMC full-text XML for OA articles."""
        try:
            client = await self._get_client()
            url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
            resp = await client.get(url, timeout=15)
            if resp.status_code == 200 and len(resp.text) > 1000:
                text = self._extract_xml_text(resp.text)
                if text and len(text) > 500:
                    return {
                        "text": text, "source": "full_text_xml",
                        "paywalled": False, "pdf_url": None, "error": None,
                    }
        except Exception as e:
            logger.debug(f"EuropePMC XML failed for {pmcid}: {e}")
        return {"text": "", "source": "", "paywalled": False, "pdf_url": None, "error": None}

    async def _try_pmc_html(self, pmcid: str) -> dict:
        """Strategy 2: PMC OA HTML page."""
        try:
            client = await self._get_client()
            url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
            resp = await client.get(url, timeout=15)
            if resp.status_code != 200:
                return {"text": "", "source": "", "paywalled": False, "pdf_url": None, "error": None}
            text = resp.text
            # Noise filter
            if any(ind in text for ind in self._NOISE_INDICATORS):
                return {"text": "", "source": "", "paywalled": False, "pdf_url": None, "error": None}
            if "<article" in text or 'id="main-content"' in text:
                extracted = self._html_to_text(text)
                if len(extracted) > 500:
                    return {
                        "text": extracted, "source": "full_text_html",
                        "paywalled": False, "pdf_url": None, "error": None,
                    }
        except Exception as e:
            logger.debug(f"PMC HTML failed for {pmcid}: {e}")
        return {"text": "", "source": "", "paywalled": False, "pdf_url": None, "error": None}

    async def _try_unpaywall(self, doi: str) -> Optional[str]:
        """Strategy 3: Unpaywall API to find OA PDF URL."""
        try:
            client = await self._get_client()
            url = f"https://api.unpaywall.org/v2/{doi}?email={self.email}"
            resp = await client.get(url, timeout=10)
            if resp.status_code != 200:
                return None
            data = resp.json()
            # Try best_oa_location first, then any oa_locations
            best = data.get("best_oa_location", {})
            if best:
                pdf_url = best.get("url_for_pdf") or best.get("url")
                if pdf_url and self._is_likely_pdf_url(pdf_url):
                    return pdf_url
            # Fallback: scan all oa_locations
            for loc in data.get("oa_locations", []):
                pdf_url = loc.get("url_for_pdf") or loc.get("url")
                if pdf_url and self._is_likely_pdf_url(pdf_url):
                    return pdf_url
        except Exception as e:
            logger.debug(f"Unpaywall failed for {doi}: {e}")
        return None

    async def _try_s2(self, doi: str, pmid: str) -> Optional[str]:
        """Strategy 4: Semantic Scholar API to find OA PDF URL."""
        identifier = f"DOI:{doi}" if doi else f"PMID:{pmid}"
        try:
            client = await self._get_client()
            url = f"https://api.semanticscholar.org/graph/v1/paper/{identifier}?fields=openAccessPdf"
            resp = await client.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                pdf_info = data.get("openAccessPdf", {})
                if pdf_info and pdf_info.get("url"):
                    return pdf_info["url"]
            elif resp.status_code == 429:
                # Rate limited, wait and skip
                await asyncio.sleep(1)
        except Exception as e:
            logger.debug(f"Semantic Scholar failed for {identifier}: {e}")
        return None

    async def _download_and_extract_pdf(self, url: str) -> Optional[str]:
        """Download a PDF and extract text using pymupdf (fitz) or pdfplumber."""
        try:
            client = await self._get_client()
            resp = await client.get(
                url,
                timeout=self.pdf_timeout,
                headers={"Accept": "application/pdf"},
            )
            if resp.status_code != 200:
                return None
            content_type = resp.headers.get("content-type", "")
            if "pdf" not in content_type and not resp.content[:5] == b"%PDF-":
                # Not actually a PDF
                return None
            if len(resp.content) > self.max_pdf_size_bytes:
                logger.debug(f"PDF too large: {len(resp.content)} bytes")
                return None
            # Save to temp file and extract
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(resp.content)
                tmp_path = f.name
            try:
                text = self._extract_pdf_text(tmp_path)
                return text if text and len(text) > 200 else None
            finally:
                os.unlink(tmp_path)
        except Exception as e:
            logger.debug(f"PDF download/extract failed for {url}: {e}")
            return None

    # ── Text extraction helpers ──

    @staticmethod
    def _extract_xml_text(xml_text: str) -> str:
        """Extract readable text from XML (Europe PMC full-text format)."""
        # Remove tags but keep text content
        text = re.sub(r"<[^>]+>", " ", xml_text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:15000]

    @staticmethod
    def _html_to_text(html: str) -> str:
        """Basic HTML to text conversion."""
        # Remove scripts and styles
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:15000]

    @staticmethod
    def _extract_pdf_text(pdf_path: str) -> str:
        """Extract text from PDF using available libraries.

        Tries pymupdf (fitz) first (faster), falls back to pdfplumber.
        """
        # Try pymupdf (fitz) — faster and more robust
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text())
            doc.close()
            text = "\n".join(text_parts)
            if text.strip():
                return text.strip()[:15000]
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"PyMuPDF extraction failed: {e}")

        # Fallback: pdfplumber
        try:
            import pdfplumber
            text_parts = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    t = page.extract_text() or ""
                    text_parts.append(t)
            text = "\n".join(text_parts)
            if text.strip():
                return text.strip()[:15000]
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"pdfplumber extraction failed: {e}")

        logger.warning("No PDF extraction library available (install pymupdf or pdfplumber)")
        return ""

    @staticmethod
    def _is_likely_pdf_url(url: str) -> bool:
        """Heuristic check if URL likely points to a PDF.

        Conservative: only accept URLs that look like PDFs or are from known OA repos.
        Non-PDF URLs (HTML landing pages) waste a download round-trip.
        """
        url_lower = url.lower()
        # Direct PDF extension
        if url_lower.endswith(".pdf"):
            return True
        # Known OA repositories that serve PDF without .pdf extension
        oa_domains = (
            "europepmc.org/backend/ptpmcrender",
            "www.ncbi.nlm.nih.gov/pmc/articles",
            "arxiv.org/pdf",
            "biorxiv.org/content",
            "chemrxiv.org/engage/api-gateway",
            "doi.org/10."  # publisher direct, let content-type filter
        )
        if any(d in url_lower for d in oa_domains):
            return True
        return False

    @staticmethod
    def smart_truncate(text: str, max_chars: int = 6000) -> str:
        """Truncate text at sentence boundary."""
        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars]
        for i in range(len(truncated) - 1, max(0, len(truncated) - 500), -1):
            if truncated[i] in ".!?" and (i + 1 >= len(truncated) or truncated[i + 1] in " \n"):
                return truncated[:i + 1] + f"\n\n[... {len(text) - i - 1} characters truncated ...]"
        for i in range(len(truncated) - 1, max(0, len(truncated) - 100), -1):
            if truncated[i] == " ":
                return truncated[:i] + f"\n\n[... {len(text) - i} characters truncated ...]"
        return truncated + f"\n\n[... {len(text) - max_chars} characters truncated ...]"
