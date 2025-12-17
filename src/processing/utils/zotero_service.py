# File: src/processing/utils/zotero_service.py
"""
ZoteroService - Phase 3.6 (Final)
Consistent, canonical metadata output for Intelligent Merge 2.2

Key Guarantees:
- Always returns canonical metadata dict:
      { title, authors, doi, journal, year, publisher, payload }
- Always returns confidence + provenance + source
- Safe DOI-first lookup
- Safe fallback title/author search
- Fully compatible with merge_metadata() and Multi-Referee system
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
from .author_utils import _fuzzy_score

try:
    from pyzotero import zotero
    PYZOTERO_AVAILABLE = True
except ImportError:
    PYZOTERO_AVAILABLE = False


# ============================================================
# Utility: DOI normalization for fuzzy compare
# ============================================================

def normalize_doi_for_compare(doi: Optional[str]) -> str:
    if not doi:
        return ""
    d = doi.strip().lower()
    for prefix in ["https://doi.org/", "http://doi.org/", "doi:"]:
        if d.startswith(prefix):
            d = d[len(prefix):]
    return d.strip()

# ============================================================
# Disk-backed JSON Cache
# ============================================================

class DiskCache:
    """
    Simple disk-backed JSON cache storing one file per key.
    """

    def __init__(self, cache_dir: str, ttl_seconds: int = 86400 * 30):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(seconds=int(ttl_seconds))

    def _key_to_path(self, key: str) -> Path:
        safe = "".join(
            c if c.isalnum() or c in "-_." else "_"
            for c in key
        )[:260]
        return self.cache_dir / f"{safe}.json"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        p = self._key_to_path(key)
        if not p.exists():
            return None

        try:
            mtime = datetime.fromtimestamp(p.stat().st_mtime)
            if datetime.utcnow() - mtime > self.ttl:
                # expired
                try:
                    p.unlink()
                except Exception:
                    pass
                return None

            with p.open("r", encoding="utf8") as fh:
                return json.load(fh)

        except Exception as e:
            logging.getLogger(__name__).warning(f"DiskCache.get failed for {p}: {e}")
            return None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        p = self._key_to_path(key)
        try:
            with p.open("w", encoding="utf8") as fh:
                json.dump(value, fh, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.getLogger(__name__).warning(f"DiskCache.set failed for {p}: {e}")

    def list_keys(self, prefix: Optional[str] = None) -> List[str]:
        out = []
        for f in self.cache_dir.glob("*.json"):
            name = f.stem
            if prefix and not name.startswith(prefix):
                continue
            out.append(name)
        return out

    def inspect(self, prefix: Optional[str] = None) -> List[Dict[str, Any]]:
        results = []
        for name in self.list_keys(prefix=prefix):
            val = self.get(name)
            results.append({"key": name, "value": val})
        return results


# ============================================================
# Canonicalization helper
# ============================================================

def _canonicalize(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts Zotero 'data' or 'payload' dict into the canonical metadata format
    required by Intelligent Merge 2.2 (Phase 3.6).
    """
    if not isinstance(payload, dict):
        payload = {}

    creators = payload.get("creators", [])
    authors = []
    for c in creators:
        if isinstance(c, dict):
            first = c.get("firstName", "")
            last = c.get("lastName", "")
            authors.append(f"{first} {last}".strip())

    return {
        "title": payload.get("title"),
        "authors": authors,
        "doi": payload.get("DOI") or payload.get("doi"),
        "journal": payload.get("publicationTitle") or payload.get("journal"),
        "year": payload.get("date") or payload.get("year"),
        "publisher": payload.get("publisher"),
        "payload": payload,
    }


# ============================================================
# ZoteroService (Final Phase 3.6)
# ============================================================

class ZoteroService:
    """
    High-level Zotero metadata retrieval service.
    Returns fully canonical dicts for Intelligent Merge 2.2.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, cache_dir: Optional[str] = None):
        if not PYZOTERO_AVAILABLE:
            raise ImportError("pyzotero library is required for ZoteroService.")

        self.logger = logging.getLogger(__name__)
        config = config or {}

        self.enabled = config.get("enabled", True)
        self.offline_mode = config.get("offline_mode", False)

        # Credentials
        self.api_key = os.getenv("ZOTERO_API_KEY") or config.get("api_key")
        self.library_id = os.getenv("ZOTERO_LIBRARY_ID") or config.get("library_id")
        self.library_type = config.get("library_type", "user")

        if not self.library_id:
            raise ValueError("ZoteroService: Missing 'library_id'.")

        if not self.api_key and not self.offline_mode:
            self.logger.warning("Missing Zotero API key; service will operate in degraded mode.")

        # Cache
        self.cache = DiskCache(cache_dir or "./zotero_cache")
        self.max_search = int(config.get("max_search_results", 10))

        # Initialize API client
        self.zotero_api = None
        if self.enabled and not self.offline_mode and self.api_key:
            try:
                self.zotero_api = zotero.Zotero(self.library_id, self.library_type, self.api_key)
                self.zotero_api.key_info()  # verify credentials
                self.logger.info(f"Connected to Zotero library '{self.library_id}'.")
            except Exception as e:
                self.logger.error(f"Failed to authenticate with Zotero API: {e}")
                self.zotero_api = None

        elif self.offline_mode:
            self.logger.info("ZoteroService is in OFFLINE MODE.")
        else:
            self.logger.info("ZoteroService disabled or missing credentials.")

    # -----------------------------------------------------------
    # Internal: Safe search wrapper
    # -----------------------------------------------------------

    def _search_items(self, query: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if not query or self.offline_mode or not self.zotero_api:
            return []

        try:
            items = self.zotero_api.items(q=query, limit=limit or self.max_search)
            return items
        except Exception as e:
            self.logger.warning(f"Zotero API search failed for query '{query[:50]}': {e}")
            return []

    # -----------------------------------------------------------
    # Public API
    # -----------------------------------------------------------

    def get_canonical_metadata(self, vlm_metadata: Any) -> Dict[str, Any]:
        """
        Final, Hardened Version.
        Performs a multi-stage search: DOI-first, then a series of targeted
        text searches to maximize the chance of finding a match.
        """
        # --- Input Normalization ---
        if isinstance(vlm_metadata, str):
            vlm_metadata = {"title": vlm_metadata}
        vlm_metadata = vlm_metadata or {}
        doi = vlm_metadata.get("doi") or vlm_metadata.get("DOI")
        title = vlm_metadata.get("title")
        authors = vlm_metadata.get("authors", [])

        # --- Offline Mode ---
        if self.offline_mode:
            self.logger.info("Zotero in offline mode, returning VLM data.")
            return {"record": vlm_metadata, "confidence": 0.70, "source": "vlm_extraction", "provenance": {"source_key": "offline_vlm"}}

        # --- Path 1: DOI-First Search (Highest Confidence) ---
        if doi:
            normalized_input_doi = normalize_doi_for_compare(doi)
            cache_key = f"doi_{normalized_input_doi}"

            # Cache hit?
            cached = self.cache.get(cache_key)
            if cached:
                self.logger.info(f"Zotero cache hit for DOI: {normalized_input_doi}")
                
                # The cache already has a canonicalized record with authors
                record = cached.get("record")
                
                if not record:
                    # Fallback: try to canonicalize payload if record missing
                    payload = cached.get("payload", {})
                    record = _canonicalize(payload)
                
                return {
                    "record": record,
                    "confidence": cached.get("confidence", 0.95),
                    "provenance": cached.get("provenance", {"source_key": cache_key}),
                    "source": cached.get("source", "zotero_cache")
                }

            # Query Zotero by DOI
            items = self._search_items(doi, limit=5)
            for item in items:
                data = item.get("data", {})
                item_doi = data.get("DOI") or data.get("doi")
                normalized_item = normalize_doi_for_compare(item_doi or "")

                if normalized_input_doi and normalized_input_doi in normalized_item:
                    self.logger.info(f"Zotero API match found for DOI: {normalized_input_doi}")
                    record = _canonicalize(data)
                    provenance = {"source_key": cache_key}

                    # Save to cache
                    self.cache.set(cache_key, {
                        "record": record,
                        "confidence": 0.95,
                        "provenance": provenance,
                        "source": "zotero_api"
                    })

                    return {
                        "record": record,
                        "confidence": 0.95,
                        "provenance": provenance,
                        "source": "zotero_api"
                    }

        # --- Path 2: Multi-Stage Text Search (Fuzzy Fallback) ---
        if title:
            # Strategy 1: Search by Title AND First Author's Last Name (high precision)
            first_author_lastname = ""
            if authors:
                # A simple way to get the last name from "First Last" or "Last, First"
                name_parts = authors[0].replace(",", " ").split()
                if name_parts:
                    first_author_lastname = name_parts[-1]

            if first_author_lastname:
                query = f"{title} {first_author_lastname}"
                self.logger.info(f"Zotero fuzzy search (stage 1: title+author): '{query[:70]}...'")
                items = self._search_items(query)
                if items:
                    # If we get any results with this precise query, we can be very confident.
                    # We'll trust the first result.
                    best_match = items[0].get("data", {})
                    record = _canonicalize(best_match)
                    provenance = {"source_key": f"fuzzy_{query[:50]}"}
                    # Cache the result under its DOI if it has one
                    if record.get('doi'):
                        self.cache.set(f"doi_{normalize_doi_for_compare(record.get('doi'))}", {"record": record, "confidence": 0.85, "provenance": provenance, "source": "zotero_fuzzy_match"})
                    return {"record": record, "confidence": 0.85, "source": "zotero_fuzzy_match", "provenance": provenance}

            # Strategy 2: Search by Title Only (broader search)
            self.logger.info(f"Zotero fuzzy search (stage 2: title only): '{title[:70]}...'")
            items = self._search_items(title)
            if items:
                # Find the best fuzzy match from the results
                best_item, best_score = None, 0
                for item in items:
                    item_title = item.get("data", {}).get("title", "")
                    score = _fuzzy_score(title, item_title) # Assumes _fuzzy_score exists
                    if score > best_score:
                        best_score, best_item = score, item
                
                if best_item and best_score >= 85: # High confidence threshold
                    data = best_item.get("data", {})
                    
                    # DEBUG: Log what we're getting from Zotero
                    self.logger.debug(f"Zotero fuzzy match data keys: {data.keys()}")
                    self.logger.debug(f"Zotero creators field: {data.get('creators', 'NOT FOUND')}")
                    
                    record = _canonicalize(data)
                    provenance = {"source_key": f"fuzzy_{title[:50]}"}
                    if record.get('doi'):
                        self.cache.set(f"doi_{normalize_doi_for_compare(record.get('doi'))}", {"record": record, "confidence": 0.85, "provenance": provenance, "source": "zotero_fuzzy_match"})
                    self.logger.info(f"Found high-confidence fuzzy match (Score: {best_score}) for title.")
                    return {"record": record, "confidence": 0.85, "source": "zotero_fuzzy_match", "provenance": provenance}

        # --- Path 3: Final Fallback ---
        self.logger.warning(f"Zotero could not find a canonical record for '{title[:50]}...'. Falling back to VLM data.")
        return {"record": vlm_metadata, "confidence": 0.70, "source": "vlm_extraction", "provenance": {"source_key": "vlm_fallback"}}