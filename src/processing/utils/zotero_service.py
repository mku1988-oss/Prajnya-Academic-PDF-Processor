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

    # Internal: Safe search wrapper

    def _search_items(self, query: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if not query or self.offline_mode or not self.zotero_api:
            return []

        try:
            items = self.zotero_api.items(q=query, limit=limit or self.max_search)
            return items
        except Exception as e:
            self.logger.warning(f"Zotero API search failed for query '{query[:50]}': {e}")
            return []

    # Public API

    def get_canonical_metadata(self, vlm_metadata: Any) -> Dict[str, Any]:
        """
        Final, Hardened Version.
        Performs a multi-stage search: DOI-first, then a series of targeted
        text searches to maximize the chance of finding a match.
        """
        if isinstance(vlm_metadata, str):
            vlm_metadata = {"title": vlm_metadata}
        vlm_metadata = vlm_metadata or {}
        doi = vlm_metadata.get("doi") or vlm_metadata.get("DOI")
        title = vlm_metadata.get("title")
        authors = vlm_metadata.get("authors", [])

        if self.offline_mode:
            self.logger.info("Zotero in offline mode, returning VLM data.")
            return {"record": vlm_metadata, "confidence": 0.70, "source": "vlm_extraction", "provenance": {"source_key": "offline_vlm"}}

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

            # Look the DOI up in our local index.
            #
            # We CANNOT ask Zotero to search by DOI. The Web API's `q` parameter only
            # searches titles + creators (qmode=titleCreatorYear) or titles + creators +
            # attachment full-text (qmode=everything). The DOI field is not searchable in
            # either mode -- the API docs say "Searching of other fields will be possible
            # in the future". Passing a DOI as `q` therefore always returned 0 results and
            # this path silently fell through to the fuzzy title search.
            # See _find_item_data_by_doi() for how the local index is kept in sync.
            data = self._find_item_data_by_doi(normalized_input_doi)
            if data:
                self.logger.info(f"Zotero DOI index match for: {normalized_input_doi}")
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

        self.logger.warning(f"Zotero could not find a canonical record for '{title[:50]}...'. Falling back to VLM data.")
        return {"record": vlm_metadata, "confidence": 0.70, "source": "vlm_extraction", "provenance": {"source_key": "vlm_fallback"}}

    # Internal: local DOI index
    #
    # WHY THIS EXISTS: the Zotero Web API cannot search by DOI. Its `q` parameter searches
    # titles and creators (qmode=titleCreatorYear), optionally plus attachment full-text
    # (qmode=everything) -- never the DOI field. So to find "the library item whose DOI is
    # X" we must hold a DOI -> itemKey map ourselves.
    #
    # SCALING: we do NOT re-download the library on every run. Zotero versions the whole
    # library with a single integer. We store that version alongside the index; on the next
    # run we ask for `items(since=<stored_version>)`, which returns ONLY what changed.
    # A library of any size therefore costs one full walk once, then a single ~0.3s call
    # per run when nothing has changed.

    def _doi_index_path(self) -> Path:
        """Location of the on-disk DOI index (kept out of DiskCache, which expires by TTL)."""
        return self.cache.cache_dir / "doi_index.json"

    def _load_doi_index(self) -> Dict[str, Any]:
        """Read the stored index, or an empty one if absent/corrupt."""
        path = self._doi_index_path()
        if not path.exists():
            return {"library_version": None, "doi_to_key": {}}
        try:
            with path.open("r", encoding="utf8") as fh:
                stored = json.load(fh)
            if not isinstance(stored.get("doi_to_key"), dict):
                raise ValueError("malformed doi_to_key")
            return stored
        except Exception as e:
            self.logger.warning(f"DOI index unreadable ({e}); rebuilding from scratch.")
            return {"library_version": None, "doi_to_key": {}}

    def _save_doi_index(self, library_version: int, doi_to_key: Dict[str, str]) -> None:
        """Persist the index so the next run only needs to fetch changes."""
        try:
            with self._doi_index_path().open("w", encoding="utf8") as fh:
                json.dump(
                    {"library_version": library_version, "doi_to_key": doi_to_key},
                    fh, ensure_ascii=False, indent=2,
                )
        except Exception as e:
            self.logger.warning(f"Could not save DOI index: {e}")

    def _index_items(self, items: List[Dict[str, Any]], doi_to_key: Dict[str, str]) -> None:
        """
        Fold a batch of Zotero items into the DOI map (in place).

        An item that lost its DOI (or became an attachment) must also lose its index entry,
        so we first drop any existing entry pointing at this item key, then re-add if the
        item currently has a DOI.

        Duplicate entries for one paper are common in real libraries (this library has ~23).
        When two items share a DOI we keep the one added FIRST -- the original. Ordering by
        dateAdded (item key as tie-break) rather than by arrival order matters: a delta sync
        visits items in a different order than the initial full walk, so "keep whichever we
        saw first" would let the chosen item flip between runs.
        """
        for item in items:
            data = item.get("data") or {}
            item_key = data.get("key") or item.get("key")
            if not item_key:
                continue

            # Drop stale entries for this item (DOI may have changed or been removed).
            for existing_doi in [d for d, k in doi_to_key.items() if self._entry_key(k) == item_key]:
                doi_to_key.pop(existing_doi, None)

            normalized = normalize_doi_for_compare(data.get("DOI") or data.get("doi") or "")
            if not normalized:
                continue

            candidate = {"key": item_key, "dateAdded": data.get("dateAdded") or ""}
            incumbent = doi_to_key.get(normalized)

            if incumbent is None:
                doi_to_key[normalized] = candidate
                continue

            if self._entry_key(incumbent) == item_key:
                doi_to_key[normalized] = candidate  # same item, refreshed
                continue

            # Deterministic winner: earliest dateAdded, then lowest item key.
            incumbent_rank = (self._entry_date(incumbent), self._entry_key(incumbent))
            candidate_rank = (candidate["dateAdded"], item_key)
            winner = incumbent if incumbent_rank <= candidate_rank else candidate

            self.logger.debug(
                f"DOI {normalized} maps to duplicate items "
                f"({self._entry_key(incumbent)}, {item_key}); keeping {self._entry_key(winner)} "
                f"(earliest dateAdded)."
            )
            doi_to_key[normalized] = winner

    @staticmethod
    def _entry_key(entry: Any) -> str:
        """Item key from an index entry, tolerating the older plain-string index format."""
        if isinstance(entry, dict):
            return entry.get("key", "")
        return entry or ""

    @staticmethod
    def _entry_date(entry: Any) -> str:
        """dateAdded from an index entry; empty for the older plain-string format."""
        return entry.get("dateAdded", "") if isinstance(entry, dict) else ""

    def _sync_doi_index(self) -> Dict[str, str]:
        """
        Bring the local DOI index up to date with the library, and return the DOI -> key map.

        Costs one full library walk the first time, then only the changed items thereafter.
        Returns an empty map (never raises) if Zotero is unreachable, so callers fall back
        to the fuzzy title search.
        """
        if self.offline_mode or not self.zotero_api:
            return {}

        try:
            current_version = int(self.zotero_api.last_modified_version())
        except Exception as e:
            self.logger.warning(f"Could not read Zotero library version: {e}")
            return {}

        stored = self._load_doi_index()
        stored_version = stored.get("library_version")
        doi_to_key = dict(stored.get("doi_to_key") or {})

        try:
            if stored_version is None:
                self.logger.info("Building Zotero DOI index for the first time (one full pass)...")
                all_items = self.zotero_api.everything(self.zotero_api.items())
                self._index_items(all_items, doi_to_key)

            elif int(stored_version) < current_version:
                self.logger.info(
                    f"Updating Zotero DOI index: v{stored_version} -> v{current_version}"
                )
                changed = self.zotero_api.everything(self.zotero_api.items(since=int(stored_version)))
                self._index_items(changed, doi_to_key)

                # Items deleted since our last sync must leave the index too.
                deleted_keys = set(self.zotero_api.deleted(since=int(stored_version)).get("items", []))
                for doi in [d for d, k in doi_to_key.items() if self._entry_key(k) in deleted_keys]:
                    doi_to_key.pop(doi, None)

            else:
                # Library unchanged since last run -- nothing to fetch.
                return doi_to_key

        except Exception as e:
            self.logger.warning(f"Zotero DOI index sync failed: {e}. Using last known index.")
            return doi_to_key

        self._save_doi_index(current_version, doi_to_key)
        self.logger.info(f"Zotero DOI index holds {len(doi_to_key)} DOIs (library v{current_version}).")
        return doi_to_key

    def _find_item_data_by_doi(self, normalized_doi: str) -> Optional[Dict[str, Any]]:
        """
        Return the Zotero item `data` block whose DOI equals `normalized_doi`, or None.

        Matching is EXACT on the normalised DOI. (The previous code used a substring test,
        which would let "10.1/abc" match "10.1/abc123".)
        """
        if not normalized_doi:
            return None

        item_key = self._entry_key(self._sync_doi_index().get(normalized_doi))
        if not item_key:
            return None

        try:
            item = self.zotero_api.item(item_key)
            return item.get("data") or None
        except Exception as e:
            self.logger.warning(f"Could not fetch Zotero item {item_key}: {e}")
            return None