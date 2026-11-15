#!/usr/bin/env python
"""Polite web requests for the citation services, with an on-disk cache.

For biology researchers: Crossref and OpenAlex are free public registries.
Asking them nicely means identifying ourselves, not asking too fast, and
asking once: every answer is kept on disk, so re-running the same corpus
sends nothing at all and gives the same output.

WHAT IS CACHED (plan C5). The raw answer, one file per request: the FACT
the registry told us. The tidy record is derived from it each time it is
read, so a fixed mapping is never hidden behind an old stored result.
200 and 404 answers are kept (404 means "no such DOI here", a real answer).
A timeout, a refused connection or a 5xx is NEVER kept: those say nothing
about the paper.

WHAT IS SENT. Only a DOI, or a title plus a first-author surname. Never page
text, never the file name, never a path (plan C5).

CERTIFICATES. A certificate failure falls back to the system `curl`, which
trusts the operating system's certificate store (many institutional
networks re-sign traffic). Verification is never switched off.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlencode

import requests

#: Outcome of a request, for the caller to branch on.
STATUS_OK = "ok"                    # HTTP 200, body is JSON
STATUS_NOT_FOUND = "not_found"      # HTTP 404: the registry has no such record
STATUS_UNREACHABLE = "unreachable"  # no answer (offline, timeout, 5xx, 429 after retries)

DEFAULT_USER_AGENT = "Prajnya/6 (scientific PDF processing; citation lookup)"


class CitationHttp:
    """One polite, caching HTTP client shared by the citation services."""

    def __init__(self, cache_dir: str | Path, pacing_seconds: float = 1.2,
                 timeout_seconds: float = 20.0, contact_email: Optional[str] = None,
                 retries_on_429: int = 2, logger: Optional[logging.Logger] = None):
        """
        Args:
            cache_dir: folder for the raw answers (created if missing).
            pacing_seconds: the least time between two live requests.
            timeout_seconds: per request.
            contact_email: put in the User-Agent when the researcher fills it
                in; otherwise the agent name is generic (nothing private).
            retries_on_429: how many times to wait and retry "slow down".
        """
        self.cache_dir = Path(cache_dir)
        self.pacing_seconds = float(pacing_seconds)
        self.timeout_seconds = float(timeout_seconds)
        self.retries_on_429 = int(retries_on_429)
        self.user_agent = (f"{DEFAULT_USER_AGENT}; mailto:{contact_email}"
                           if contact_email else DEFAULT_USER_AGENT)
        self.log = logger or logging.getLogger("citations.http")
        self._last_live_request = 0.0
        self.live_requests = 0     # for tests and the `citations` timing bucket

    def get_json(self, url: str, params: Optional[Dict[str, Any]] = None
                 ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        GET a JSON document, from the cache when it is there.

        Returns:
            `(status, body)`: STATUS_OK with the parsed body, STATUS_NOT_FOUND
            with None, or STATUS_UNREACHABLE with None.
        """
        full_url = _full_url(url, params)
        # The cache identity leaves out credentials: a changed key must not
        # discard good answers, and a key must never be written to disk.
        cache_url = _full_url(url, _without_secrets(params))
        cached = self._read_cache(cache_url)
        if cached is not None:
            return cached

        status, body = self._live_get(full_url)
        if status in (STATUS_OK, STATUS_NOT_FOUND):
            self._write_cache(cache_url, status, body)
        return status, body

    def forget(self, url: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Drop one cached answer (`--force` refreshes a document's own entries)."""
        path = self._cache_path(_full_url(url, _without_secrets(params)))
        if path.is_file():
            path.unlink()

    # --- helpers (after the main functions, per CLAUDE.md) -----------------

    def _live_get(self, full_url: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        """One live request, paced, retrying a 429 and falling back to curl."""
        for attempt in range(self.retries_on_429 + 1):
            self._wait_for_turn()
            try:
                response = requests.get(full_url, timeout=self.timeout_seconds,
                                        headers={"User-Agent": self.user_agent})
            except requests.exceptions.SSLError:
                return self._curl_get(full_url)
            except requests.exceptions.RequestException as error:
                self.log.warning("Citation request failed (%s).", type(error).__name__)
                return STATUS_UNREACHABLE, None
            finally:
                self._last_live_request = time.monotonic()
                self.live_requests += 1

            if response.status_code == 429 and attempt < self.retries_on_429:
                time.sleep(_retry_after(response) or 5.0)
                continue
            return _interpret(response.status_code, response.text)
        return STATUS_UNREACHABLE, None

    def _curl_get(self, full_url: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        """The certificate fallback: the system curl, which verifies normally."""
        curl = shutil.which("curl")
        if not curl:
            return STATUS_UNREACHABLE, None
        try:
            done = subprocess.run(
                [curl, "-sS", "-L", "-A", self.user_agent, "-w", "\n%{http_code}",
                 "--max-time", str(int(self.timeout_seconds)), full_url],
                capture_output=True, timeout=self.timeout_seconds + 5)
        except (OSError, subprocess.TimeoutExpired):
            return STATUS_UNREACHABLE, None
        text = done.stdout.decode("utf-8", errors="replace")
        body, _, code = text.rpartition("\n")
        return _interpret(int(code) if code.strip().isdigit() else 0, body)

    def _wait_for_turn(self) -> None:
        """Sleep until `pacing_seconds` have passed since the last live request."""
        wait = self.pacing_seconds - (time.monotonic() - self._last_live_request)
        if wait > 0 and self._last_live_request:
            time.sleep(wait)

    def _cache_path(self, full_url: str) -> Path:
        return self.cache_dir / (hashlib.sha1(full_url.encode("utf-8")).hexdigest() + ".json")

    def _read_cache(self, full_url: str) -> Optional[Tuple[str, Optional[Dict[str, Any]]]]:
        path = self._cache_path(full_url)
        if not path.is_file():
            return None
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            return stored["status"], stored.get("body")
        except (OSError, ValueError, KeyError):
            return None     # a damaged entry is simply asked for again

    def _write_cache(self, full_url: str, status: str, body: Optional[Dict[str, Any]]) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            payload = {"url": full_url, "status": status, "body": body}
            # newline="" so the file's line endings are exactly ours (CLAUDE.md trap).
            with open(self._cache_path(full_url), "w", encoding="utf-8", newline="") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False))
        except OSError as error:
            self.log.warning("Could not write the citation cache (%s).", error)


def _full_url(url: str, params: Optional[Dict[str, Any]]) -> str:
    """The URL with its parameters in a fixed order, so equal asks share one cache file."""
    if not params:
        return url
    return f"{url}?{urlencode(sorted((k, str(v)) for k, v in params.items()))}"


def _without_secrets(params: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The parameters minus anything that is a credential."""
    if not params:
        return params
    return {k: v for k, v in params.items() if k not in ("api_key", "mailto")}


def _interpret(status_code: int, text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Turn an HTTP answer into (status, body)."""
    if status_code == 200:
        try:
            return STATUS_OK, json.loads(text)
        except ValueError:
            return STATUS_UNREACHABLE, None
    if status_code == 404:
        return STATUS_NOT_FOUND, None
    return STATUS_UNREACHABLE, None


def _retry_after(response: "requests.Response") -> Optional[float]:
    """Seconds the server asked us to wait, capped at one minute."""
    try:
        return min(float(response.headers.get("Retry-After", "")), 60.0)
    except ValueError:
        return None
