#!/usr/bin/env python3
"""
Did a document really finish? One answer, shared by every part that asks.

For biology researchers: a run can end "successfully" while part of the
document was lost. A page whose reading raised an error used to be written
empty, the file still said `success: true`, and the ledger marked the
document done, so the next run skipped it and nobody was told (audit D1;
PMC10904388 lost page 4 in every benchmark run this way, D9).

This module gives each document one of three statuses:

- "success": every page was read.
- "partial": the document was written, but some pages failed. Their page
  numbers are listed, and the document is NOT recorded as processed, so the
  next run tries it again.
- "failed": the strategy itself stopped with an error, or no page came out.

The document JSON (`document_info.status`), the batch summary, Prajnya
Studio and the ledger all read the status from here, so they cannot
disagree.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Sequence

#: The `processing_method` a page carries when its reading failed. The title
#: page's two-shot read and a body page's read each have their own.
FAILED_METHODS = ("two_shot_title_failed", "glm_failed")

SUCCESS, PARTIAL, FAILED = "success", "partial", "failed"


def failed_page_numbers(pages: Iterable[Any]) -> List[int]:
    """
    The 1-based numbers of the pages whose reading failed, in order.

    Args:
        pages: the document's pages, as `ContentPage` objects or as the
            dicts they become in the JSON.
    """
    return sorted(_page_number(page) for page in pages
                  if _page_metadata(page).get("processing_method") in FAILED_METHODS)


def document_status(errors: Sequence[str], page_count: int,
                    failed_pages: Sequence[int]) -> str:
    """
    "success", "partial" or "failed" for one processed document.

    Args:
        errors: the errors the strategy recorded (an exception that stopped it).
        page_count: how many pages came out.
        failed_pages: the pages whose reading failed (`failed_page_numbers`).
    """
    if errors or page_count == 0:
        return FAILED
    if failed_pages:
        return PARTIAL
    return SUCCESS


# --- helpers (after the main functions, per CLAUDE.md) ----------------------

def _page_metadata(page: Any) -> dict:
    """A page's metadata dict, whether the page is a model or a plain dict."""
    if isinstance(page, dict):
        return page.get("metadata") or {}
    return getattr(page, "metadata", None) or {}


def _page_number(page: Any) -> int:
    if isinstance(page, dict):
        return int(page.get("page_number") or 0)
    return int(getattr(page, "page_number", 0) or 0)
