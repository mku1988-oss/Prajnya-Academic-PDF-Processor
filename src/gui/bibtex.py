#!/usr/bin/env python3
"""
BibTeX export for ledger records — extracted from V5's `ledger_master.py`
(the export logic was buried at lines 1138-1337 of one 1591-line page file;
see `docs/gui_plan.md` section 4).

For biology researchers: this turns processed-paper metadata into a `.bib`
file a reference manager (Zotero, EndNote, LaTeX) can import directly.

**One adaptation beyond a straight port.** V5 had no book support, so
`detect_entry_type` always returned `"article"`. V6's `LedgerRecord` carries
`document_type` (session 9's type-specific completeness rule), so this
module picks the BibTeX entry type from it — `@book` for a book, `@incollection`
for a book chapter, `@article` otherwise — and uses `editor={...}` instead of
`author={...}` for a book (which is cited by its editors, not authors; see
CLAUDE.md's per-chapter-authorship history).
"""

from __future__ import annotations

import re
from typing import List

from src.processing.ledger.semantic_ledger import LedgerRecord

_ACCENTS = {
    "ä": "a", "ö": "o", "ü": "u", "ß": "ss",
    "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
    "à": "a", "è": "e", "ì": "i", "ò": "o", "ù": "u",
    "â": "a", "ê": "e", "î": "i", "ô": "o", "û": "u",
    "ã": "a", "õ": "o", "ñ": "n", "ç": "c",
}

#: Order matters — the backslash escape must go first, or later escapes
#: that themselves insert a backslash would get re-escaped.
_BIBTEX_ESCAPES = [
    ("\\", r"\\"), ("{", r"\{"), ("}", r"\}"), ("$", r"\$"),
    ("%", r"\%"), ("&", r"\&"), ("_", r"\_"), ("#", r"\#"),
]

_ENTRY_TYPES = {"book": "book", "book_chapter": "incollection"}


def build_bibtex(records: List[LedgerRecord]) -> str:
    """Formats a list of ledger records as one `.bib` file's contents.

    Duplicate cite keys (e.g. two 2023 papers by the same first author in
    the same journal) are disambiguated with a numeric suffix, in the
    order given.
    """
    used_keys: set = set()
    entries = [_format_entry(record, _unique_cite_key(record, used_keys))
               for record in records]
    return "\n\n".join(entries)


def generate_cite_key(record: LedgerRecord) -> str:
    """Cite key in the form `lastname_journal_year`, e.g. `smith_nature_2023`."""
    names = record.authors or record.editors
    if names:
        first = names[0]
        lastname = first.split(",")[0].strip() if "," in first else (
            first.split()[0] if " " in first else first
        )
        lastname = _sanitize(lastname)
    else:
        lastname = "anonymous"

    venue = record.journal or record.publisher
    venue_abbrev = _sanitize(venue.replace(" ", "")[:10]) if venue else "unknown"

    year = str(record.year) if record.year else "nd"
    return f"{lastname}_{venue_abbrev}_{year}"


def _unique_cite_key(record: LedgerRecord, used_keys: set) -> str:
    base = generate_cite_key(record)
    key = base
    counter = 2
    while key in used_keys:
        key = f"{base}_{counter}"
        counter += 1
    used_keys.add(key)
    return key


def _sanitize(text: str) -> str:
    """Lowercase, transliterate accents, strip anything but [a-z0-9_]."""
    if not text:
        return ""
    text = text.lower()
    for accented, plain in _ACCENTS.items():
        text = text.replace(accented, plain)
    return re.sub(r"[^a-z0-9_]", "", text)


def _escape(text: str) -> str:
    if not text:
        return ""
    for old, new in _BIBTEX_ESCAPES:
        text = text.replace(old, new)
    return text


def _format_names(names: List[str]) -> str:
    """BibTeX joins multiple names with ` and `, e.g. 'Smith, J. and Doe, A.'."""
    return " and ".join(names) if names else "Anonymous"


def _entry_type(record: LedgerRecord) -> str:
    return _ENTRY_TYPES.get(record.document_type or "", "article")


def _format_entry(record: LedgerRecord, cite_key: str) -> str:
    lines = [f"@{_entry_type(record)}{{{cite_key},"]
    lines.append(f"  title={{{_escape(record.title) or 'Untitled'}}},")

    # A book is cited by its editors; an article or chapter by its authors.
    if not record.authors and record.editors:
        lines.append(f"  editor={{{_escape(_format_names(record.editors))}}},")
    else:
        lines.append(f"  author={{{_escape(_format_names(record.authors))}}},")

    if record.journal:
        lines.append(f"  journal={{{_escape(record.journal)}}},")
    elif record.publisher:
        lines.append(f"  publisher={{{_escape(record.publisher)}}},")

    lines.append(f"  year={{{record.year or 'nd'}}},")
    lines.append(f"  doi={{{_escape(record.doi) if record.doi else 'notfound'}}},")
    lines.append("  note={Exported from Prajnya Studio}")
    lines.append("}")
    return "\n".join(lines)
