"""
The links a document carries, arranged for the Results screen.

For biology researchers: Prajnya links what a paper's text points at. A
mention such as "Fig. 3B" or "Table 2" is linked to the figure or table object
with that printed label (the `mentions` block, at every citation level); at
the `online` level each in-text citation ("[12]", "(Doi et al., 2005)") is
linked to its entry in the reference list (the `citations` block). This
module turns those blocks into what a reader wants to see:

- for a figure or table: every sentence that mentions it, with its page
  (following a mention one hop to its figure is how a search system finds
  the evidence for a claim, and this is the same hop seen from the figure);
- for a reference: every sentence that cites it;
- for a page: every link that starts on it.

Everything here is a pure function of the document JSON, so it is tested
without a running interface. Offsets in the JSON index each page's
`content`, and the sentences are cut from that same text.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# A passage longer than this is cut, with an ellipsis, around the link.
MAX_PASSAGE_CHARACTERS = 320
# Words after which a full stop does not end a sentence ("Fig. 3", "et al. 2005").
ABBREVIATIONS = {"fig", "figs", "al", "e.g", "i.e", "eq", "eqs", "ref", "refs", "no",
                 "vs", "approx", "ca", "cf", "dr", "st", "suppl", "sect", "vol"}


def mentions_by_object(document: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """figure_id or table_id -> every passage that mentions it, in reading order.

    A mention naming an object split into several crops lists each crop's id
    in its targets; the passage is filed under every one of them."""
    pages = _pages_by_number(document)
    found: Dict[str, List[Dict[str, Any]]] = {}
    for kind in ("figures", "tables"):
        for mention in (document.get("mentions") or {}).get(kind) or []:
            passage = _passage_for(pages, mention.get("page_number"),
                                   mention.get("content_offset"), mention.get("text") or "")
            for target in mention.get("targets") or []:
                found.setdefault(target, []).append(passage)
    return found


def citations_by_reference(document: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Each reference-list entry with the passages that cite it.

    Returns one row per entry, in list order: {"reference": entry, "passages":
    [...]}. Empty when the document has no reference list (it was not
    processed at the `online` level)."""
    citations = document.get("citations") or {}
    references = citations.get("references") or []
    if not references:
        return []
    pages = _pages_by_number(document)
    by_position: Dict[int, List[Dict[str, Any]]] = {}
    for marker in citations.get("markers") or []:
        passage = _marker_passage(pages, marker)
        for position in marker.get("numbers") or []:
            by_position.setdefault(position, []).append(passage)
    return [{"reference": entry, "passages": by_position.get(entry.get("position"), [])}
            for entry in references]


def links_on_page(document: Dict[str, Any], page_number: int) -> List[Dict[str, Any]]:
    """Every link that starts on one page, in reading order.

    Each row: {"kind": "figure" | "table" | "citation", "text": as printed,
    "targets": what it points at, "offset": where it stands (None if
    unknown)}. Targets are object ids for mentions and reference-list
    entries for citations."""
    rows: List[Dict[str, Any]] = []
    for kind, singular in (("figures", "figure"), ("tables", "table")):
        for mention in (document.get("mentions") or {}).get(kind) or []:
            if mention.get("page_number") == page_number:
                rows.append({"kind": singular, "text": mention.get("text") or "",
                             "targets": list(mention.get("targets") or []),
                             "offset": mention.get("content_offset")})
    citations = document.get("citations") or {}
    entry_at = {r.get("position"): r for r in citations.get("references") or []}
    # Markers are listed in reading order, but one a vision read wrote
    # differently has no offset; it is placed right after the marker before it.
    last_known = 0
    for marker in citations.get("markers") or []:
        if marker.get("page_number") != page_number:
            continue
        offset = marker.get("content_offset")
        last_known = offset if offset is not None else last_known
        rows.append({"kind": "citation", "text": marker.get("text") or "",
                     "targets": [entry_at[n] for n in marker.get("numbers") or []
                                 if n in entry_at],
                     "offset": offset, "_order": last_known})
    rows.sort(key=lambda row: row.get("_order", row["offset"] or 0))      # stable
    for row in rows:
        row.pop("_order", None)
    return rows


def reference_label(entry: Dict[str, Any], limit: Optional[int] = None) -> str:
    """A short name for a reference entry: '[12] Doi, 2005: title' (first author
    only: the entry does not say how many authors follow). With `limit`, cut
    at a word boundary with an ellipsis."""
    author = entry.get("first_author") or "Unknown"
    year = entry.get("year") or "n.d."
    title = entry.get("title") or entry.get("printed_text") or ""
    number = entry.get("number") or entry.get("position")
    prefix = f"[{number}] " if number is not None else ""
    label = f"{prefix}{author}, {year}: {title}".rstrip(": ")
    if limit is not None and len(label) > limit:
        label = label[:limit].rsplit(" ", 1)[0].rstrip(",:;") + " …"
    return label


def link_counts(document: Dict[str, Any]) -> Dict[str, int]:
    """How many figure mentions, table mentions and citations the document links."""
    mentions = document.get("mentions") or {}
    citations = document.get("citations") or {}
    return {"figures": sum(1 for m in mentions.get("figures") or [] if m.get("targets")),
            "tables": sum(1 for m in mentions.get("tables") or [] if m.get("targets")),
            "citations": sum(1 for m in citations.get("markers") or [] if m.get("numbers"))}


# ---------------------------------------------------------------------------
# Helpers (after the main functions, per the project rule)
# ---------------------------------------------------------------------------

def _pages_by_number(document: Dict[str, Any]) -> Dict[int, str]:
    return {page.get("page_number"): page.get("content") or ""
            for page in document.get("pages") or []}


def _passage_for(pages: Dict[int, str], page_number: Optional[int], offset: Optional[int],
                 printed: str) -> Dict[str, Any]:
    """{"page", "text", "sentence"}: the sentence around a link."""
    content = pages.get(page_number, "")
    sentence = sentence_around(content, offset) if offset is not None else ""
    return {"page": page_number, "text": printed, "sentence": sentence or printed}


def _marker_passage(pages: Dict[int, str], marker: Dict[str, Any]) -> Dict[str, Any]:
    """A citation's passage. A marker with no offset (a superscript a vision
    read wrote differently) is shown with the text printed just before it."""
    passage = _passage_for(pages, marker.get("page_number"), marker.get("content_offset"),
                           marker.get("text") or "")
    if marker.get("content_offset") is None and marker.get("context_before"):
        context = marker["context_before"]
        # The stored context starts wherever its character count fell:
        # drop the cut-off first word ("ctive disease" -> "disease").
        if " " in context and not context[0].isspace():
            context = context.split(" ", 1)[1]
        joiner = "" if context.rstrip().endswith(("(", "[")) else " "
        passage["sentence"] = _tidy(f"… {context.rstrip()}{joiner}{marker.get('text') or ''}")
    return passage


def sentence_around(content: str, offset: int) -> str:
    """The sentence of `content` that holds position `offset`, tidied for
    display (Markdown marks and line breaks removed), cut to
    MAX_PASSAGE_CHARACTERS around the offset if it is longer."""
    if not content or offset < 0 or offset > len(content):
        return ""
    start = _sentence_start(content, offset)
    end = _sentence_end(content, offset)
    text = content[start:end]
    if len(text) > MAX_PASSAGE_CHARACTERS:
        middle = offset - start
        left = max(0, middle - MAX_PASSAGE_CHARACTERS // 2)
        text = ("… " if left else "") + text[left:left + MAX_PASSAGE_CHARACTERS] + " …"
    return _tidy(text)


def _sentence_start(content: str, offset: int) -> int:
    """Just after the sentence end (or blank line) before `offset`."""
    position = offset
    while position > 0:
        if content.endswith(("\n\n", "\n\r\n"), 0, position):     # a blank line
            return position
        if content[position - 1] in " \n" and _ends_sentence(content, position - 2):
            return position
        position -= 1
    return 0


def _sentence_end(content: str, offset: int) -> int:
    """Just after the first sentence end (or blank line) after `offset`."""
    position = offset
    while position < len(content):
        if content.startswith("\n\n", position) or content.startswith("\r\n\r\n", position):
            return position
        if (content[position] in ".?!" and _ends_sentence(content, position)
                and (position + 1 == len(content) or content[position + 1] in " \r\n")):
            return position + 1
        position += 1
    return len(content)


def _ends_sentence(content: str, index: int) -> bool:
    """Whether the character at `index` is a full stop (or ? !) that ends a
    sentence: not after an abbreviation ("Fig.", "et al."), not a single
    initial ("J."), not a decimal point."""
    if index < 0 or content[index] not in ".?!":
        return False
    if content[index] != ".":
        return True
    word = re.search(r"([A-Za-z.]+)$", content[:index])
    if word is None:
        return True
    previous = word.group(1).lower().rstrip(".")
    return previous not in ABBREVIATIONS and len(previous) > 1


def _tidy(text: str) -> str:
    """Markdown emphasis and HTML tags removed, whitespace collapsed.

    An underscore or asterisk inside a word is text, not emphasis: gene and
    strain names such as "Aac5_2166" keep theirs."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"(?<!\w)[*_]+|[*_]+(?!\w)", "", text)      # emphasis marks only
    text = re.sub(r"`+|^#+\s*", "", text, flags=re.MULTILINE)
    return re.sub(r"\s+", " ", text).strip()
