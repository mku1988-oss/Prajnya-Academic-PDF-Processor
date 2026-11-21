#!/usr/bin/env python
"""Each page's `body`: the article's running text, for splitting into passages.

For biology researchers: a page's `content` is everything printed on it, as
read. Much of that is not the article's running text: the running header
and footer, the figure legends, the tables, the reference list, and
Prajnya's own placeholder lines that mark where a figure crop belongs.
Every one of these is already kept elsewhere in the file (legends under
`metadata.figures`, tables under `metadata.tables`, references under
`citations.references`). A program that splits text into passages for
search wants only the running text, so that a passage is never half a
legend or a run of reference entries.

`body` is `content` with those parts left out. Nothing is deleted from
`content`, and every part left out is recorded as a span of `content`:
furniture in `metadata.furniture` (with its text; `page_furniture.py`), the
rest in `metadata.outside_body`:

- `placeholder`: Prajnya's "[[FIGURE ...]]" / "[[TABLE ...]]" lines and its
  "<!-- ... -->" comments;
- `table`: a table written as HTML or as a Markdown grid;
- `caption`: a figure or table legend, where the page text holds the
  legend stored on that object (the whole legend, matched word by word);
- `reference_list`: from the "References" heading to the end of the list.
  The list ends at a heading that follows a reference list (STAR Methods,
  Acknowledgements, Appendix, Supplementary ...), as the reference-list
  reader decides it, or at a later page with few reference-like lines. Text
  after the list (Cell Press STAR Methods, eLife appendices) is body.

Derived on every write from the stored page text and objects, never cached.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .page_furniture import label_page_furniture
from .reference_pass import looks_like_a_citation
from .reference_list import _AFTER_LIST_HEADING as AFTER_LIST_HEADING
from .reference_list import _REFERENCE_HEADING as REFERENCE_HEADING

#: How many lines after a "References" heading may hold the first entry.
ENTRY_LOOKAHEAD = 3

#: A numbered entry's start: "1. Lee AJ", "[12] Doi", "(3) Muskovic".
_ENTRY_START = re.compile(r"^[\[(]?\d{1,3}[.\])]\s+\S")

#: A year, and an author written "Surname AB," (initials without stops).
_YEAR = re.compile(r"\b(?:1[89]|20)\d{2}[a-z]?\b")
_SURNAME_INITIALS = re.compile(r"\b[A-Z][\w'’-]+,?\s+[A-Z]{1,3}[,.]")

#: A legend's first line: "Fig. 2.", "Figure 3 |", "TABLE 1" and a number.
_LEGEND_START = re.compile(r"^(?:fig(?:ure)?s?\.?|table)\s*\d+[A-Za-z]?\s*[.:|]", re.IGNORECASE)

#: A legend shorter than this is not looked for in the text: a short string
#: ("Figure 2") would also match a sentence.
MIN_CAPTION_CHARACTERS = 30

#: A legend is found by this many words at its start and at its end ...
CAPTION_ANCHOR_WORDS = 6

#: ... no further apart in the page text than its own length times this.
CAPTION_LENGTH_SLACK = 1.3

_PLACEHOLDER = re.compile(r"\[\[(?:FIGURE|TABLE) [^\n]*?\]\]|<!--.*?-->", re.DOTALL)
_HTML_TABLE = re.compile(r"<table\b.*?</table>", re.DOTALL | re.IGNORECASE)
#: Two or more consecutive lines that start and end with "|".
_MARKDOWN_TABLE = re.compile(r"(?:^[ \t]*\|[^\n]*\|[ \t]*(?:\r?\n|$)){2,}", re.MULTILINE)
#: Markdown around a heading line ("## **References**").
_HEADING_MARKUP = re.compile(r"[#*_]+")


def pages_with_body(pages: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """
    The pages with `metadata.furniture`, `metadata.outside_body` and `body`
    added (contract 1.3).

    Every page with text gets all three: empty lists and `body == content`
    when nothing was left out. The pages given are not changed: each page and
    its metadata are copied (the document writer's pages are shared with the
    stored profile, which must not carry a second copy of every page's text).
    """
    furniture = label_page_furniture(pages)
    outside = outside_body_spans(pages)
    written = []
    for page in pages:
        if not _readable(page):
            written.append(page)
            continue
        number = page["page_number"]
        copy = dict(page)
        copy["metadata"] = dict(page.get("metadata") or {},
                                furniture=furniture.get(number, []),
                                outside_body=outside.get(number, []))
        copy["body"] = text_without(page["content"],
                                    furniture.get(number, []) + outside.get(number, []))
        written.append(copy)
    return written


def outside_body_spans(pages: Sequence[Mapping[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """
    {page_number: [{"kind", "start", "end"}, ...]}: the parts of each page's
    `content` that are not running text (furniture aside), in text order,
    never overlapping.
    """
    readable = [page for page in pages if _readable(page)]
    lists = _reference_list_spans(readable)
    found: Dict[int, List[Dict[str, Any]]] = {}
    for page in readable:
        content = page["content"]
        spans = _pattern_spans(content, _PLACEHOLDER, "placeholder")
        spans = _add(spans, _pattern_spans(content, _HTML_TABLE, "table"))
        spans = _add(spans, _pattern_spans(content, _MARKDOWN_TABLE, "table"))
        spans = _add(spans, lists.get(page["page_number"], []))
        spans = _add(spans, _caption_spans(content, page.get("metadata") or {}, spans))
        if spans:
            found[page["page_number"]] = sorted(spans, key=lambda s: s["start"])
    return found


def text_without(content: str, spans: Sequence[Mapping[str, Any]]) -> str:
    """
    `content` with each span left out. A span that is whole lines is left
    out with its line break, so no empty line is left in its place.
    """
    pieces, position = [], 0
    for start, end in _merged(spans):
        if start < position:
            start = position
        pieces.append(content[position:start])
        position = _through_line_break(content, start, end)
    pieces.append(content[position:])
    return "".join(pieces)


# --- helpers (after the main functions, per CLAUDE.md) ---------------------

def _readable(page: Any) -> bool:
    return (isinstance(page, Mapping) and isinstance(page.get("page_number"), int)
            and isinstance(page.get("content"), str))


def _pattern_spans(content: str, pattern: re.Pattern, kind: str) -> List[Dict[str, Any]]:
    return [{"kind": kind, "start": m.start(), "end": m.end()}
            for m in pattern.finditer(content)]


def _add(spans: List[Dict[str, Any]], new: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """`spans` plus each new span that overlaps none of them."""
    kept = list(spans)
    for span in new:
        if not any(span["start"] < old["end"] and old["start"] < span["end"] for old in kept):
            kept.append(span)
    return kept


def _caption_spans(content: str, metadata: Mapping[str, Any],
                   taken: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Where the page text holds a legend stored on one of its objects.

    The page text writes a legend with Markdown bold ("**Fig 2.** ...") and
    sometimes other code points for the same letter (micro sign and Greek
    mu), so both are compared in a normalised form (`_normalised`). The
    legend is found by its first `CAPTION_ANCHOR_WORDS` words and its last
    ones, in order and not too far apart; the span runs from the one to the
    other. A match inside a span already left out (a placeholder repeats
    the legend) does not count.
    """
    text, origin = _normalised(content)
    spans: List[Dict[str, Any]] = []
    objects = list(metadata.get("figures") or []) + list(metadata.get("tables") or [])
    for caption in {(item.get("caption") or "").strip() for item in objects}:
        if len(caption) < MIN_CAPTION_CHARACTERS:
            continue
        words = _normalised(caption)[0].split()
        head = " ".join(words[:CAPTION_ANCHOR_WORDS])
        tail = " ".join(words[-CAPTION_ANCHOR_WORDS:])
        reach = int(len(" ".join(words)) * CAPTION_LENGTH_SLACK) + len(tail)
        position = text.find(head)
        while position >= 0:
            end = text.find(tail, position, position + reach)
            if end >= 0:
                start, stop = origin[position], origin[end + len(tail) - 1] + 1
                # Take in the bold marks around the legend ("**Fig 2.**").
                while start > 0 and content[start - 1] in "*_":
                    start -= 1
                while stop < len(content) and content[stop] in "*_":
                    stop += 1
                if not any(start < old["end"] and old["start"] < stop
                           for old in list(taken) + spans):
                    spans.append({"kind": "caption", "start": start, "end": stop})
                    break
            position = text.find(head, position + 1)
    return spans


def _normalised(text: str) -> Tuple[str, List[int]]:
    """Text compared loosely: NFKC, no Markdown marks, single spaces; with
    the index in the original text of every character kept."""
    kept, origin = [], []
    for index, char in enumerate(text):
        for folded in unicodedata.normalize("NFKC", char):
            if folded in "*_#":
                continue
            if folded.isspace():
                if kept and kept[-1] != " ":
                    kept.append(" ")
                    origin.append(index)
                continue
            kept.append(folded)
            origin.append(index)
    return "".join(kept), origin


def _reference_list_spans(pages: Sequence[Mapping[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """Each page's part of a reference list, from its heading to its end.

    A list runs on over the following pages until a line that ends it
    (`_ends_the_list`) or the first page of another book chapter
    (`metadata.chapter` changes). Whether a page "looks like" references
    is not used: `metadata.reference_density` is near 0 on a book's
    two-column lists, whose entries wrap over several short lines.
    """
    spans: Dict[int, List[Dict[str, Any]]] = {}
    in_list, chapter = False, None
    for page in pages:
        lines = _lines(page["content"])
        page_chapter = (page.get("metadata") or {}).get("chapter")
        if in_list and page_chapter != chapter:
            in_list = False
        continuing = in_list
        start = 0 if continuing else _list_heading(lines)
        if start is None:
            continue
        in_list, chapter = True, page_chapter
        # On a page the list runs onto, its very first line may end it
        # ("Appendix 1" at the top of the page); on the heading's page, the
        # lines after the heading.
        end = next((s for s, _, text in lines
                    if (s >= start if continuing else s > start) and _ends_the_list(text)), None)
        if end is not None:
            in_list = False
        stop = end if end is not None else len(page["content"])
        if stop > start:
            spans[page["page_number"]] = [{"kind": "reference_list", "start": start, "end": stop}]
    return spans


def _list_heading(lines: List[Tuple[int, int, str]]):
    """Where a "References" heading followed by reference entries starts.

    The next `ENTRY_LOOKAHEAD` lines, read together, must look like an
    entry (or the first must start a numbered one), so a
    JSTOR cover page's "REFERENCES / Linked references are available on
    JSTOR" starts no list.
    """
    for index, (start, _, text) in enumerate(lines):
        if not REFERENCE_HEADING.match(_plain(text)):
            continue
        following = [line.strip() for _, _, line in lines[index + 1:index + 1 + ENTRY_LOOKAHEAD]]
        # An entry wraps over short lines: judge the lines together.
        if following and _looks_like_entries(following):
            return start
    return None


def _looks_like_entries(lines: List[str]) -> bool:
    """Whether a few lines read together are reference entries: numbered, or
    two bibliographic marks (the reference pass's test), or a year with
    Vancouver-style authors ("Arroyo-Crespo JJ, Vicent MJ. 2019.")."""
    joined = " ".join(lines)
    return bool(_ENTRY_START.match(lines[0]) or looks_like_a_citation(joined)
                or (_YEAR.search(joined) and _SURNAME_INITIALS.search(joined)))


def _ends_the_list(line: str) -> bool:
    """A heading that follows a reference list (as the list reader decides),
    or a figure or table legend: old papers print the legends after the list."""
    plain = _plain(line)
    return bool(AFTER_LIST_HEADING.match(plain) or _LEGEND_START.match(plain))


def _lines(content: str) -> List[Tuple[int, int, str]]:
    """(start, end, text) of every non-blank line."""
    lines, position = [], 0
    for raw in content.splitlines(keepends=True):
        text = raw.rstrip("\r\n")
        if text.strip():
            lines.append((position, position + len(text), text))
        position += len(raw)
    return lines


def _plain(text: str) -> str:
    """A heading line without Markdown marks, as the list reader sees it."""
    return _HEADING_MARKUP.sub("", text).strip()


def _merged(spans: Sequence[Mapping[str, Any]]) -> List[Tuple[int, int]]:
    """The spans as sorted, non-overlapping (start, end) pairs."""
    merged: List[List[int]] = []
    for span in sorted(spans, key=lambda s: s["start"]):
        if merged and span["start"] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], span["end"])
        else:
            merged.append([span["start"], span["end"]])
    return [(start, end) for start, end in merged]


def _through_line_break(content: str, start: int, end: int) -> int:
    """`end`, moved past the line break when the span is whole lines."""
    line_start = content.rfind("\n", 0, start) + 1
    line_end = content.find("\n", end)
    rest_of_line = content[end:] if line_end < 0 else content[end:line_end]
    if content[line_start:start].strip() or rest_of_line.strip():
        return end
    for newline in ("\r\n", "\n", "\r"):
        if content.startswith(newline, end):
            return end + len(newline)
    return end
