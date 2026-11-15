#!/usr/bin/env python
"""Builds the `citations` block every output document carries.

For biology researchers: the block answers, for one document, the questions
a program citing it has to ask:

    level               how the record was established (local/zotero/online)
    metadata_source     which step actually supplied it
    source_label        the short form to cite it by: "Bella, 2016"
    cited_by_file_name  true when no author, editor or title was usable and
                        the label is the PDF's file name instead
    reference_list      whether the paper's own reference list is included
    references          that list (level `online` only)
    markers             the in-text citation marks linked to it (numbered
                        lists only), each with its place in the page text

The block is present at EVERY level, so a reader never has to guess the
level from what is missing.

IT IS DERIVED, NEVER STORED AS A FACT. The label and the source are worked
out from the metadata each time the document is written. When a researcher
corrects an author by hand, the file is re-written and the label follows
(`review.regenerate_document_json`). The standing rule: cache facts, derive
on read.

THE FILE NAME NEVER GOES INTO `title` (researcher, session 32). A program
downstream uses the title to recognise headings and to export BibTeX, and
"JRuberti.pdf" is neither. The file name appears only as `source_label`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..utils.author_utils import split_name
from .level import DEFAULT_LEVEL, SET_BY_DEFAULT
from .markers import locate_markers

#: Which step supplied the metadata. Written as `metadata_source`.
SOURCE_EXTRACTED = "extracted"   # Prajnya's own reading of the PDF
SOURCE_ZOTERO = "zotero"
SOURCE_CROSSREF = "crossref"
SOURCE_OPENALEX = "openalex"
SOURCE_USER = "user"             # corrected by hand in the review step
SOURCE_FILE_NAME = "file_name"   # nothing usable: cited by its file name
METADATA_SOURCES: Tuple[str, ...] = (
    SOURCE_EXTRACTED, SOURCE_ZOTERO, SOURCE_CROSSREF, SOURCE_OPENALEX,
    SOURCE_USER, SOURCE_FILE_NAME)

#: The state of the paper's own reference list. Written as
#: `reference_list.status`. A reader can tell "none exists" from "we did
#: not look" from "we could not reach the service".
LIST_NOT_REQUESTED = "not_requested"   # levels local and zotero
LIST_FOUND = "found"
LIST_NOT_AVAILABLE = "not_available"   # asked; no service holds one
LIST_NOT_REACHED = "not_reached"       # the network failed
REFERENCE_LIST_STATUSES: Tuple[str, ...] = (
    LIST_NOT_REQUESTED, LIST_FOUND, LIST_NOT_AVAILABLE, LIST_NOT_REACHED)

#: A title used as a label is cut to this many characters.
MAX_TITLE_LABEL_CHARS = 80

#: Words that belong to the surname that follows them ("van der Flier").
_SURNAME_PARTICLES = {"van", "von", "der", "den", "de", "del", "della", "di",
                      "da", "dos", "du", "la", "le", "ten", "ter", "st", "st."}


def build_citations_block(metadata: Optional[Dict[str, Any]], file_name: str,
                          level: Optional[str] = None,
                          level_set_by: Optional[str] = None,
                          reference_data: Optional[Dict[str, Any]] = None,
                          page_contents: Optional[Mapping[int, str]] = None) -> Dict[str, Any]:
    """
    Build one document's `citations` block.

    Args:
        metadata: the merged bibliographic record (`metadata_final`).
        file_name: the PDF's file name, e.g. "JRuberti.pdf". Used as the
            label only when the metadata gives nothing to cite by.
        level: the citation level the run used. None means the document was
            written outside a run (a test, a script), which is recorded as
            the default level, set by "default".
        level_set_by: which source decided the level (see `level.py`).
        reference_data: the paper's own reference list, as the finder
            returns it (`{"reference_list": {...}, "references": [...]}`;
            plus a `markers` list when the paper's numbered marks were
            linked; a `report` key, if present, is ignored). Lists come only
            at level `online`; None (every other level) leaves the block
            saying "not requested".
        page_contents: `{1-based page number: the page's content string}`,
            exactly what goes into `pages[].content`. The markers' offsets
            are worked out against it each time the block is built, so they
            always match the text being written. Without it no offset is set.

    Returns:
        The block, JSON-serializable. Without `reference_data` the list is
        "not requested" and `references` and `markers` are empty. Markers
        come only from a numbered list; each carries `content_offset`, or
        None when its place in the page text is not certain.

    Example:
        >>> block = build_citations_block(
        ...     {"authors": ["Jordi Bella"], "year": 2016,
        ...      "validation_source": "zotero_api"}, "bella.pdf", "zotero", "config")
        >>> block["source_label"], block["metadata_source"]
        ('Bella, 2016', 'zotero')
    """
    metadata = metadata if isinstance(metadata, dict) else {}
    label, by_file_name = source_label(metadata, file_name)
    reference_list, references = _reference_parts(reference_data)

    return {
        "level": level or DEFAULT_LEVEL,
        "level_set_by": level_set_by or SET_BY_DEFAULT,
        "metadata_source": (SOURCE_FILE_NAME if by_file_name
                            else metadata_source_of(metadata)),
        "source_label": label,
        "cited_by_file_name": by_file_name,
        "reference_list": reference_list,
        "references": references,
        "markers": locate_markers(_markers_of(reference_data), page_contents or {}),
    }


def source_label(metadata: Dict[str, Any], file_name: str) -> Tuple[str, bool]:
    """
    The short form a document is cited by, and whether it is the file name.

    The chain, first that applies:
      1. authors:  "Bella, 2016" / "Myllyharju and Kivirikko, 2004" /
                   "Fullerton et al., 2006"
      2. editors (a book):  "Halper (ed.), 2021"
      3. the title, shortened:  "Collagen structure: new tricks ..., 2016"
      4. the file name:  "Silberbush-2019-Fig2a,3.pdf"

    A missing year is written "n.d." (no date), the usual convention.

    Args:
        metadata: the merged bibliographic record.
        file_name: the PDF's file name.

    Returns:
        `(label, cited_by_file_name)`.
    """
    year = _year_text(metadata.get("year"))

    surnames = _surnames(metadata.get("authors"))
    if surnames:
        return f"{_join_surnames(surnames)}, {year}", False

    editors = _surnames(metadata.get("editors"))
    if editors:
        marker = "(ed.)" if len(editors) == 1 else "(eds.)"
        return f"{_join_surnames(editors)} {marker}, {year}", False

    title = _short_title(metadata.get("title"))
    if title:
        return f"{title}, {year}", False

    return file_name, True


def metadata_source_of(metadata: Dict[str, Any]) -> str:
    """
    Which step supplied the record, read from its `validation_source`.

    `validation_source` is a detailed, historical value ("zotero_api",
    "zotero_fuzzy_match", "front_matter_text_layer", ...). This reduces it
    to the short closed list a reader can switch on.

    Args:
        metadata: the merged bibliographic record.

    Returns:
        One of `METADATA_SOURCES` (never `file_name`: that is decided by
        `source_label`, from what the record lacks, not from its source).
    """
    source = str(metadata.get("validation_source") or "").lower()
    if source == "user_corrected":
        return SOURCE_USER
    for name in (SOURCE_ZOTERO, SOURCE_CROSSREF, SOURCE_OPENALEX):
        if source.startswith(name):
            return name
    return SOURCE_EXTRACTED


def reference_data_of(block: Any) -> Optional[Dict[str, Any]]:
    """
    The reference list held by an existing `citations` block, or None.

    Used to carry a list forward when a document is rewritten without
    asking the online services again.

    Args:
        block: a `citations` block read from an output file.

    Returns:
        `{"reference_list": ..., "references": ..., "markers": ...}`, or None when the block
        is missing or its list status is `not_requested`.
    """
    if not isinstance(block, dict):
        return None
    reference_list = block.get("reference_list")
    if not isinstance(reference_list, dict):
        return None
    if reference_list.get("status", LIST_NOT_REQUESTED) == LIST_NOT_REQUESTED:
        return None
    return {"reference_list": reference_list,
            "references": block.get("references") or [],
            "markers": block.get("markers") or []}


# --- helpers (after the main functions, per CLAUDE.md) ----------------------

def _reference_parts(reference_data: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Any]]:
    """
    The block's `reference_list` and `references`, copied from the data.

    Copies, so a later change to the block cannot alter the finder's result.
    Anything that is not a dict holding a `reference_list` dict gives the
    "not requested" pair.
    """
    if isinstance(reference_data, dict) and isinstance(reference_data.get("reference_list"), dict):
        return (dict(reference_data["reference_list"]),
                list(reference_data.get("references") or []))
    return {"status": LIST_NOT_REQUESTED}, []


def _markers_of(reference_data: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The marker dicts the data holds (a list of dicts), else none."""
    if not isinstance(reference_data, dict):
        return []
    markers = reference_data.get("markers")
    if not isinstance(markers, list):
        return []
    return [m for m in markers if isinstance(m, dict)]


def _surnames(names: Any) -> List[str]:
    """The surname of each name in a list; names that give none are dropped."""
    if not isinstance(names, (list, tuple)):
        return []
    found = [_surname(str(name)) for name in names if name]
    return [surname for surname in found if surname]


def _surname(name: str) -> Optional[str]:
    """
    One person's surname, keeping particles: "Annette van der Flier" ->
    "van der Flier"; "Bella, Jordi" -> "Bella".
    """
    first, last = split_name(name)
    if not last:
        return None
    if "," in name or not first:
        return last

    # "First Last" form: lower-case particles just before the last word
    # belong to the surname.
    given = first.split()
    particles: List[str] = []
    while given and given[-1].lower() in _SURNAME_PARTICLES and given[-1].islower():
        particles.insert(0, given.pop())
    return " ".join(particles + [last])


def _join_surnames(surnames: List[str]) -> str:
    """One, two, or "first et al." for three or more."""
    if len(surnames) == 1:
        return surnames[0]
    if len(surnames) == 2:
        return f"{surnames[0]} and {surnames[1]}"
    return f"{surnames[0]} et al."


def _year_text(year: Any) -> str:
    """The year as text, or "n.d." when there is none."""
    try:
        return str(int(str(year).strip()[:4]))
    except (TypeError, ValueError):
        return "n.d."


def _short_title(title: Any) -> str:
    """The title on one line, cut at a word boundary when it is long."""
    text = " ".join(str(title).split()) if isinstance(title, str) else ""
    if len(text) <= MAX_TITLE_LABEL_CHARS:
        return text
    cut = text[:MAX_TITLE_LABEL_CHARS].rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:") + " ..."
