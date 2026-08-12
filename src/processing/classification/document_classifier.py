#!/usr/bin/env python3
"""
Document classification — research article, book chapter, or whole book.

For biology researchers: the pipeline reads an article's title page very
differently from a book's front matter, so it has to know which it is holding
before it starts. This module makes that decision from the text layer alone —
no VLM, no GPU — by collecting evidence from the first few pages and scoring
it. It is CPU-cheap; the whole classification costs a few milliseconds on top
of the front-matter scan that already read those pages.

**This is a rewrite of V5's classifier, not a port.** V5's version was
measured on `Tendon assembly 2021 [1-7].pdf` — the front matter of a Springer
volume, carrying ISSN, ISBN and a book DOI on page 5 and a table of contents
on pages 6-7 — and it returned `research_article` at 0.60 (article 60 /
chapter 55 / book 30). Three independent causes, each fixed here:

1. **ISSN scored +25 for `research_article`, unconditionally.** But a book
   *series* — "Advances in Experimental Medicine and Biology", in this case —
   carries an ISSN perfectly legitimately. That false signal was the margin
   that decided the file. Fixed by `_score_identifiers`: an ISSN only argues
   for an article when there is **no ISBN beside it**; an ISSN *with* an ISBN
   is a series marker and argues for a book.
2. **Table-of-contents detection never fired.** V5 looked only for
   `table of contents` and `^contents$`. A real book TOC page is often headed
   with nothing but a roman numeral (`v`) and lists entries with **dot
   leaders**: `1 Introduction . . . . . . . 1`. Fixed by `_count_toc_evidence`,
   which looks for the dot leaders themselves.
3. **Scanning deeper did not rescue it** — measured at 5 and 10 pages, both
   still 0.60, because cause 2 held at any depth. Fixed only by fixing 2.

A fourth trap is deliberately avoided: **page count is weak evidence.** The
Springer file above is 7 pages because it is an *extract* of a 345-page book,
and V5's "short document, therefore an article" rule fought the correct answer
all the way. Structural evidence outranks length here.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, Optional

from ..extractors.front_matter import FrontMatterIdentifiers
from ..models import ClassificationSignals, DocumentType
from ..utils.regex_patterns import (
    ABSTRACT_PATTERNS,
    CHAPTER_HEADING_PATTERNS,
    DOT_LEADER_LINE_PATTERN,
    EXPLICIT_CHAPTER_PATTERNS,
    JOURNAL_SOURCE_PATTERNS,
    PEER_REVIEW_DATE_PATTERNS,
    TOC_KEYWORD_PATTERNS,
    VOLUME_INFO_PATTERNS,
)

LOG = logging.getLogger(__name__)

# A page needs this many dot-leader lines before we call it a contents page.
# One or two can appear in a figure legend or an index fragment; three in a
# row is a table of contents.
MIN_DOT_LEADER_LINES = 3

# Below this the classification is reported as uncertain, so a caller can fall
# back to the safe default rather than acting on a coin toss.
UNCERTAIN_BELOW = 0.60


@dataclass
class ClassificationOutcome:
    """
    The decision, its confidence, and the evidence behind it.

    Attributes:
        document_type: the winning `DocumentType`.
        confidence: 0.5 .. 1.0, scaled by how far the winner beat the runner-up.
        signals: the raw evidence, for logging and for the profile.
        scores: points per type, so a wrong answer can be explained rather
            than merely observed.
        is_confident: False when the margin was thin (see `UNCERTAIN_BELOW`).
    """
    document_type: DocumentType
    confidence: float
    signals: ClassificationSignals
    scores: Dict[str, int] = field(default_factory=dict)

    @property
    def is_confident(self) -> bool:
        return self.confidence >= UNCERTAIN_BELOW


def classify_document(page_texts: Dict[int, str], page_count: int,
                      identifiers: Optional[FrontMatterIdentifiers] = None
                      ) -> ClassificationOutcome:
    """
    Decide whether a document is a research article, a book chapter, or a book.

    Args:
        page_texts: text layer keyed by 1-based page number — exactly what
            `scan_front_matter` already collected, so no page is read twice.
        page_count: the document's true length in pages. Note this may be far
            larger than `len(page_texts)`; only the front matter is scanned.
        identifiers: the DOI/ISBN/ISSN found by the front-matter scan. Passing
            None means "no identifiers found".

    Returns:
        A `ClassificationOutcome`. When no evidence separates the types it
        returns RESEARCH_ARTICLE at 0.50 with `is_confident` False — the same
        safe default V5 used, but honestly labelled as a guess.

    Example:
        >>> outcome = classify_document(texts, 345, identifiers)   # doctest: +SKIP
        >>> outcome.document_type, outcome.confidence
        (<DocumentType.BOOK: 'book'>, 0.94)
    """
    signals = gather_classification_signals(page_texts, page_count, identifiers)

    if not _has_substantive_evidence(signals):
        # Nothing but the page count to go on — which is exactly what a scanned
        # PDF with no text layer looks like. Deciding from length alone would
        # be a guess wearing a confidence score, so say so instead.
        LOG.info("No classification evidence in the text layer; "
                 "defaulting to research_article (uncertain)")
        return ClassificationOutcome(
            document_type=DocumentType.RESEARCH_ARTICLE, confidence=0.50,
            signals=signals, scores={"research_article": 0, "book_chapter": 0, "book": 0})

    scores = {"research_article": 0, "book_chapter": 0, "book": 0}
    _score_page_count(scores, page_count)
    _score_identifiers(scores, signals.metadata)
    _score_keywords(scores, signals.text_content)
    _score_structure(scores, signals.structure, signals.metadata)
    _apply_tie_breakers(scores, signals)

    document_type, confidence = _decide_winner(scores)
    LOG.info("Classified as %s (%.2f) from scores %s",
             document_type.value, confidence, scores)

    return ClassificationOutcome(document_type=document_type, confidence=confidence,
                                 signals=signals, scores=scores)


def gather_classification_signals(page_texts: Dict[int, str], page_count: int,
                                  identifiers: Optional[FrontMatterIdentifiers] = None
                                  ) -> ClassificationSignals:
    """
    Collect the evidence, without judging it.

    Kept separate from scoring on purpose: the evidence is worth storing on the
    profile and logging whatever the verdict, and a scoring change should never
    silently change what was observed.

    Args:
        page_texts: text layer keyed by 1-based page number.
        page_count: the document's true length.
        identifiers: front-matter DOI/ISBN/ISSN, or None.

    Returns:
        A `ClassificationSignals` with its four groups populated.
    """
    identifiers = identifiers or FrontMatterIdentifiers()

    metadata = {
        "has_doi": identifiers.doi is not None,
        "has_isbn": identifiers.isbn is not None,
        "has_issn": identifiers.issn is not None,
        "has_book_doi": _looks_like_a_book_doi(identifiers.doi),
        "doi": identifiers.doi,
        "isbn": identifiers.isbn,
        "issn": identifiers.issn,
        "identifier_pages": identifiers.identifier_pages(),
    }

    text_content = _gather_keyword_signals(page_texts)
    structure = _gather_structure_signals(page_texts)

    return ClassificationSignals(
        metadata=metadata,
        page_count={"page_count": page_count, "pages_scanned": len(page_texts)},
        text_content=text_content,
        structure=structure,
    )


# --- evidence gathering (helpers, after the main functions) ------------------

# Dot leaders: "1 Introduction . . . . . . . . 3" or "Preface......ix". The
# line must END in a page number, which is what separates a contents entry
# from an ellipsis in prose. The pattern lives in `utils/regex_patterns` so
# the book strategy's TOC parser recognises exactly the same lines.
DOT_LEADER_PATTERN = DOT_LEADER_LINE_PATTERN

# A DOI whose suffix says "this is part of a book". Three shapes, all seen in
# the sample corpus and all publisher conventions rather than guesses:
#   10.1007/978-3-030-80614-9      Springer — the book's own ISBN as the suffix
#   10.1016/bs.ctdb.2018.02.004    Elsevier — 'bs' for book series
#   10.1007/b97264                 Springer — 'b' plus digits for a book
# A journal article's DOI never takes these forms, so this is a cheap way to
# recognise a chapter whose page carries no ISBN at all.
BOOK_DOI_PATTERN = re.compile(r'^10\.[0-9]{4,}/(?:978[-0-9]|bs\.|b[0-9])', re.IGNORECASE)


def _looks_like_a_book_doi(doi: Optional[str]) -> bool:
    """True when the DOI's own shape says it belongs to a book or a series."""
    return bool(doi and BOOK_DOI_PATTERN.match(doi))


def _has_substantive_evidence(signals: ClassificationSignals) -> bool:
    """
    True when something other than the page count actually said anything.

    Page count is the one signal that is always available, even for a PDF
    whose text layer is empty. Letting it decide alone would turn "we read
    nothing" into a confident verdict.
    """
    metadata = signals.metadata
    if metadata["has_doi"] or metadata["has_isbn"] or metadata["has_issn"]:
        return True

    text_content = signals.text_content
    if text_content["abstract_count"] or text_content["chapter_heading_count"]:
        return True
    if any(text_content[flag] for flag in
           ("has_explicit_chapter", "has_volume_info",
            "has_journal_source", "has_peer_review_dates")):
        return True

    structure = signals.structure
    return bool(structure["toc_pages"] or structure["inline_contents_pages"])


def _page_facts(text: str) -> Dict:
    """Everything one page has to say, read once."""
    lowered = text.lower()
    return {
        "has_abstract": any(re.search(p, lowered, re.IGNORECASE | re.MULTILINE)
                            for p in ABSTRACT_PATTERNS),
        "chapter_heading_count": sum(len(re.findall(p, lowered, re.IGNORECASE))
                                     for p in CHAPTER_HEADING_PATTERNS),
        "has_explicit_chapter": any(p.search(text) for p in EXPLICIT_CHAPTER_PATTERNS),
        "has_volume_info": any(p.search(text) for p in VOLUME_INFO_PATTERNS),
        "has_journal_source": any(p.search(text) for p in JOURNAL_SOURCE_PATTERNS),
        "has_peer_review_dates": any(p.search(text) for p in PEER_REVIEW_DATE_PATTERNS),
        "has_contents_keyword": any(re.search(p, lowered, re.IGNORECASE | re.MULTILINE)
                                    for p in TOC_KEYWORD_PATTERNS),
        "dot_leader_lines": _count_toc_evidence(text),
    }


def _gather_keyword_signals(page_texts: Dict[int, str]) -> Dict:
    """Count the article-vs-book keyword markers across the scanned pages."""
    signals = {
        "abstract_count": 0,
        "chapter_heading_count": 0,
        "has_explicit_chapter": False,
        "has_volume_info": False,
        "has_journal_source": False,
        "has_peer_review_dates": False,
    }

    for text in page_texts.values():
        if not text:
            continue
        facts = _page_facts(text)
        signals["abstract_count"] += 1 if facts["has_abstract"] else 0
        signals["chapter_heading_count"] += facts["chapter_heading_count"]
        for flag in ("has_explicit_chapter", "has_volume_info",
                     "has_journal_source", "has_peer_review_dates"):
            signals[flag] = signals[flag] or facts[flag]

    return signals


def _gather_structure_signals(page_texts: Dict[int, str]) -> Dict:
    """
    Find contents pages — and tell a book's from a chapter's.

    Two fixes to V5 live here. The first is dot leaders (defect 2): a book's
    real contents page is headed `v`, not "Table of Contents", so the words
    V5 searched for are never on it.

    The second was found by measuring this classifier on the corpus. An
    Elsevier book chapter opens with its OWN contents list — "Contents / 1.
    Introduction 108 / 1.1 …" — on the same page as its title, authors and
    abstract. Counting that as a book's contents page pushed a chapter into
    `research_article`. So a contents list that SHARES a page with an abstract
    or a chapter heading is recorded separately as `inline_contents_pages`: it
    is evidence of a chapter, not of a whole book. A book's contents page
    stands alone.
    """
    toc_pages = []
    inline_contents_pages = []
    dot_leader_pages = []

    for page_number, text in sorted(page_texts.items()):
        if not text:
            continue
        facts = _page_facts(text)
        has_leaders = facts["dot_leader_lines"] >= MIN_DOT_LEADER_LINES
        if has_leaders:
            dot_leader_pages.append(page_number)
        if not (facts["has_contents_keyword"] or has_leaders):
            continue

        introduces_one_work = facts["has_abstract"] or facts["has_explicit_chapter"]
        if introduces_one_work:
            inline_contents_pages.append(page_number)
        else:
            toc_pages.append(page_number)

    return {
        "toc_pages": toc_pages,
        "toc_page_count": len(toc_pages),
        "inline_contents_pages": inline_contents_pages,
        "dot_leader_pages": dot_leader_pages,
    }


def _count_toc_evidence(text: str) -> int:
    """
    How many lines on this page look like contents entries.

    A contents entry is a title, a run of dot leaders, and a page number:

        1 Introduction . . . . . . . . . . . . . . .   3
        Preface ......................................  ix

    V5 searched only for the words "table of contents", which the real page
    never contains — its heading is the roman page number `v`. Counting the
    leaders themselves is what makes a book's TOC visible at all.
    """
    return len(DOT_LEADER_PATTERN.findall(text))


# --- scoring (helpers) ------------------------------------------------------

def _score_page_count(scores: Dict[str, int], page_count: int) -> None:
    """
    Length is weak evidence, and deliberately weaker here than in V5.

    A book's front matter extracted to its own PDF is a handful of pages, and
    V5's "+20 research_article for anything under 5 pages" pulled hard in the
    wrong direction on exactly such a file. Length only speaks loudly at the
    long end, where it is genuinely reliable.
    """
    if page_count < 5:
        scores["research_article"] += 10
    elif page_count <= 40:
        scores["research_article"] += 10
        scores["book_chapter"] += 10
    elif page_count > 60:
        scores["book"] += 40
        scores["research_article"] -= 20


def _score_identifiers(scores: Dict[str, int], metadata: Dict) -> None:
    """
    Score DOI / ISBN / ISSN — and read ISSN *in context* (V5 defect 1).

    An ISSN identifies a serial publication. A journal is a serial; so is a
    book series. V5 scored it +25 for `research_article` either way, and on
    the Springer volume that single unconditional bonus was the whole margin.
    Here an ISSN argues for an article only when it stands alone. Next to an
    ISBN it means "this volume belongs to a series", which is a book signal.
    """
    if metadata["has_doi"]:
        scores["research_article"] += 20
        scores["book_chapter"] += 10        # chapters carry their own DOIs

    if metadata["has_isbn"]:
        scores["book_chapter"] += 30
        scores["book"] += 30

    # The DOI's own shape can say "book" even when no ISBN is printed — which
    # is exactly the case for an Elsevier book chapter (see BOOK_DOI_PATTERN).
    if metadata["has_book_doi"]:
        scores["book"] += 20
        scores["book_chapter"] += 20

    if metadata["has_issn"]:
        is_series_issn = metadata["has_isbn"] or metadata["has_book_doi"]
        if is_series_issn:
            # A series ISSN, printed beside an ISBN or a book DOI. Reinforces
            # the book reading and scores NOTHING for research_article.
            scores["book"] += 15
            scores["book_chapter"] += 10
        else:
            scores["research_article"] += 15


def _score_keywords(scores: Dict[str, int], text_content: Dict) -> None:
    """Score the content markers. Peer-review dates remain the strongest."""
    if text_content["abstract_count"] > 0:
        scores["research_article"] += 25
        scores["book_chapter"] += 5

    if text_content["has_peer_review_dates"]:
        scores["research_article"] += 40

    if text_content["has_journal_source"]:
        scores["research_article"] += 50

    if text_content["has_volume_info"]:
        scores["research_article"] += 5
        scores["book_chapter"] += 5


def _score_structure(scores: Dict[str, int], structure: Dict, metadata: Dict) -> None:
    """
    Score contents pages and explicit chapter headings.

    A contents page is the single most book-like thing a document can contain,
    so it is weighted accordingly — but only now that we can actually see one.
    """
    if structure["toc_page_count"] > 0:
        scores["book"] += 40
        # Multiple contents pages means a long work, not a chapter reprint.
        if structure["toc_page_count"] > 1:
            scores["book"] += 15

    # A contents list sharing a page with an abstract belongs to the chapter
    # that page introduces — see `_gather_structure_signals`.
    if structure["inline_contents_pages"]:
        scores["book_chapter"] += 25


def _apply_tie_breakers(scores: Dict[str, int], signals: ClassificationSignals) -> None:
    """
    Resolve the remaining ambiguities, chiefly book vs book chapter.

    An explicit "Chapter 3" heading argues for a chapter — but only when
    something else already says "book" (an ISBN, or real length). Standing
    alone it is as likely to be a cross-reference in an article.
    """
    text_content = signals.text_content
    structure = signals.structure
    metadata = signals.metadata
    page_count = signals.page_count["page_count"]

    if text_content["has_explicit_chapter"]:
        corroborated = (metadata["has_isbn"] or metadata["has_book_doi"]
                        or page_count > 10)
        scores["book_chapter"] += 30 if corroborated else 10

    # A contents page belongs to the whole work, not to one chapter of it.
    if structure["toc_page_count"] > 0:
        scores["book_chapter"] -= 20

    # Strong, self-contained article evidence with no book identifier at all.
    if scores["research_article"] > 60 and not metadata["has_isbn"]:
        scores["book_chapter"] -= 20

    # A settled book verdict rules out the article reading entirely.
    if scores["book"] > 60:
        scores["research_article"] = 0


def _decide_winner(scores: Dict[str, int]) -> tuple[DocumentType, float]:
    """
    Pick the winner and convert the margin over the runner-up into confidence.

    A 50-point margin reads as certainty; a tie reads as 0.50, which is what
    "I do not know" should look like. When nothing scores at all we return the
    safe default — a research article — at 0.50, not at a fabricated 0.60.
    """
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    winner, winner_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0

    if winner_score <= 0:
        LOG.warning("No classification evidence; defaulting to research_article")
        return DocumentType.RESEARCH_ARTICLE, 0.50

    confidence = min(1.0, 0.5 + (winner_score - runner_up) / 50.0)
    return DocumentType(winner), round(confidence, 2)
