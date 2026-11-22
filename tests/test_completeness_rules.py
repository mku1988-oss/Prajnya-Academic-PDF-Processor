#!/usr/bin/env python3
"""
What a record needs depends on what it IS.

Found on the session-9 full sample run: a Springer volume was flagged "missing
authors, journal" while correctly holding editors, publisher, ISBN and year. It
was not incomplete — it was being judged by a paper's rule. A flag that can
never be cleared teaches people to ignore flags, which costs more than the
missing field ever would.

So each type has its own rule:

    research_article   title + authors + DOI + journal      (unchanged)
    book               title + (editors|authors) + (publisher|isbn) + year
    book_chapter       title + authors + (journal|publisher) + DOI

The either-or groups are bibliography, not laxity: an edited volume is cited by
its editors and a monograph by its authors, and demanding both would flag every
book ever published.

Three things have to agree for this to be coherent, and each is tested here:
the rule, the LEDGER being able to store what the rule requires, and the review
prompt being able to CORRECT it. A field that can be demanded but not fixed is
a permanent flag by another route.
"""

import pytest

from src.processing.ledger.semantic_ledger import LedgerRecord
from src.processing.models import DocumentType
from src.processing.review import (
    COMPLETENESS_RULES,
    REQUIRED_FIELDS,
    assess_metadata_completeness,
    build_ledger_record,
    describe_rule,
    record_needs_review,
    rule_for,
)
from src.processing.review_prompt import DONE_CHOICE, EDITABLE_FIELDS, review_record


def scripted(answers):
    """An input_fn that replays answers, then raises EOFError like a real pipe."""
    queue = list(answers)

    def _input(_prompt=""):
        if not queue:
            raise EOFError
        return queue.pop(0)

    return _input

# The real Springer volume from the sample run, as the pipeline extracted it.
THE_BOOK = {
    "title": "Progress in Heritable Soft Connective Tissue Diseases",
    "authors": [],
    "editors": ["Jaroslava Halper"],
    "publisher": "Springer Nature Switzerland AG",
    "isbn": "9783030806132",
    "year": 2021,
    "journal": None,
    "doi": "10.1007/978-3-030-80614-9",
    "validation_confidence": 0.95,
}

# The real Elsevier chapter from the same run: authors and a DOI, but no
# container title anywhere — a genuine extraction gap the flag should catch.
THE_CHAPTER = {
    "title": "Collagen Fibril Assembly and Function",
    "authors": ["A. One", "B. Two", "C. Three", "D. Four"],
    "doi": "10.1016/bs.ctdb.2018.02.004",
    "journal": None,
    "publisher": None,
    "validation_confidence": 0.95,
}


class TestTheBookThatStartedThis:
    def test_the_real_volume_is_now_complete(self):
        """It always was; the rule was wrong, not the extraction."""
        assessment = assess_metadata_completeness(THE_BOOK, DocumentType.BOOK)

        assert assessment.is_complete
        assert assessment.missing_fields == []
        assert not assessment.requires_review

    def test_the_same_volume_judged_as_a_paper_is_still_flagged(self):
        """The old behaviour, kept visible so the change is not accidental."""
        assessment = assess_metadata_completeness(THE_BOOK,
                                                  DocumentType.RESEARCH_ARTICLE)
        assert assessment.missing_fields == ["authors", "journal"]

    def test_a_book_needs_no_doi(self):
        """Plenty of books have none; the ISBN is the identifier that matters."""
        without_doi = {key: value for key, value in THE_BOOK.items() if key != "doi"}
        assert assess_metadata_completeness(without_doi, DocumentType.BOOK).is_complete


class TestEitherOrGroups:
    def test_a_monograph_is_cited_by_its_authors(self):
        """No editors, but authors — an authored book, and perfectly complete."""
        monograph = {**THE_BOOK, "editors": [], "authors": ["Solo Author"]}
        assert assess_metadata_completeness(monograph, DocumentType.BOOK).is_complete

    def test_neither_authors_nor_editors_is_incomplete(self):
        anonymous = {**THE_BOOK, "editors": [], "authors": []}
        assessment = assess_metadata_completeness(anonymous, DocumentType.BOOK)

        assert assessment.missing_fields == ["editors_or_authors"]

    def test_an_isbn_stands_in_for_a_publisher(self):
        no_publisher = {**THE_BOOK, "publisher": None}
        assert assess_metadata_completeness(no_publisher, DocumentType.BOOK).is_complete

    def test_neither_publisher_nor_isbn_is_incomplete(self):
        nothing = {**THE_BOOK, "publisher": None, "isbn": None}
        assessment = assess_metadata_completeness(nothing, DocumentType.BOOK)

        assert "publisher_or_isbn" in assessment.missing_fields


class TestTheChapterRule:
    def test_the_real_chapter_is_still_flagged_for_its_container(self):
        """This flag is CORRECT: a chapter with no book title cannot be resolved."""
        assessment = assess_metadata_completeness(THE_CHAPTER,
                                                  DocumentType.BOOK_CHAPTER)

        assert assessment.missing_fields == ["journal_or_publisher"]

    def test_a_chapter_with_its_book_named_is_complete(self):
        complete = {**THE_CHAPTER, "publisher": "Elsevier"}
        assert assess_metadata_completeness(complete,
                                            DocumentType.BOOK_CHAPTER).is_complete

    def test_a_chapter_still_needs_its_own_authors(self):
        """The whole point of chapter-level attribution."""
        anonymous = {**THE_CHAPTER, "publisher": "Elsevier", "authors": []}
        assessment = assess_metadata_completeness(anonymous,
                                                  DocumentType.BOOK_CHAPTER)
        assert "authors" in assessment.missing_fields


class TestTheArticleRuleIsUnchanged:
    """The user's original rule. A refactor must not quietly relax it."""

    def test_the_article_rule_is_exactly_the_four(self):
        assert COMPLETENESS_RULES["research_article"] == REQUIRED_FIELDS
        assert rule_for(DocumentType.RESEARCH_ARTICLE) == \
            ("title", "authors", "doi", "journal")

    @pytest.mark.parametrize("field", REQUIRED_FIELDS)
    def test_each_one_is_still_required(self, field):
        paper = {"title": "T", "authors": ["A"], "doi": "10.1/x",
                 "journal": "J", "validation_confidence": 0.9}
        paper.pop(field)

        assessment = assess_metadata_completeness(paper,
                                                  DocumentType.RESEARCH_ARTICLE)
        assert assessment.missing_fields == [field]

    @pytest.mark.parametrize("unknown", [None, "", "newsletter", "BOOK-ish"])
    def test_an_unknown_type_is_judged_as_a_paper(self, unknown):
        """Over-flag an unclassified document rather than wave it through."""
        assert rule_for(unknown) == REQUIRED_FIELDS

    def test_a_string_type_works_as_well_as_the_enum(self):
        """run_prajnya has the enum; a ledger row has only its string."""
        assert rule_for("book") == rule_for(DocumentType.BOOK)


class TestTheLedgerCanHoldWhatTheRuleDemands:
    def test_a_book_record_stores_editors_publisher_and_its_type(self, tmp_path):
        pdf = tmp_path / "book.pdf"
        pdf.write_bytes(b"%PDF")

        record = build_ledger_record(pdf, THE_BOOK, document_type=DocumentType.BOOK)

        assert record.editors == ["Jaroslava Halper"]
        assert record.publisher == "Springer Nature Switzerland AG"
        assert record.document_type == "book"
        assert record.requires_review is False

    def test_a_stored_book_is_reassessed_as_a_book(self):
        """
        The reason the type is stored at all. Without it, `--review` months
        later would judge a book by the paper rule and re-flag it forever.
        """
        record = LedgerRecord(
            document_id="b1", pdf_filename="book.pdf", document_type="book",
            title=THE_BOOK["title"], authors=[], editors=["Jaroslava Halper"],
            publisher="Springer", year=2021, validation_confidence=0.95,
            validation_source="vlm", last_updated="2026-08-07", source_hash="b1")

        assert not record_needs_review(record)

    def test_a_row_with_no_type_falls_back_to_the_strict_rule(self):
        """Rows written before session 9 — err towards asking."""
        record = LedgerRecord(
            document_id="old", pdf_filename="old.pdf", title="T", authors=["A"],
            doi=None, journal=None, validation_confidence=0.9,
            validation_source="vlm", last_updated="2026-08-01", source_hash="old")

        assert record_needs_review(record)


class TestTheReviewScreenCanFixWhatIsDemanded:
    def test_every_requirable_field_is_editable(self):
        """
        The invariant that keeps this coherent. If a rule can demand a field the
        prompt cannot show, that document is flagged forever.
        """
        demanded = set()
        for rule in COMPLETENESS_RULES.values():
            for requirement in rule:
                if isinstance(requirement, tuple):
                    demanded.update(requirement)
                else:
                    demanded.add(requirement)

        # `isbn` is the one exception: it is an alternative to publisher, never
        # demanded alone, and is machine-read from the copyright page.
        assert demanded - {"isbn"} <= set(EDITABLE_FIELDS)

    def test_a_book_can_be_completed_by_typing_its_editors(self):
        record = LedgerRecord(
            document_id="b2", pdf_filename="book.pdf", document_type="book",
            title="A Volume", authors=[], editors=[], publisher="Springer",
            year=2021, requires_review=True, review_needed_fields=["editors_or_authors"],
            validation_confidence=0.5, validation_source="vlm",
            last_updated="2026-08-07", source_hash="b2")

        updated = review_record(
            record,
            input_fn=scripted([str(EDITABLE_FIELDS.index("editors") + 1),
                               "Jaroslava Halper", DONE_CHOICE]),
            output_fn=lambda *_: None)

        assert updated.editors == ["Jaroslava Halper"]
        assert updated.review_status == "reviewed"
        assert updated.requires_review is False

    def test_a_corrected_book_is_not_re_flagged_for_a_journal(self):
        """The bug this whole change exists to kill."""
        record = LedgerRecord(
            document_id="b3", pdf_filename="book.pdf", document_type="book",
            title="A Volume", authors=[], editors=["An Editor"],
            publisher="Springer", year=2021, validation_confidence=0.1,
            validation_source="vlm", last_updated="2026-08-07", source_hash="b3")

        updated = review_record(record, input_fn=scripted([DONE_CHOICE]),
                                output_fn=lambda *_: None)

        assert updated.review_needed_fields == []
        assert updated.review_status == "reviewed"


class TestDescribingTheRule:
    def test_the_log_line_says_what_this_kind_needs(self):
        described = describe_rule(DocumentType.BOOK)

        assert "editors or authors" in described
        assert "publisher or isbn" in described
