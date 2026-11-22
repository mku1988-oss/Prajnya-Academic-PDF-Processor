#!/usr/bin/env python3
"""
What "reviewed" means — decided once, for every front end.

For biology researchers: when you fix a record's metadata, something has to
decide whether the record is now *complete* — a proper citation — or still
provisional. That decision is what these tests cover.

**Why it had to be extracted.** The logic lived inside `review_prompt.py`,
which drives a numbered-menu conversation at a terminal. A graphical form
cannot reuse that conversation — the collecting of values is entirely different
— but it MUST reach the same verdict. Two copies of "is this record complete
now?" is how two front ends start disagreeing about whether a document is still
flagged, and a user would see a record marked reviewed in one screen and
provisional in the other.

So: collecting the values is a user-interface job and differs. Deciding what
they mean is not, and is `review.apply_corrections`.

The `reviewed_by` field was hardcoded to `"cli"` when a terminal was the only
way in. It is a parameter now, because a GUI correction did not happen at a CLI.
"""

from __future__ import annotations

import pytest

from src.processing.ledger.semantic_ledger import LedgerRecord
from src.processing.review import apply_corrections


def article(**overrides) -> LedgerRecord:
    """A flagged research article, missing its journal."""
    values = dict(document_id="abc123", pdf_filename="paper.pdf",
                  title="A paper", authors=["Ann Lee"], doi="10.1/x",
                  journal=None, document_type="research_article",
                  requires_review=True, review_status="provisional",
                  validation_confidence=0.5, validation_source="merged",
                  last_updated="2026-08-10T00:00:00Z", source_hash="deadbeef")
    values.update(overrides)
    return LedgerRecord(**values)


class TestTheVerdict:
    def test_completing_a_record_marks_it_reviewed(self):
        updated = apply_corrections(article(), {
            "title": "A paper", "authors": ["Ann Lee"], "doi": "10.1/x",
            "journal": "Cell Biology International", "year": 2006})

        assert updated.review_status == "reviewed"
        assert updated.requires_review is False
        assert updated.review_needed_fields == []

    def test_an_incomplete_correction_stays_provisional(self):
        """Half a fix is not a fix — the record keeps waiting."""
        updated = apply_corrections(article(), {
            "title": "A paper", "authors": ["Ann Lee"], "doi": "10.1/x",
            "journal": None})

        assert updated.review_status == "provisional"
        assert updated.requires_review is True
        assert "journal" in updated.review_needed_fields

    def test_low_pipeline_confidence_does_not_re_flag_a_completed_record(self):
        """
        The defect session 9 found live: a record came back `reviewed` while
        `requires_review` stayed True, because the pipeline's OLD low
        confidence was re-applied after a human had already answered that
        doubt. Completeness alone decides the flag here.
        """
        record = article(validation_confidence=0.1)
        updated = apply_corrections(record, {
            "title": "A paper", "authors": ["Ann Lee"], "doi": "10.1/x",
            "journal": "Cell Biology International"})

        assert updated.review_status == "reviewed"
        assert updated.requires_review is False

    def test_a_book_is_judged_as_a_book(self):
        """
        A book corrected months later must not be reassessed as a paper and
        re-flagged for a journal it never had.
        """
        book = article(document_type="book", title="A volume", authors=[],
                       doi=None, journal=None)
        updated = apply_corrections(book, {
            "title": "A volume", "editors": ["Ed Editor"],
            "publisher": "Springer", "year": 2021})

        assert updated.review_status == "reviewed"
        assert updated.requires_review is False

    def test_the_original_record_is_not_mutated(self):
        record = article()
        apply_corrections(record, {"title": "Something else"})
        assert record.title == "A paper"
        assert record.requires_review is True


class TestWhoDidTheReviewing:
    def test_it_defaults_to_cli(self):
        """Unchanged for the terminal, which is what `review_prompt` passes."""
        assert apply_corrections(article(), {}).last_reviewed_by == "cli"

    def test_a_gui_correction_is_not_recorded_as_a_cli_one(self):
        """The hardcode this extraction removed."""
        updated = apply_corrections(article(), {}, reviewed_by="gui")
        assert updated.last_reviewed_by == "gui"

    def test_any_reviewer_name_is_accepted(self):
        updated = apply_corrections(article(), {}, reviewed_by="m.kumar")
        assert updated.last_reviewed_by == "m.kumar"


class TestValuesFromEitherKindOfFrontEnd:
    """
    A terminal returns one comma-separated string; a form returns a list. Both
    are legitimate, so both are normalized here rather than by each caller.
    """

    @pytest.mark.parametrize("authors,expected", [
        ("Ann Lee, Bo Chen", ["Ann Lee", "Bo Chen"]),
        (["Ann Lee", "Bo Chen"], ["Ann Lee", "Bo Chen"]),
        (("Ann Lee",), ["Ann Lee"]),
        ("", []),
        (None, []),
        ("Ann Lee,  , Bo Chen", ["Ann Lee", "Bo Chen"]),
    ])
    def test_authors_normalize_the_same_way_either_way(self, authors, expected):
        assert apply_corrections(article(), {"authors": authors}).authors == expected

    def test_editors_normalize_identically(self):
        updated = apply_corrections(article(), {"editors": "Ed One, Ed Two"})
        assert updated.editors == ["Ed One", "Ed Two"]

    def test_a_missing_key_clears_the_field(self):
        """A review that blanks a wrong title must be able to say so."""
        assert apply_corrections(article(), {}).title is None


class TestThePromptDelegatesRatherThanDuplicating:
    """
    Duplicating the 30 lines is how the two front ends start disagreeing.
    """

    def test_review_record_calls_apply_corrections(self):
        import inspect

        from src.processing.review_prompt import review_record

        source = inspect.getsource(review_record)
        assert "apply_corrections(" in source

    def test_the_prompt_no_longer_decides_the_verdict_itself(self):
        import inspect

        from src.processing.review_prompt import review_record

        source = inspect.getsource(review_record)
        assert "review_status" not in source
        assert "requires_review" not in source
        assert 'last_reviewed_by="cli"' not in source

    def test_the_terminal_path_still_reports_itself_as_cli(self):
        """Behaviour must not have changed for the front end that existed."""
        import inspect

        from src.processing.review_prompt import review_record

        assert 'reviewed_by="cli"' in inspect.getsource(review_record)
