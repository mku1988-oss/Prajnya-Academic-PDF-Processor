#!/usr/bin/env python3
"""
Tests for metadata review — the assessment, the ledger row, and the correction.

The rule under test is the user's, inherited from V5: a record is complete only
when it has ALL of title, authors, DOI and journal. Anything less is
`provisional` and waits for a human.

Two behaviours here matter more than the rest, and most of these tests exist to
protect them:

**Regeneration patches, it does not rebuild.** V5's `regenerate_json_from_ledger`
constructed a fresh file from the ledger row, which set `"pages": []` — fixing a
typo in a title destroyed every page of extracted content. Here the pages must
survive a correction, and `test_pages_survive_a_correction` fails loudly if that
ever regresses.

**Nobody at the keyboard means nothing is accepted.** EOF, a closed pipe and
Ctrl-A-Ctrl-C all end the conversation without recording an answer, leaving the
record flagged. An unattended run must never be able to mark something reviewed.
"""

import json
from pathlib import Path

import pytest

from src.processing.ledger.semantic_ledger import LedgerRecord, generate_hash
from src.processing.review import (
    LOW_CONFIDENCE,
    REQUIRED_FIELDS,
    apply_record_to_metadata,
    assess_metadata_completeness,
    build_ledger_record,
    record_needs_review,
    records_needing_review,
    regenerate_document_json,
    regenerate_from_ledger,
    save_record,
)
from src.processing.review_prompt import (
    DONE_CHOICE,
    EDITABLE_FIELDS,
    offer_deferred_review,
    review_ledger,
    review_metadata,
    review_record,
)

COMPLETE = {
    "title": "An NMR method for water compartments",
    "authors": ["Fullerton", "Cameron"],
    "doi": "10.1016/j.cellbi.2005.09.009",
    "journal": "Cell Biology International",
    "year": 2006,
    "validation_confidence": 0.95,
    "validation_source": "zotero",
}


class FakeLedger:
    """A ledger in a dict — enough for every path the review flow takes."""

    def __init__(self, records=()):
        self._records = {record.document_id: record for record in records}
        self.commits = 0
        self.fail_on_update = False

    def get_record(self, document_id):
        return self._records.get(document_id)

    def get_all_records(self):
        return list(self._records.values())

    def update_record(self, record):
        if self.fail_on_update:
            raise OSError("ledger is locked")
        self._records[record.document_id] = record

    def commit(self):
        self.commits += 1


def make_record(**overrides) -> LedgerRecord:
    values = dict(document_id="abc123", pdf_filename="paper.pdf",
                  title=COMPLETE["title"], authors=list(COMPLETE["authors"]),
                  doi=COMPLETE["doi"], journal=COMPLETE["journal"], year=2006,
                  validation_confidence=0.95, validation_source="zotero",
                  last_updated="2026-08-06T00:00:00", source_hash="abc123")
    values.update(overrides)
    return LedgerRecord(**values)


def pick(field: str) -> str:
    """
    The menu number for a field, derived rather than hardcoded.

    The menu grew when books arrived (editors and publisher became editable),
    which silently invalidated every test that typed "4" meaning DOI. Deriving
    it means adding a field never rewrites a test's meaning.
    """
    return str(EDITABLE_FIELDS.index(field) + 1)


def scripted(answers):
    """An input_fn that replays answers, then raises EOFError like a real pipe."""
    queue = list(answers)

    def _input(_prompt=""):
        if not queue:
            raise EOFError
        return queue.pop(0)

    return _input


class TestAssessment:
    def test_a_complete_record_needs_nobody(self):
        assessment = assess_metadata_completeness(COMPLETE)
        assert assessment.is_complete
        assert not assessment.requires_review
        assert assessment.review_status == "reviewed"

    @pytest.mark.parametrize("field", REQUIRED_FIELDS)
    def test_each_required_field_is_actually_required(self, field):
        """All four, not 'most of four' — a citation needs every one."""
        metadata = dict(COMPLETE)
        metadata.pop(field)

        assessment = assess_metadata_completeness(metadata)
        assert assessment.missing_fields == [field]
        assert assessment.requires_review
        assert assessment.review_status == "provisional"

    def test_an_empty_author_list_is_not_authors(self):
        assessment = assess_metadata_completeness({**COMPLETE, "authors": []})
        assert "authors" in assessment.missing_fields

    def test_whitespace_is_not_a_title(self):
        assessment = assess_metadata_completeness({**COMPLETE, "title": "   "})
        assert "title" in assessment.missing_fields

    def test_low_confidence_flags_an_otherwise_complete_record(self):
        metadata = {**COMPLETE, "validation_confidence": LOW_CONFIDENCE - 0.01}
        assessment = assess_metadata_completeness(metadata)

        assert assessment.is_complete          # nothing is missing
        assert assessment.requires_review      # but it is still doubted
        assert any("low_confidence" in item for item in assessment.suspicious_items)

    def test_confidence_at_the_threshold_passes(self):
        metadata = {**COMPLETE, "validation_confidence": LOW_CONFIDENCE}
        assert not assess_metadata_completeness(metadata).requires_review

    def test_absent_confidence_is_unknown_not_good(self):
        """A missing number is not evidence of quality — flag it."""
        metadata = {key: value for key, value in COMPLETE.items()
                    if key != "validation_confidence"}
        assessment = assess_metadata_completeness(metadata)

        assert assessment.confidence == 0.0
        assert assessment.requires_review

    def test_provenance_confidence_wins_over_the_flat_field(self):
        metadata = {**COMPLETE, "provenance": {"validation_confidence": 0.10}}
        assert assess_metadata_completeness(metadata).confidence == pytest.approx(0.10)

    def test_suspicious_authors_are_flagged_even_when_present(self):
        metadata = {**COMPLETE, "provenance": {"suspicious_authors": True,
                                               "validation_confidence": 0.99}}
        assessment = assess_metadata_completeness(metadata)

        assert assessment.is_complete
        assert assessment.suspicious_items == ["authors"]
        assert assessment.requires_review

    def test_garbage_in_is_reviewed_not_raised(self):
        """A document we cannot assess is exactly one a human should see."""
        assessment = assess_metadata_completeness(None)
        assert assessment.requires_review
        assert assessment.missing_fields == list(REQUIRED_FIELDS)

    def test_describe_names_what_is_wrong(self):
        assessment = assess_metadata_completeness({"title": "T"})
        assert "missing" in assessment.describe()
        assert "doi" in assessment.describe()


class TestBuildingTheRecord:
    def test_identity_is_the_file_hash(self, tmp_path):
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4 content")

        record = build_ledger_record(pdf, {k: v for k, v in COMPLETE.items()})
        assert record.document_id == generate_hash(str(pdf))
        assert record.source_hash == record.document_id
        assert record.pdf_filename == "paper.pdf"

    def test_a_supplied_source_hash_is_kept(self, tmp_path):
        """The pipeline already hashed the file; do not disagree with it."""
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF")

        record = build_ledger_record(pdf, {**COMPLETE, "source_hash": "deadbeef"})
        assert record.document_id == "deadbeef"

    def test_an_unreadable_file_still_produces_a_record(self, tmp_path):
        record = build_ledger_record(tmp_path / "gone.pdf", dict(COMPLETE))
        assert record.document_id.startswith("path:")

    def test_the_flag_and_the_missing_fields_are_stored(self, tmp_path):
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF")
        metadata = {key: value for key, value in COMPLETE.items() if key != "doi"}

        record = build_ledger_record(pdf, metadata)
        assert record.requires_review
        assert record.review_needed_fields == ["doi"]
        assert record.review_status == "provisional"

    def test_authors_given_as_a_string_become_a_list(self, tmp_path):
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF")

        record = build_ledger_record(pdf, {**COMPLETE, "authors": "Fullerton, Cameron"})
        assert record.authors == ["Fullerton", "Cameron"]

    @pytest.mark.parametrize("value,expected", [
        (2006, 2006), ("2006", 2006), ("2006-05-01", 2006), (None, None), ("n/a", None),
    ])
    def test_year_coercion(self, tmp_path, value, expected):
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF")

        record = build_ledger_record(pdf, {**COMPLETE, "year": value})
        assert record.year == expected


class TestNeedsReview:
    def test_a_reviewed_record_is_never_re_flagged(self):
        record = make_record(review_status="reviewed", requires_review=True)
        assert not record_needs_review(record)

    def test_the_stored_flag_is_believed(self):
        assert record_needs_review(make_record(requires_review=True))

    def test_a_legacy_row_with_no_flag_is_judged_on_its_fields(self):
        """V5 rows, and V6 rows written before the flag existed."""
        assert record_needs_review(make_record(doi=None))
        assert record_needs_review(make_record(authors=[]))
        assert not record_needs_review(make_record())

    def test_enumeration_survives_a_broken_ledger(self):
        class Broken:
            def get_all_records(self):
                raise OSError("no")

        assert records_needing_review(Broken()) == []

    def test_only_the_flagged_come_back(self):
        ledger = FakeLedger([make_record(document_id="a"),
                             make_record(document_id="b", doi=None)])
        assert [r.document_id for r in records_needing_review(ledger)] == ["b"]


class TestSaving:
    def test_a_saved_record_is_committed(self):
        ledger = FakeLedger()
        assert save_record(ledger, make_record())
        assert ledger.commits == 1

    def test_a_locked_ledger_does_not_end_the_run(self):
        ledger = FakeLedger()
        ledger.fail_on_update = True
        assert save_record(ledger, make_record()) is False

    def test_no_ledger_is_not_an_error(self):
        assert save_record(None, make_record()) is False


class TestRegeneration:
    def write_output(self, tmp_path: Path) -> Path:
        payload = {
            "document_info": {"filename": "paper.pdf"},
            "metadata_final": {"title": "Wrong title", "authors": [], "doi": None,
                               "journal": None, "isbn": "9781234567890"},
            "pages": [{"page_number": 1, "content": "page one text"},
                      {"page_number": 2, "content": "page two text"}],
            "summary": {"errors": [], "page_count": 2},
            "profile": {"processing_result": {"metadata": {"title": "Wrong title"}}},
        }
        path = tmp_path / "paper.v6.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_pages_survive_a_correction(self, tmp_path):
        """THE V5 DEFECT. Fixing a title must not delete the document."""
        path = self.write_output(tmp_path)

        assert regenerate_document_json(path, make_record())

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert len(payload["pages"]) == 2
        assert payload["pages"][1]["content"] == "page two text"

    def test_the_correction_lands_in_the_metadata(self, tmp_path):
        path = self.write_output(tmp_path)
        regenerate_document_json(path, make_record())

        metadata = json.loads(path.read_text(encoding="utf-8"))["metadata_final"]
        assert metadata["title"] == COMPLETE["title"]
        assert metadata["authors"] == COMPLETE["authors"]
        assert metadata["doi"] == COMPLETE["doi"]

    def test_untouched_fields_are_preserved(self, tmp_path):
        """Only what a human can edit is replaced; the rest is left alone."""
        path = self.write_output(tmp_path)
        regenerate_document_json(path, make_record())

        metadata = json.loads(path.read_text(encoding="utf-8"))["metadata_final"]
        assert metadata["isbn"] == "9781234567890"

    def test_the_nested_profile_copy_is_kept_in_step(self, tmp_path):
        """The file carries metadata twice; it must not contradict itself."""
        path = self.write_output(tmp_path)
        regenerate_document_json(path, make_record())

        payload = json.loads(path.read_text(encoding="utf-8"))
        nested = payload["profile"]["processing_result"]["metadata"]
        assert nested["title"] == COMPLETE["title"]

    def test_the_review_verdict_is_written_into_the_file(self, tmp_path):
        path = self.write_output(tmp_path)
        regenerate_document_json(path, make_record(requires_review=True,
                                                   review_status="provisional",
                                                   review_needed_fields=["doi"]))

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["review_status"] == "provisional"
        assert payload["review_needed_fields"] == ["doi"]

    def test_a_missing_file_is_reported_not_invented(self, tmp_path):
        path = tmp_path / "never_written.json"
        assert regenerate_document_json(path, make_record()) is False
        assert not path.exists()

    def test_an_unreadable_file_is_left_exactly_as_it_was(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{not json at all", encoding="utf-8")

        assert regenerate_document_json(path, make_record()) is False
        assert path.read_text(encoding="utf-8") == "{not json at all"

    def test_a_batch_is_regenerated_from_the_ledger(self, tmp_path):
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF")
        path = self.write_output(tmp_path)
        ledger = FakeLedger([make_record(document_id=generate_hash(str(pdf)))])

        assert regenerate_from_ledger({pdf: path}, ledger) == 1
        assert json.loads(path.read_text(encoding="utf-8"))["metadata_final"]["title"] \
            == COMPLETE["title"]

    def test_a_document_with_no_ledger_row_is_skipped(self, tmp_path):
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF")
        path = self.write_output(tmp_path)

        assert regenerate_from_ledger({pdf: path}, FakeLedger()) == 0
        assert json.loads(path.read_text(encoding="utf-8"))["metadata_final"]["title"] \
            == "Wrong title"


_BASELINE = Path(__file__).parent / "baselines" / "nmr_known_good_document.json"


# A release gate: the archived run holds a copyrighted paper's text, so it is
# not published, and this check runs on the maintainer's machine only.
@pytest.mark.skipif(not _BASELINE.is_file(),
                    reason="release gate: the archived NMR run is not published")
class TestParityIsUnharmed:
    """
    Recording a document rewrites its JSON. That must not move the parity gate.

    Session 9 made every run patch its own output with the ledger's view of the
    metadata. The parity harness reads exactly those fields, so the cheap way to
    know the change is safe — without a GPU or a clean card — is to run the
    patch over the ARCHIVED known-good document and re-run the harness's own
    checks against the result.
    """

    BASELINE = Path(__file__).parent / "baselines" / "nmr_known_good_document.json"

    def test_a_recorded_baseline_still_passes_the_harness(self, tmp_path):
        from tests.test_parity_nmr import build_report

        copied = tmp_path / "recorded.json"
        copied.write_text(self.BASELINE.read_text(encoding="utf-8"), encoding="utf-8")
        metadata = json.loads(copied.read_text(encoding="utf-8"))["metadata_final"]

        record = build_ledger_record(tmp_path / "benchmark.pdf", metadata)
        assert regenerate_document_json(copied, record)

        before = build_report(json.loads(self.BASELINE.read_text(encoding="utf-8")))
        after = build_report(json.loads(copied.read_text(encoding="utf-8")))

        assert {check.name for check in after.checks if not check.passed} \
            == {check.name for check in before.checks if not check.passed}

    def test_the_benchmark_metadata_is_byte_identical_after_recording(self, tmp_path):
        copied = tmp_path / "recorded.json"
        copied.write_text(self.BASELINE.read_text(encoding="utf-8"), encoding="utf-8")
        before = json.loads(copied.read_text(encoding="utf-8"))["metadata_final"]

        record = build_ledger_record(tmp_path / "benchmark.pdf", before)
        regenerate_document_json(copied, record)
        after = json.loads(copied.read_text(encoding="utf-8"))["metadata_final"]

        for field in ("title", "authors", "doi", "journal", "year",
                      "validation_source", "validation_confidence"):
            assert after.get(field) == before.get(field), field


class TestApplyingARecord:
    def test_only_the_editable_fields_are_replaced(self):
        patched = apply_record_to_metadata(make_record(),
                                           {"publisher": "Elsevier", "title": "old"})
        assert patched["publisher"] == "Elsevier"
        assert patched["title"] == COMPLETE["title"]

    def test_a_missing_metadata_block_is_not_fatal(self):
        assert apply_record_to_metadata(make_record(), None)["doi"] == COMPLETE["doi"]


class TestTheConversation:
    def test_editing_each_field(self):
        record = make_record(title=None, authors=[], doi=None, journal=None)
        answers = [pick("title"), "A better title",
                   pick("authors"), "Fullerton, Cameron",
                   pick("journal"), "Cell Biology International",
                   pick("doi"), "10.1016/j.cellbi.2005.09.009",
                   pick("year"), "2006",
                   DONE_CHOICE]

        updated = review_record(record, input_fn=scripted(answers), output_fn=lambda *_: None)

        assert updated.title == "A better title"
        assert updated.authors == ["Fullerton", "Cameron"]
        assert updated.journal == "Cell Biology International"
        assert updated.doi == "10.1016/j.cellbi.2005.09.009"
        assert updated.year == 2006

    def test_a_completed_record_is_marked_reviewed(self):
        record = make_record(title=None, requires_review=True,
                             review_needed_fields=["title"])
        updated = review_record(record, input_fn=scripted([pick("title"), "A title", DONE_CHOICE]),
                                output_fn=lambda *_: None)

        assert updated.review_status == "reviewed"
        assert updated.requires_review is False
        assert updated.review_needed_fields == []
        assert updated.last_reviewed_by == "cli"

    def test_a_reviewed_record_is_not_left_contradicting_itself(self):
        """
        Found by an end-to-end run: status said 'reviewed' while the flag still
        said 'requires_review', because the pipeline's OLD low confidence was
        being re-applied after a human had already answered that doubt. A GUI
        reading the flag would show a corrected record as still pending, and no
        number of corrections would ever clear it.
        """
        record = make_record(title=None, validation_confidence=0.0,
                             requires_review=True, review_needed_fields=["title"])
        updated = review_record(record, input_fn=scripted([pick("title"), "A title", DONE_CHOICE]),
                                output_fn=lambda *_: None)

        assert updated.review_status == "reviewed"
        assert updated.requires_review is False
        assert not record_needs_review(updated)

    def test_saving_something_still_incomplete_stays_provisional(self):
        """You may save an incomplete record — but it is not 'reviewed'."""
        record = make_record(doi=None)
        updated = review_record(record, input_fn=scripted([DONE_CHOICE, "y"]),
                                output_fn=lambda *_: None)

        assert updated.review_status == "provisional"
        assert updated.review_needed_fields == ["doi"]

    def test_declining_to_save_incomplete_returns_to_the_menu(self):
        record = make_record(doi=None)
        answers = [DONE_CHOICE, "n",               # not yet
                   pick("doi"), "10.1/x",          # fix it
                   DONE_CHOICE]                    # now done
        updated = review_record(record, input_fn=scripted(answers),
                                output_fn=lambda *_: None)

        assert updated.doi == "10.1/x"
        assert updated.review_status == "reviewed"

    def test_blank_keeps_the_current_value(self):
        updated = review_record(make_record(), input_fn=scripted([pick("title"), "", DONE_CHOICE]),
                                output_fn=lambda *_: None)
        assert updated.title == COMPLETE["title"]

    def test_blank_clears_the_doi_because_a_wrong_one_is_worse(self):
        """A wrong DOI fetches the wrong paper from Zotero. Removal must be possible."""
        updated = review_record(make_record(), input_fn=scripted([pick("doi"), "", DONE_CHOICE, "y"]),
                                output_fn=lambda *_: None)
        assert updated.doi is None

    def test_a_bad_year_is_refused_not_guessed(self):
        printed = []
        updated = review_record(make_record(), input_fn=scripted([pick("year"), "twenty-six", DONE_CHOICE]),
                                output_fn=printed.append)

        assert updated.year == 2006
        assert any("four-digit" in line for line in printed)

    def test_an_unknown_menu_choice_just_asks_again(self):
        printed = []
        updated = review_record(make_record(), input_fn=scripted(["99", DONE_CHOICE]),
                                output_fn=printed.append)

        assert updated is not None
        assert any("Not one of the options" in line for line in printed)

    def test_a_menu_nobody_can_satisfy_stops_instead_of_spinning(self):
        """Something answering forever without ever choosing is not a person."""
        endless = lambda _prompt="": "not an option"

        assert review_record(make_record(), input_fn=endless,
                             output_fn=lambda *_: None) is None

    def test_eof_ends_the_conversation_without_recording_an_answer(self):
        """Nobody at the keyboard must never mean 'reviewed'."""
        assert review_record(make_record(doi=None), input_fn=scripted([]),
                             output_fn=lambda *_: None) is None

    def test_eof_stops_at_once_rather_than_arguing_with_the_pipe(self):
        """EOF is not a wrong answer to be corrected — it is the end."""
        printed = []
        review_record(make_record(), input_fn=scripted([]), output_fn=printed.append)

        assert not any("Not one of the options" in line for line in printed)

    def test_eof_midway_through_a_field_also_stops(self):
        assert review_record(make_record(), input_fn=scripted([pick("title")]),
                             output_fn=lambda *_: None) is None

    def test_review_metadata_returns_the_input_when_nobody_can_answer(self):
        metadata = {"title": "T"}
        assert review_metadata(metadata, input_fn=scripted([]),
                               output_fn=lambda *_: None) == metadata


class TestReviewingTheLedger:
    def test_nothing_flagged_says_so(self):
        printed = []
        assert review_ledger(FakeLedger([make_record()]), input_fn=scripted([]),
                             output_fn=printed.append) == 0
        assert any("Nothing" in line for line in printed)

    def test_each_correction_is_saved_as_it_is_made(self):
        """Stopping halfway keeps the work already done."""
        ledger = FakeLedger([make_record(document_id="a", pdf_filename="a.pdf", title=None),
                             make_record(document_id="b", pdf_filename="b.pdf", title=None)])

        corrected = review_ledger(ledger, input_fn=scripted([pick("title"), "First", DONE_CHOICE,
                                                             pick("title"), "Second", DONE_CHOICE]),
                                  output_fn=lambda *_: None)

        assert corrected == 2
        assert ledger.get_record("a").title == "First"
        assert ledger.get_record("b").title == "Second"

    def test_stopping_halfway_leaves_the_rest_flagged(self):
        ledger = FakeLedger([make_record(document_id="a", title=None),
                             make_record(document_id="b", title=None)])

        corrected = review_ledger(ledger, input_fn=scripted([pick("title"), "First", DONE_CHOICE]),
                                  output_fn=lambda *_: None)

        assert corrected == 1
        assert record_needs_review(ledger.get_record("b"))


class TestTheCliWiring:
    """`run_prajnya.py`'s side: every document is recorded, nothing prompts mid-batch."""

    def processed(self, tmp_path: Path, **metadata):
        from src.processing.models import (
            DocumentProfile, ExtractedMetadata, ProcessingResult,
        )

        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4 body")
        profile = DocumentProfile(
            pdf_path=pdf, page_count=8, file_size=pdf.stat().st_size,
            processing_result=ProcessingResult(
                metadata=ExtractedMetadata(**metadata)))
        return pdf, profile

    def test_a_complete_document_is_recorded_and_not_flagged(self, tmp_path, caplog):
        from run_prajnya import record_document

        pdf, profile = self.processed(
            tmp_path, title=COMPLETE["title"], authors=list(COMPLETE["authors"]),
            doi=COMPLETE["doi"], journal=COMPLETE["journal"],
            validation_confidence=0.95)
        out_path = tmp_path / "paper.v6.json"
        out_path.write_text(json.dumps({"metadata_final": {}, "pages": []}),
                            encoding="utf-8")
        ledger = FakeLedger()

        record = record_document(pdf, profile, out_path, ledger,
                                 __import__("logging").getLogger("test"))

        assert record.requires_review is False
        assert ledger.get_record(generate_hash(str(pdf))) is not None

    def test_an_incomplete_document_is_flagged_in_the_ledger_and_the_json(self, tmp_path):
        from run_prajnya import record_document

        pdf, profile = self.processed(tmp_path, title="Only a title")
        out_path = tmp_path / "paper.v6.json"
        out_path.write_text(json.dumps({"metadata_final": {}, "pages": [{"a": 1}]}),
                            encoding="utf-8")
        ledger = FakeLedger()

        record = record_document(pdf, profile, out_path, ledger,
                                 __import__("logging").getLogger("test"))

        assert record.requires_review
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        assert payload["review_status"] == "provisional"
        assert payload["pages"] == [{"a": 1}]        # still not destroyed

    def test_an_unattended_batch_reports_instead_of_asking(self, tmp_path, caplog):
        """--noninteractive must never reach a prompt, only a log line."""
        import argparse
        import logging

        from run_prajnya import _offer_review

        results = {str(tmp_path / "a.pdf"): {"requires_review": True,
                                             "output": str(tmp_path / "a.v6.json")}}
        args = argparse.Namespace(noninteractive=True)

        with caplog.at_level(logging.INFO):
            _offer_review(results, FakeLedger(), args, logging.getLogger("test"))

        assert any("--review" in message for message in caplog.messages)

    def test_a_batch_with_nothing_flagged_says_nothing(self, tmp_path):
        import argparse
        import logging

        from run_prajnya import _offer_review

        results = {str(tmp_path / "a.pdf"): {"requires_review": False,
                                             "output": str(tmp_path / "a.v6.json")}}
        args = argparse.Namespace(noninteractive=False)

        # No ledger, no prompt, no exception — there is simply nothing to ask.
        _offer_review(results, None, args, logging.getLogger("test"))

    def test_a_document_is_recorded_even_with_no_ledger(self, tmp_path):
        """Losing the bookkeeping file must not lose the run."""
        from run_prajnya import record_document

        pdf, profile = self.processed(tmp_path, title="T")
        out_path = tmp_path / "paper.v6.json"
        out_path.write_text(json.dumps({"metadata_final": {}}), encoding="utf-8")

        record = record_document(pdf, profile, out_path, None,
                                 __import__("logging").getLogger("test"))
        assert record.pdf_filename == "paper.pdf"


class TestDeferredReview:
    def test_nothing_pending_asks_nothing(self):
        assert offer_deferred_review([], FakeLedger(), input_fn=scripted([]),
                                     output_fn=lambda *_: None) is False

    def test_declining_leaves_everything_flagged(self):
        ledger = FakeLedger([make_record(title=None)])
        assert offer_deferred_review([Path("a.pdf")], ledger, input_fn=scripted(["n"]),
                                     output_fn=lambda *_: None) is False
        assert record_needs_review(ledger.get_all_records()[0])

    def test_accepting_runs_the_review(self):
        ledger = FakeLedger([make_record(document_id="a", title=None)])
        assert offer_deferred_review([Path("a.pdf")], ledger,
                                     input_fn=scripted(["y", pick("title"), "Fixed", DONE_CHOICE]),
                                     output_fn=lambda *_: None) is True
        assert ledger.get_record("a").title == "Fixed"

    def test_an_unattended_run_cannot_be_hung_by_the_offer(self):
        """EOF at the offer means 'no', not 'wait forever'."""
        assert offer_deferred_review([Path("a.pdf")], FakeLedger(),
                                     input_fn=scripted([]),
                                     output_fn=lambda *_: None) is False

    def test_the_pending_files_are_named(self):
        printed = []
        offer_deferred_review([Path("one.pdf"), Path("two.pdf")], FakeLedger(),
                              input_fn=scripted(["n"]), output_fn=printed.append)
        assert any("one.pdf" in line for line in printed)
        assert any("two.pdf" in line for line in printed)
