"""
Citation levels, stage S0b: what the level changes in the metadata step.

For biology researchers: the level (`local`, `zotero`, `online`) decides who
is asked to confirm a paper's bibliographic record. These tests pin:

  1. at `local`, confidence comes from how well Prajnya's own two readings
     of the title page agree, so a well-read paper is NOT flagged for review
     (before this rule every `local` document scored 0.0 and was flagged);
  2. a record with nothing to cite by is cited by its file name, flagged,
     and the file name stays out of the title;
  3. a ledger row written at one level is not reused at another;
  4. the DOI's origin survives the merge (it used to be dropped);
  5. the level travels: command line -> request -> child process -> ledger.

The corpus-level check is `eval/citation_levels/gate_s0b.py`.
"""
import logging

import pytest

from src.processing.citations.level import BUILT_LEVELS, require_built
from src.processing.citations.local_confidence import (first_author_is_on_page,
                                                       has_confident_text_layer_doi,
                                                       local_confidence,
                                                       title_is_on_page)
from src.processing.ledger.csv_backend import CsvLedgerBackend
from src.processing.ledger.semantic_ledger import LedgerRecord
from src.processing.review import (assess_metadata_completeness,
                                   build_ledger_record)
from src.processing.run_request import RunRequest
from src.processing.strategies.research_article.metadata_enrichment import (
    apply_citation_level, enrich_metadata)
from src.processing.utils.metadata_merger import merge_metadata

LOG = logging.getLogger("test")

TITLE = "An NMR method to characterize multiple water compartments on mammalian collagen"
PAGE = ("Cell Biology International 30 (2006) 66-73\n"
        "An NMR method to characterize multiple water\ncompartments on mammalian collagen\n"
        "Gary D. Fullerton*, Elena Nes, Maxwell Amurao\ndoi:10.1016/j.cellbi.2005.09.009")
GOOD_DOI = {"doi": "10.1016/j.cellbi.2005.09.009", "doi_source": "pymupdf_text_layer",
            "doi_confidence": 0.98}


# --- 1. confidence at level `local` -------------------------------------------------

class TestLocalConfidence:
    def test_a_labelled_doi_and_a_title_on_the_page_is_confirmed(self):
        merged = {"title": TITLE, "authors": ["Gary D. Fullerton"], **GOOD_DOI}
        assert local_confidence(merged, PAGE) == (0.80, "text_layer")

    def test_title_and_first_author_without_a_doi_just_passes(self):
        """0.75 is exactly `review.LOW_CONFIDENCE`, which flags only BELOW it."""
        merged = {"title": TITLE, "authors": ["Gary D. Fullerton"]}
        assert local_confidence(merged, PAGE) == (0.75, "text_layer")

    def test_a_bare_doi_from_the_body_does_not_confirm(self):
        """A 0.85 DOI may belong to a CITED paper."""
        merged = {"title": TITLE, "authors": ["Someone Else"], **GOOD_DOI,
                  "doi_confidence": 0.85}
        assert local_confidence(merged, PAGE) == (0.60, "vlm_extraction")

    def test_a_doi_the_vision_model_read_does_not_confirm(self):
        merged = {"title": TITLE, "authors": ["Nobody Here"], **GOOD_DOI,
                  "doi_source": "vlm_extraction"}
        assert local_confidence(merged, PAGE)[0] == 0.60

    def test_a_title_that_is_not_on_the_page_is_unconfirmed(self):
        merged = {"title": "A completely different invented paper title",
                  "authors": ["Gary D. Fullerton"], **GOOD_DOI}
        assert local_confidence(merged, PAGE) == (0.60, "vlm_extraction")

    def test_nothing_at_all_is_unconfirmed(self):
        assert local_confidence({}, "") == (0.60, "vlm_extraction")

    @pytest.mark.parametrize("title, expected", [
        (TITLE, True),
        (TITLE.upper(), True),                       # set in capitals on the page
        ("An NMR method", True),
        ("NMR", False),                              # too short to mean anything
        (None, False),
    ])
    def test_title_on_page(self, title, expected):
        assert title_is_on_page(title, PAGE) is expected

    @pytest.mark.parametrize("authors, expected", [
        (["Gary D. Fullerton"], True), (["Fullerton, Gary D."], True),
        (["Jordi Bella"], False), ([], False), (None, False),
    ])
    def test_first_author_on_page(self, authors, expected):
        assert first_author_is_on_page(authors, PAGE) is expected

    def test_accents_are_folded(self):
        assert first_author_is_on_page(["Marc André Meyers"], "MARC ANDRE MEYERS")

    def test_a_confident_doi_needs_all_three_things(self):
        assert has_confident_text_layer_doi(GOOD_DOI)
        assert not has_confident_text_layer_doi({**GOOD_DOI, "doi": None})
        assert not has_confident_text_layer_doi({**GOOD_DOI, "doi_confidence": "x"})


class TestALocalDocumentIsNotFlaggedForNoReason:
    """THE DEFECT S0b FIXES: every `local` document used to score 0.0."""

    def _enrich(self, tmp_path, level):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-fake")
        extracted = {"title": TITLE, "authors": ["Gary D. Fullerton", "Elena Nes"],
                     "doi": "10.1016/j.cellbi.2005.09.009", "year": 2006,
                     "journal": "Cell Biology International"}

        class TextLayer:
            doi, doi_confidence = "10.1016/j.cellbi.2005.09.009", 0.98

        return enrich_metadata(extracted, pdf, title_page_text=PAGE, profile=TextLayer(),
                               citation_level=level, logger=LOG)

    def test_at_local_a_well_read_paper_is_complete(self, tmp_path):
        merged = self._enrich(tmp_path, "local")
        assert merged["validation_confidence"] == 0.80
        assert merged["validation_source"] == "text_layer"
        assert merged["citation_level"] == "local"
        assert not assess_metadata_completeness(merged, "research_article").requires_review

    def test_with_no_level_named_nothing_changes(self, tmp_path):
        """Callers that never name a level keep the behaviour from before levels."""
        merged = self._enrich(tmp_path, None)
        assert merged["validation_confidence"] == 0.0
        assert merged["validation_source"] == "unknown"
        assert "citation_level" not in merged


# --- 2. the file-name fallback -------------------------------------------------------

class TestTheFileNameFallback:
    @pytest.mark.parametrize("level", BUILT_LEVELS)
    def test_nothing_to_cite_by_is_flagged_at_every_level(self, level):
        merged = {"title": None, "authors": [], "validation_confidence": 0.95,
                  "validation_source": "zotero_api"}
        apply_citation_level(merged, level, "", "scan0042.pdf", logger=LOG)
        assert merged["validation_source"] == "file_name"
        assert merged["validation_confidence"] == 0.0
        assert assess_metadata_completeness(merged, "research_article").requires_review

    def test_the_file_name_is_not_written_into_the_record(self):
        """Researcher's decision, session 32: it lives in `source_label` only."""
        merged = {"title": None, "authors": []}
        apply_citation_level(merged, "local", "", "JRuberti.pdf", logger=LOG)
        assert merged["title"] is None
        assert "JRuberti" not in repr(merged)

    def test_at_zotero_a_found_record_is_left_exactly_as_zotero_set_it(self):
        merged = {"title": TITLE, "authors": ["Gary D. Fullerton"],
                  "validation_confidence": 0.95, "validation_source": "zotero_api"}
        apply_citation_level(merged, "zotero", PAGE, "nmr.pdf", logger=LOG)
        assert (merged["validation_confidence"], merged["validation_source"]) == (
            0.95, "zotero_api")


# --- 3. a ledger row from another level is not reused -------------------------------

class _Ledger:
    def __init__(self, level):
        self.level = level

    def get_record(self, document_id):
        ledger = self

        class Row:
            citation_level = ledger.level

            @staticmethod
            def model_dump():
                return {"title": "Stored", "citation_level": ledger.level}
        return Row()


class TestTheLedgerRowBelongsToItsLevel:
    def _enrich(self, tmp_path, stored_level, level):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-fake")
        return enrich_metadata({"title": TITLE, "authors": ["Gary D. Fullerton"]}, pdf,
                               title_page_text=PAGE, ledger=_Ledger(stored_level),
                               citation_level=level, logger=LOG)

    def test_a_row_from_the_same_level_is_reused(self, tmp_path):
        assert self._enrich(tmp_path, "local", "local")["title"] == "Stored"

    def test_a_row_from_another_level_is_derived_again(self, tmp_path):
        """A `local` row was never checked against Zotero; it must not pass as one."""
        assert self._enrich(tmp_path, "local", "zotero")["title"] == TITLE

    def test_a_row_from_before_levels_is_derived_again(self, tmp_path):
        assert self._enrich(tmp_path, None, "local")["title"] == TITLE


# --- 4. the DOI's origin survives the merge -------------------------------------------

class TestTheMergeKeepsWhereTheDoiCameFrom:
    def test_kept_when_the_merged_doi_is_the_extracted_one(self):
        merged = merge_metadata({"title": TITLE, **GOOD_DOI}, {}, PAGE)
        assert merged["doi_source"] == "pymupdf_text_layer"
        assert merged["doi_confidence"] == 0.98

    def test_dropped_when_the_canonical_record_supplied_a_different_doi(self):
        """It described the extracted DOI, which is no longer the record's."""
        merged = merge_metadata({"title": TITLE, **GOOD_DOI}, {"doi": "10.1000/other"}, PAGE)
        assert merged["doi"] == "10.1000/other"
        assert "doi_source" not in merged


# --- 5. the level travels ---------------------------------------------------------------

class TestTheLevelTravels:
    def test_through_a_request_written_for_a_child_process(self):
        request = RunRequest(citations="zotero")
        assert RunRequest.from_dict(request.to_dict()).citations == "zotero"

    def test_absent_means_the_config_decides(self):
        assert RunRequest.from_dict(RunRequest().to_dict()).citations is None

    def test_from_the_command_line(self, tmp_path):
        from types import SimpleNamespace
        args = SimpleNamespace(pdf=str(tmp_path / "a.pdf"), out=None, backend=None, pages=None,
                               chapters=None, max_pages=None, front_matter_pages=10,
                               force=False, recursive=False, noninteractive=True,
                               verbose=False, citations="local")
        assert RunRequest.from_args(args).citations == "local"

    def test_into_the_ledger_row_and_back_from_disk(self, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-fake")
        record = build_ledger_record(pdf, {"title": TITLE, "authors": ["A B"],
                                           "validation_confidence": 0.8,
                                           "validation_source": "text_layer",
                                           "citation_level": "local"})
        assert record.citation_level == "local"

        ledger = CsvLedgerBackend(file_path=str(tmp_path / "ledger.csv"))
        ledger.update_record(record)
        ledger.commit()
        reread = CsvLedgerBackend(file_path=str(tmp_path / "ledger.csv"))
        assert reread.get_record(record.document_id).citation_level == "local"

    def test_a_ledger_written_before_levels_still_loads(self, tmp_path):
        """An existing ledger has no `citation_level` column."""
        path = tmp_path / "old.csv"
        fields = [name for name in LedgerRecord.model_fields if name != "citation_level"]
        row = {name: "" for name in fields}
        row.update(document_id="h", pdf_filename="a.pdf", validation_confidence="0.95",
                   validation_source="zotero_api", last_updated="2026-08-01", source_hash="h",
                   requires_review="False", manual_entry="False",
                   review_status="reviewed", sync_status="NOT_PROCESSED")
        path.write_text(",".join(fields) + "\n" + ",".join(row[name] for name in fields) + "\n",
                        encoding="utf-8")
        record = CsvLedgerBackend(file_path=str(path)).get_record("h")
        assert record is not None and record.citation_level is None


class TestLevelsNotBuiltYet:
    def test_online_is_built_since_stage_s1(self):
        assert require_built("online") == "online"

    def test_a_level_that_is_not_built_is_refused_with_a_plain_message(self):
        with pytest.raises(NotImplementedError, match="local, zotero, online"):
            require_built("something_new")

    @pytest.mark.parametrize("level", BUILT_LEVELS)
    def test_built_levels_pass(self, level):
        assert require_built(level) == level
