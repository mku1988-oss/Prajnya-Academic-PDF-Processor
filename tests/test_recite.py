"""
`--recite`: change a processed document's citation level without the GPU (S0c).

For biology researchers: re-citing works out a document's bibliographic
record again, at another level, from what its run already read. These tests
pin what would be costly to get wrong on a whole library:

  1. only the record changes; pages, figures and tables are never touched;
  2. a record a person corrected by hand is never replaced;
  3. a value only the old level knew (a publisher from Zotero) does not
     linger in a record derived without it;
  4. a document that cannot be re-cited is reported, and stops nothing;
  5. a batch that skips documents processed at another level says so.

The corpus-level check is `eval/citation_levels/gate_s0c.py`.
"""
import copy
import json
import logging
from pathlib import Path

import pytest

from src.processing.citations.recite import (LEVEL_RECORDED, RECITED, SKIPPED,
                                             recite_document, recite_documents,
                                             title_page_reading)
from src.processing.document_json import CONTRACT_VERSION
from src.processing.ledger.csv_backend import CsvLedgerBackend
from src.processing.ledger.semantic_ledger import LedgerRecord

LOG = logging.getLogger("test")

TITLE = "An NMR method to characterize multiple water compartments on mammalian collagen"
PAGE_TEXT = (f"{TITLE}\nGary D. Fullerton, Elena Nes\n"
             "doi:10.1016/j.cellbi.2005.09.009\nCell Biology International 30 (2006) 66-73")


def _document(**metadata_changes) -> dict:
    """A document as a `zotero` run wrote it under contract 1.0 (no citations block)."""
    metadata = {"title": TITLE, "authors": ["Gary D. Fullerton", "Elena Nes"], "year": 2006,
                "doi": "10.1016/j.cellbi.2005.09.009", "journal": "Cell Biology International",
                "publisher": "Wiley", "isbn": None, "editors": [], "chapter_number": None,
                "validation_source": "zotero_api", "validation_confidence": 0.95,
                "provenance": {}, "source_hash": "hash-1", "last_updated": None}
    metadata.update(metadata_changes)
    return {
        "document_info": {"filename": "nmr.pdf", "document_type": "research_article",
                          "processing_time": 150.0, "success": True,
                          "pipeline_version": "v6.5.0", "contract_version": "1.0"},
        "metadata_final": metadata,
        "pages": [
            {"page_number": 1, "content": PAGE_TEXT, "metadata": {
                "processing_method": "two_shot_title_page", "title": TITLE,
                "authors": ["Gary D. Fullerton", "Elena Nes"],
                "doi": "10.1016/j.cellbi.2005.09.009", "publication_year": 2006,
                "journal_name": "Cell Biol. Int."}},
            {"page_number": 2, "content": "Body text.", "metadata": {
                "processing_method": "pymupdf", "figures": [], "tables": []}},
        ],
        "summary": {"errors": [], "page_count": 2, "figure_count": 0},
        "profile": {"pdf_path": "/somewhere/else/nmr.pdf", "doi": "10.1016/j.cellbi.2005.09.009",
                    "doi_confidence": 0.98, "title_page_index": 0, "page_1_text": PAGE_TEXT,
                    "processing_result": {"chapters": [], "metadata": dict(metadata)}},
    }


def _write(tmp_path: Path, document: dict, name: str = "nmr.v6.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# --- 1. only the record changes ----------------------------------------------------

class TestReciteAtLocal:
    @pytest.fixture
    def recited(self, tmp_path):
        path = _write(tmp_path, _document())
        result = recite_document(path, "local", "cli", logger=LOG)
        return result, _read(path)

    def test_the_record_is_derived_from_the_stored_readings(self, recited):
        result, document = recited
        metadata = document["metadata_final"]
        assert result.status == RECITED
        assert metadata["validation_source"] == "text_layer"
        assert metadata["validation_confidence"] == 0.80
        assert metadata["journal"] == "Cell Biol. Int."      # the title page's own reading
        assert metadata["doi_source"] == "pymupdf_text_layer"
        assert metadata["citation_level"] == "local"

    def test_a_value_only_zotero_knew_does_not_linger(self, recited):
        """`publisher: Wiley` came from Zotero; at `local` nobody supplied it."""
        _, document = recited
        assert document["metadata_final"]["publisher"] is None

    def test_what_the_level_does_not_decide_is_carried_over(self, recited):
        _, document = recited
        assert document["metadata_final"]["source_hash"] == "hash-1"

    def test_the_file_gets_its_citations_block_and_the_current_stamp(self, recited):
        _, document = recited
        citations = document["citations"]
        assert (citations["level"], citations["level_set_by"]) == ("local", "cli")
        assert citations["metadata_source"] == "extracted"
        assert citations["source_label"] == "Fullerton and Nes, 2006"
        assert document["document_info"]["contract_version"] == CONTRACT_VERSION
        assert document["document_info"]["pipeline_version"] == "v6.5.0"   # who READ it

    def test_the_nested_copy_agrees(self, recited):
        _, document = recited
        nested = document["profile"]["processing_result"]
        assert nested["metadata"] == document["metadata_final"]
        assert nested["citation_level"] == "local"

    def test_pages_are_never_touched(self, recited, tmp_path):
        """What was read stays as read; only the derived furniture labels and
        `body` (contract 1.3) are added to each page."""
        _, document = recited
        for written, stored in zip(document["pages"], _document()["pages"]):
            metadata = dict(written["metadata"])
            assert isinstance(metadata.pop("furniture"), list)
            assert isinstance(metadata.pop("outside_body"), list)
            assert metadata == stored["metadata"]
            assert written["body"] == written["content"] == stored["content"]
            assert {k: v for k, v in written.items() if k not in ("body", "metadata")}                 == {k: v for k, v in stored.items() if k != "metadata"}
        assert len(document["pages"]) == len(_document()["pages"])
        assert document["summary"] == _document()["summary"]

    def test_the_result_reports_what_changed(self, recited):
        result, _ = recited
        assert result.level_before is None and result.level_after == "local"
        assert result.changed["journal"] == ["Cell Biology International", "Cell Biol. Int."]
        assert result.changed["publisher"] == ["Wiley", None]
        assert not result.requires_review

    def test_the_pdf_does_not_need_to_exist(self, recited):
        """`profile.pdf_path` points nowhere; only the stored readings are used."""
        result, _ = recited
        assert result.status == RECITED

    def test_no_temporary_file_is_left_behind(self, tmp_path):
        path = _write(tmp_path, _document())
        recite_document(path, "local", "cli", logger=LOG)
        assert [p.name for p in tmp_path.iterdir()] == ["nmr.v6.json"]

    def test_twice_gives_the_same_file(self, tmp_path):
        path = _write(tmp_path, _document())
        recite_document(path, "local", "cli", logger=LOG)
        once = path.read_bytes()
        recite_document(path, "local", "cli", logger=LOG)
        assert path.read_bytes() == once


# --- 2. records that are kept ----------------------------------------------------------

class TestRecordsThatAreKept:
    def test_a_hand_correction_is_never_replaced(self, tmp_path):
        document = _document(title="The title a person typed", validation_source="user_corrected")
        path = _write(tmp_path, document)

        result = recite_document(path, "local", "config", logger=LOG)

        after = _read(path)
        assert result.status == LEVEL_RECORDED and "by hand" in result.reason
        assert after["metadata_final"]["title"] == "The title a person typed"
        assert after["metadata_final"]["publisher"] == "Wiley"
        assert after["citations"]["level"] == "local"
        assert after["citations"]["metadata_source"] == "user"
        assert result.changed == {}

    def test_a_book_keeps_its_front_matter_record(self, tmp_path):
        document = _document(authors=[], editors=["Jaroslava Halper"],
                             validation_source="front_matter_text_layer")
        document["document_info"]["document_type"] = "book"
        path = _write(tmp_path, document)

        result = recite_document(path, "zotero", "config", logger=LOG)

        after = _read(path)
        assert result.status == LEVEL_RECORDED and "book" in result.reason
        assert after["metadata_final"]["validation_source"] == "front_matter_text_layer"
        assert after["citations"]["source_label"] == "Halper (ed.), 2006"

    def test_a_reviewed_ledger_row_keeps_its_reviewer(self, tmp_path):
        """Rebuilding the row from the metadata would erase who reviewed it."""
        path = _write(tmp_path, _document(validation_source="user_corrected"))
        ledger = CsvLedgerBackend(file_path=str(tmp_path / "ledger.csv"))
        ledger.update_record(LedgerRecord(
            document_id="hash-1", pdf_filename="nmr.pdf", title=TITLE,
            validation_confidence=0.95, validation_source="user_corrected",
            last_reviewed_by="gui", review_status="reviewed", last_updated="2026-11-01",
            source_hash="hash-1"))
        ledger.commit()

        recite_document(path, "local", "config", ledger=ledger, logger=LOG)

        row = CsvLedgerBackend(file_path=str(tmp_path / "ledger.csv")).get_record("hash-1")
        assert row.last_reviewed_by == "gui"
        assert row.review_status == "reviewed"
        assert row.citation_level == "local"


# --- 3. the ledger follows a re-cited record ---------------------------------------------

class TestTheLedgerFollows:
    def test_the_row_carries_the_new_level_and_source(self, tmp_path):
        path = _write(tmp_path, _document())
        ledger = CsvLedgerBackend(file_path=str(tmp_path / "ledger.csv"))

        recite_document(path, "local", "config", ledger=ledger, logger=LOG)

        row = CsvLedgerBackend(file_path=str(tmp_path / "ledger.csv")).get_record("hash-1")
        assert row.citation_level == "local"
        assert row.validation_source == "text_layer"
        assert row.journal == "Cell Biol. Int."

    def test_a_record_that_now_lacks_a_field_is_flagged(self, tmp_path):
        document = _document()
        document["pages"][0]["metadata"]["journal_name"] = None
        path = _write(tmp_path, document)

        result = recite_document(path, "local", "config", logger=LOG)

        assert result.requires_review
        after = _read(path)
        assert after["review_status"] == "provisional"
        assert after["review_needed_fields"] == ["journal"]


# --- 4. what cannot be re-cited is reported ------------------------------------------------

class TestWhatCannotBeRecited:
    def test_a_failed_title_page_is_skipped_and_the_file_untouched(self, tmp_path):
        document = _document()
        document["pages"][0]["metadata"] = {"processing_method": "two_shot_title_failed",
                                            "error": "HTTP 500"}
        path = _write(tmp_path, document)
        before = path.read_bytes()

        result = recite_document(path, "local", "config", logger=LOG)

        assert result.status == SKIPPED and "title-page reading" in result.reason
        assert path.read_bytes() == before

    def test_a_run_without_the_title_page_is_skipped(self, tmp_path):
        """`--pages 4-6` wrote a file with no page 1 in it."""
        document = _document()
        document["pages"] = document["pages"][1:]
        assert title_page_reading(document) is None

    def test_a_file_that_is_not_a_document_is_skipped(self, tmp_path):
        path = tmp_path / "batch_summary.v6.json"
        path.write_text('{"something": "else"}', encoding="utf-8")
        assert recite_document(path, "local", "config", logger=LOG).status == SKIPPED

    def test_one_bad_file_does_not_stop_the_others(self, tmp_path):
        broken = tmp_path / "a_broken.v6.json"
        broken.write_text("{not json", encoding="utf-8")
        good = _write(tmp_path, _document())

        results = recite_documents([broken, good], {"citations": {"level": "local"}}, logger=LOG)

        assert [r.status for r in results] == [SKIPPED, RECITED]

    def test_a_level_that_is_not_built_touches_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.processing.citations.level.BUILT_LEVELS", ("local", "zotero"))
        path = _write(tmp_path, _document())
        before = path.read_bytes()
        with pytest.raises(NotImplementedError):
            recite_documents([path], {}, override="online", logger=LOG)
        assert path.read_bytes() == before

    def test_the_nested_title_page_shape_is_read_too(self):
        document = _document()
        flat = copy.deepcopy(document["pages"][0]["metadata"])
        document["pages"][0]["metadata"] = {"processing_method": "two_shot_title_page",
                                            "metadata": flat}
        assert title_page_reading(document)["title"] == TITLE


# --- 5. a batch says when skipped documents sit at another level ----------------------------

class TestABatchSaysWhenLevelsDiffer:
    def _skip(self, tmp_path, stored_level, run_level):
        from src.processing.ledger.semantic_ledger import generate_hash
        from src.processing.runner import _skipped_result

        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-fake")
        ledger = CsvLedgerBackend(file_path=str(tmp_path / "ledger.csv"))
        ledger.update_record(LedgerRecord(
            document_id=generate_hash(str(pdf)), pdf_filename="doc.pdf",
            validation_confidence=0.9, validation_source="x", citation_level=stored_level,
            last_updated="2026-11-14", source_hash="h"))
        return _skipped_result(pdf, ledger, run_level, LOG)

    def test_same_level_is_an_ordinary_skip(self, tmp_path):
        assert self._skip(tmp_path, "zotero", "zotero") == {"skipped": "already in the ledger"}

    def test_another_level_is_named(self, tmp_path):
        result = self._skip(tmp_path, "local", "zotero")
        assert result["citation_level_differs"] is True
        assert "citation level local" in result["skipped"] and "zotero" in result["skipped"]

    def test_a_row_from_before_levels_is_named_too(self, tmp_path):
        result = self._skip(tmp_path, None, "local")
        assert result["citation_level_differs"] is True
        assert "not recorded" in result["skipped"]

    def test_the_batch_ends_with_one_summary_line(self, caplog):
        from src.processing.runner import _warn_about_other_levels

        results = {"a": {"skipped": "x", "citation_level_differs": True},
                   "b": {"skipped": "already in the ledger"}, "c": {"elapsed": 1.0}}
        with caplog.at_level(logging.WARNING):
            _warn_about_other_levels(results, "zotero", LOG)
        assert "1 document(s)" in caplog.text and "--recite" in caplog.text

    def test_no_line_when_nothing_differs(self, caplog):
        from src.processing.runner import _warn_about_other_levels

        with caplog.at_level(logging.WARNING):
            _warn_about_other_levels({"b": {"skipped": "already in the ledger"}}, "zotero", LOG)
        assert caplog.text == ""


# --- 6. the reference list follows the level ------------------------------------------------

FOUND_LIST = {
    "reference_list": {"status": "found", "source": "crossref", "style": "numbered",
                       "count": 1, "local_count": 1, "counts_agree": True},
    "references": [{"position": 1, "number": 1, "first_author": "Bella", "year": 2016,
                    "title": "T", "container": "J", "doi": None, "source": "crossref",
                    "printed_text": None}],
    "report": {"never": "written to the file"},
}


class FakeFinder:
    """Stands in for `ReferenceListFinder`; remembers what it was asked."""

    def __init__(self):
        self.calls = []

    def find(self, doi, pdf_path):
        self.calls.append((doi, pdf_path))
        return copy.deepcopy(FOUND_LIST)


class TestTheReferenceListFollowsTheLevel:
    def _recite(self, tmp_path, document, finder, level="online"):
        path = _write(tmp_path, document)
        result = recite_document(path, level, "cli", logger=LOG, reference_finder=finder)
        return result, _read(path)

    def test_the_list_is_written_and_the_report_is_not(self, tmp_path):
        finder = FakeFinder()
        result, document = self._recite(tmp_path, _document(), finder)
        citations = document["citations"]
        assert citations["reference_list"]["status"] == "found"
        assert citations["references"] == FOUND_LIST["references"]
        assert "report" not in citations and "report" not in json.dumps(citations)
        assert (result.reference_list_status, result.reference_count) == ("found", 1)

    def test_the_written_block_validates_against_the_contract(self, tmp_path):
        """The block is checked on its own: the fixture document is not a full output."""
        jsonschema = pytest.importorskip("jsonschema")
        schema = json.loads((Path(__file__).resolve().parents[1] / "contract"
                             / "prajnya_document.schema.json").read_text(encoding="utf-8"))
        _, document = self._recite(tmp_path, _document(), FakeFinder())
        block_schema = {"$schema": schema["$schema"], "$defs": schema["$defs"],
                        "$ref": "#/$defs/citations"}
        assert list(jsonschema.Draft202012Validator(block_schema)
                    .iter_errors(document["citations"])) == []

    def test_markers_are_written_with_offsets_into_the_pages(self, tmp_path):
        class MarkerFinder(FakeFinder):
            def find(self, doi, pdf_path):
                found = super().find(doi, pdf_path)
                found["markers"] = [{"page_number": 2, "text": "[1]", "kind": "bracket",
                                     "numbers": [1], "context_before": "Body text.",
                                     "content_offset": None}]
                return found
        document = _document()
        document["pages"][1]["content"] = "Body text. [1]"
        _, after = self._recite(tmp_path, document, MarkerFinder())
        assert after["citations"]["markers"][0]["content_offset"] == len("Body text. ")

    def test_without_a_finder_the_block_is_not_requested(self, tmp_path):
        result, document = self._recite(tmp_path, _document(), None, level="zotero")
        assert document["citations"]["reference_list"] == {"status": "not_requested"}
        assert document["citations"]["references"] == []
        assert (result.reference_list_status, result.reference_count) == (None, 0)

    def test_reciting_down_removes_an_old_list(self, tmp_path):
        path = _write(tmp_path, _document())
        recite_document(path, "online", "cli", logger=LOG, reference_finder=FakeFinder())
        recite_document(path, "local", "cli", logger=LOG, reference_finder=None)
        assert _read(path)["citations"]["references"] == []

    def test_a_book_is_not_asked(self, tmp_path):
        document = _document(authors=[], editors=["Jaroslava Halper"],
                             validation_source="front_matter_text_layer")
        document["document_info"]["document_type"] = "book"
        finder = FakeFinder()
        _, after = self._recite(tmp_path, document, finder)
        assert finder.calls == []
        assert after["citations"]["reference_list"] == {"status": "not_requested"}

    def test_a_missing_pdf_passes_none(self, tmp_path):
        finder = FakeFinder()
        self._recite(tmp_path, _document(), finder)
        assert finder.calls[0][1] is None
        assert finder.calls[0][0] == "10.1016/j.cellbi.2005.09.009"

    def test_an_existing_pdf_is_passed(self, tmp_path):
        pdf = tmp_path / "nmr.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        document = _document()
        document["profile"]["pdf_path"] = str(pdf)
        finder = FakeFinder()
        self._recite(tmp_path, document, finder)
        assert finder.calls[0][1] == pdf

    def test_a_hand_corrected_record_still_gets_its_list(self, tmp_path):
        document = _document(title="Typed by a person", validation_source="user_corrected")
        finder = FakeFinder()
        result, after = self._recite(tmp_path, document, finder)
        assert result.status == LEVEL_RECORDED
        assert after["citations"]["references"] == FOUND_LIST["references"]

    def test_recite_documents_builds_the_finder_once(self, tmp_path, monkeypatch):
        finder = FakeFinder()
        built = []
        monkeypatch.setattr("src.processing.citations.recite.reference_finder_for",
                            lambda level, config, logger=None: built.append(level) or finder)
        monkeypatch.setattr("src.processing.citations.recite.online_chain_for",
                            lambda *args, **kwargs: None)
        monkeypatch.setattr("src.processing.citations.recite.zotero_service_for",
                            lambda *args, **kwargs: None)
        paths = [_write(tmp_path, _document(), "a.v6.json"), _write(tmp_path, _document(), "b.v6.json")]
        recite_documents(paths, {}, override="online", logger=LOG)
        assert built == ["online"]
        assert len(finder.calls) == 2


# --- the re-stamped file must follow the version it now claims (audit D6) --------

class TestARecitedFileFollowsItsNewVersion:
    """`recite` stamps an old file with the current contract version, so it
    must also fill in what the versions since then added. Before v6.8 a 1.0
    file came back saying 1.5 without `status`, `failed_pages` or
    `author_names`; the schema did not require them, so nothing noticed."""

    @staticmethod
    def errors(document):
        jsonschema = pytest.importorskip("jsonschema")
        schema = json.loads((Path(__file__).resolve().parents[1] / "contract"
                             / "prajnya_document.schema.json").read_text(encoding="utf-8"))
        return [e.message for e in jsonschema.Draft202012Validator(schema).iter_errors(document)]

    def test_a_recited_1_0_file_is_a_valid_current_file(self, tmp_path):
        path = _write(tmp_path, _document())
        recite_document(path, "local", "cli", logger=LOG)
        document = _read(path)
        assert document["document_info"]["contract_version"] == CONTRACT_VERSION
        assert self.errors(document) == []
        assert (document["document_info"]["status"],
                document["document_info"]["failed_pages"]) == ("success", [])
        assert document["metadata_final"]["author_names"] == []

    def test_a_hand_corrected_record_is_kept_and_still_valid(self, tmp_path):
        path = _write(tmp_path, _document(validation_source="user_corrected"))
        recite_document(path, "local", "cli", logger=LOG)
        document = _read(path)
        assert document["metadata_final"]["validation_source"] == "user_corrected"
        assert self.errors(document) == []

    def test_a_lost_page_is_reported_after_reciting(self, tmp_path):
        old = _document()
        old["pages"][1]["metadata"]["processing_method"] = "glm_failed"
        path = _write(tmp_path, old)
        recite_document(path, "local", "cli", logger=LOG)
        info = _read(path)["document_info"]
        assert (info["status"], info["failed_pages"]) == ("partial", [2])
