"""
The `citations` block: the level a run used, and how a document is cited.

For biology researchers: every output file now says how its bibliographic
record was established (`level`), which step supplied it, and the short form
to cite it by ("Bella, 2016"). These tests pin three things that would be
expensive to get wrong:

  1. the level is decided by one rule, and a misspelt level stops the run
     (falling back silently could send data out that was meant to stay in);
  2. the label follows the metadata, including after a hand correction;
  3. a document with nothing to cite by is cited by its FILE NAME, and the
     file name never leaks into the title.
"""
import json
from pathlib import Path

import pytest

from src.processing.citations import (CITATION_LEVELS, build_citations_block,
                                      metadata_source_of, resolve_level,
                                      source_label)
from src.processing.citations.block import (METADATA_SOURCES,
                                            REFERENCE_LIST_STATUSES,
                                            reference_data_of)
from src.processing.citations.level import LEVEL_SET_BY
from src.processing.document_json import build_document_json
from src.processing.ledger.semantic_ledger import LedgerRecord
from src.processing.models import (DocumentProfile, DocumentType,
                                   ExtractedMetadata, ProcessingResult)
from src.processing.review import regenerate_document_json

REPO_ROOT = Path(__file__).resolve().parents[1]


# --- 1. which level, and who chose it ------------------------------------------

class TestResolveLevel:
    def test_the_config_names_the_level(self):
        assert resolve_level({"citations": {"level": "online"}}) == ("online", "config")

    def test_the_command_line_overrides_the_config(self):
        config = {"citations": {"level": "online"}}
        assert resolve_level(config, override="local") == ("local", "cli")

    def test_nothing_said_means_local(self):
        """Nothing leaves the machine unless chosen (researcher, session 31)."""
        assert resolve_level({}) == ("local", "default")
        assert resolve_level(None) == ("local", "default")

    def test_case_and_spaces_are_forgiven(self):
        assert resolve_level({"citations": {"level": " Zotero "}})[0] == "zotero"

    @pytest.mark.parametrize("zotero, expected", [
        ({"enabled": True, "offline_mode": False}, "zotero"),
        ({"enabled": True}, "zotero"),
        ({"enabled": True, "offline_mode": True}, "local"),
        ({"enabled": False}, "local"),
        ({}, "local"),
    ])
    def test_an_older_config_keeps_doing_what_it_did(self, zotero, expected):
        """A config with a zotero block and no level behaves as before."""
        assert resolve_level({"zotero": zotero}) == (expected, "legacy_config")

    def test_a_named_level_beats_the_older_zotero_block(self):
        config = {"citations": {"level": "local"}, "zotero": {"enabled": True}}
        assert resolve_level(config) == ("local", "config")

    @pytest.mark.parametrize("config, override", [
        ({"citations": {"level": "onlin"}}, None),
        ({}, "crossref"),
    ])
    def test_a_level_that_does_not_exist_stops_the_run(self, config, override):
        with pytest.raises(ValueError, match="local, zotero, online"):
            resolve_level(config, override)

    def test_the_live_config_resolves(self):
        """Verified against the real config tree, not a fixture."""
        from src.processing.config.loader import ConfigLoader
        level, set_by = resolve_level(ConfigLoader(REPO_ROOT / "config").strategy())
        assert level in CITATION_LEVELS and set_by in LEVEL_SET_BY


# --- 2. the label ----------------------------------------------------------------

class TestSourceLabel:
    @pytest.mark.parametrize("authors, year, expected", [
        (["Jordi Bella"], 2016, "Bella, 2016"),
        (["Johanna Myllyharju", "Kari I. Kivirikko"], 2004, "Myllyharju and Kivirikko, 2004"),
        (["Gary D. Fullerton", "Elena Nes", "Maxwell Amurao"], 2006, "Fullerton et al., 2006"),
        (["Bella, Jordi"], 2016, "Bella, 2016"),
        (["Annette van der Flier"], 2001, "van der Flier, 2001"),
        (["Marc André Meyers"], 2020, "Meyers, 2020"),
        (["Jordi Bella"], None, "Bella, n.d."),
    ])
    def test_authors(self, authors, year, expected):
        assert source_label({"authors": authors, "year": year}, "x.pdf") == (expected, False)

    def test_a_book_is_cited_by_its_editor(self):
        metadata = {"authors": [], "editors": ["Jaroslava Halper"], "year": 2021}
        assert source_label(metadata, "x.pdf") == ("Halper (ed.), 2021", False)

    def test_two_editors(self):
        metadata = {"editors": ["A. One", "B. Two"], "year": 2021}
        assert source_label(metadata, "x.pdf")[0] == "One and Two (eds.), 2021"

    def test_no_names_falls_to_the_title(self):
        assert source_label({"title": "A short title", "year": 1960}, "x.pdf") == (
            "A short title, 1960", False)

    def test_a_long_title_is_cut_at_a_word(self):
        title = "Saturated solutions for the control of humidity in biological research " * 3
        label, by_file_name = source_label({"title": title}, "x.pdf")
        assert label.endswith(" ..., n.d.") and len(label) < 100 and not by_file_name

    @pytest.mark.parametrize("metadata", [
        {}, {"title": None, "authors": [], "editors": []}, {"title": "   ", "authors": [""]},
        {"year": 2019},
    ])
    def test_nothing_usable_is_cited_by_the_file_name(self, metadata):
        assert source_label(metadata, "Silberbush-2019-Fig2a,3.pdf") == (
            "Silberbush-2019-Fig2a,3.pdf", True)


class TestMetadataSource:
    @pytest.mark.parametrize("validation_source, expected", [
        ("zotero_api", "zotero"), ("zotero_cache", "zotero"), ("zotero_fuzzy_match", "zotero"),
        ("crossref", "crossref"), ("openalex", "openalex"),
        ("user_corrected", "user"),
        ("vlm_extraction", "extracted"), ("front_matter_text_layer", "extracted"),
        ("unknown", "extracted"), (None, "extracted"),
    ])
    def test_reduced_to_the_closed_list(self, validation_source, expected):
        assert metadata_source_of({"validation_source": validation_source}) == expected


# --- 3. the block -----------------------------------------------------------------

class TestBlock:
    def test_a_normal_document(self):
        block = build_citations_block(
            {"authors": ["Jordi Bella"], "year": 2016, "validation_source": "zotero_api"},
            "bella.pdf", "zotero", "config")
        assert block == {
            "level": "zotero", "level_set_by": "config", "metadata_source": "zotero",
            "source_label": "Bella, 2016", "cited_by_file_name": False,
            "reference_list": {"status": "not_requested"}, "references": [], "markers": [],
        }

    def test_the_file_name_case_says_so_three_ways(self):
        block = build_citations_block({"validation_source": "zotero_api"}, "scan0042.pdf",
                                      "local", "default")
        assert block["source_label"] == "scan0042.pdf"
        assert block["cited_by_file_name"] is True
        assert block["metadata_source"] == "file_name"

    def test_written_outside_a_run_it_carries_the_default_level(self):
        block = build_citations_block({}, "x.pdf")
        assert (block["level"], block["level_set_by"]) == ("local", "default")

    def test_unusable_metadata_does_not_raise(self):
        assert build_citations_block(None, "x.pdf")["cited_by_file_name"] is True

    def test_every_value_is_one_the_vocabularies_list(self):
        block = build_citations_block({}, "x.pdf", "online", "cli")
        assert block["level"] in CITATION_LEVELS
        assert block["level_set_by"] in LEVEL_SET_BY
        assert block["metadata_source"] in METADATA_SOURCES
        assert block["reference_list"]["status"] in REFERENCE_LIST_STATUSES


# --- 4. wired into the writer and the review step ------------------------------------

def _profile(tmp_path: Path, name: str = "paper.pdf", **metadata) -> DocumentProfile:
    pdf = tmp_path / name
    pdf.write_bytes(b"%PDF-1.4\n")
    profile = DocumentProfile(pdf_path=pdf, page_count=1, file_size=9,
                              document_type=DocumentType.RESEARCH_ARTICLE)
    profile.processing_result = ProcessingResult(metadata=ExtractedMetadata(**metadata))
    return profile


class TestTheWriterCarriesTheBlock:
    def test_the_stamped_level_reaches_the_file(self, tmp_path):
        profile = _profile(tmp_path, title="T", authors=["Jordi Bella"], year=2016,
                           validation_source="zotero_api")
        profile.processing_result.citation_level = "zotero"
        profile.processing_result.citation_level_set_by = "config"
        citations = build_document_json(profile, elapsed=1.0)["citations"]
        assert (citations["level"], citations["level_set_by"]) == ("zotero", "config")
        assert citations["source_label"] == "Bella, 2016"

    def test_the_file_name_never_reaches_the_title(self, tmp_path):
        """Researcher's decision, session 32: `title` stays empty."""
        doc = build_document_json(_profile(tmp_path, name="JRuberti.pdf"), elapsed=1.0)
        assert doc["citations"]["source_label"] == "JRuberti.pdf"
        assert doc["citations"]["cited_by_file_name"] is True
        assert doc["metadata_final"]["title"] is None


class TestTheRunnerStampsTheLevel:
    """The level is read from the config once, in the runner, for every strategy."""

    def _stamp(self, tmp_path, config_text, citations=None):
        from src.processing.run_request import RunRequest
        from src.processing.runner import LOG, _stamp_citation_level

        (tmp_path / "research_article_strategy.yaml").write_text(config_text, encoding="utf-8")
        profile = _profile(tmp_path)
        request = RunRequest(config_root=tmp_path)
        if citations:
            request.citations = citations
        _stamp_citation_level(profile, request, LOG)
        return profile.processing_result

    def test_from_the_config(self, tmp_path):
        result = self._stamp(tmp_path, "citations:\n  level: online\n")
        assert (result.citation_level, result.citation_level_set_by) == ("online", "config")

    def test_from_an_older_config(self, tmp_path):
        result = self._stamp(tmp_path, "zotero:\n  enabled: true\n")
        assert (result.citation_level, result.citation_level_set_by) == ("zotero", "legacy_config")

    def test_a_bad_level_does_not_lose_the_document(self, tmp_path):
        """The extracted content is already paid for; it must still be written."""
        result = self._stamp(tmp_path, "citations:\n  level: everything\n")
        assert result.citation_level is None
        doc = build_document_json(_profile(tmp_path), elapsed=1.0)
        assert doc["citations"]["level"] == "local"


class TestACorrectionMovesTheLabel:
    """Derived, never stored: a hand-corrected author must change the label."""

    def _written(self, tmp_path) -> Path:
        profile = _profile(tmp_path, title="T", authors=["Wrong Name"], year=2016,
                           validation_source="zotero_api")
        profile.processing_result.citation_level = "zotero"
        profile.processing_result.citation_level_set_by = "config"
        path = tmp_path / "paper.v6.json"
        path.write_text(json.dumps(build_document_json(profile, elapsed=1.0)), encoding="utf-8")
        return path

    def _record(self, **changes) -> LedgerRecord:
        fields = dict(document_id="h", pdf_filename="paper.pdf", title="T",
                      authors=["Jordi Bella"], journal="J", year=2016, doi="10.1042/BJ20151169",
                      validation_confidence=0.95, validation_source="zotero_api",
                      requires_review=False, review_status="reviewed",
                      last_updated="2026-11-14", source_hash="h")
        fields.update(changes)
        return LedgerRecord(**fields)

    def test_label_and_source_follow_the_correction(self, tmp_path):
        path = self._written(tmp_path)
        assert regenerate_document_json(path, self._record(last_reviewed_by="gui"))
        citations = json.loads(path.read_text(encoding="utf-8"))["citations"]
        assert citations["source_label"] == "Bella, 2016"
        assert citations["metadata_source"] == "user"
        assert (citations["level"], citations["level_set_by"]) == ("zotero", "config")

    def test_a_reference_list_survives_a_correction(self, tmp_path):
        """A metadata correction must not delete what level `online` fetched."""
        path = self._written(tmp_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["citations"]["references"] = [{"position": 1}]
        payload["citations"]["reference_list"] = {"status": "found", "count": 1}
        path.write_text(json.dumps(payload), encoding="utf-8")

        regenerate_document_json(path, self._record())
        citations = json.loads(path.read_text(encoding="utf-8"))["citations"]
        assert citations["references"] == [{"position": 1}]
        assert citations["reference_list"]["status"] == "found"

    def test_an_older_file_is_not_given_a_block(self, tmp_path):
        """Correcting a contract 1.0 file must not change its format."""
        path = self._written(tmp_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        del payload["citations"]
        path.write_text(json.dumps(payload), encoding="utf-8")

        regenerate_document_json(path, self._record())
        assert "citations" not in json.loads(path.read_text(encoding="utf-8"))


# --- 5. the reference list (level online) ------------------------------------------------

FOUND = {"reference_list": {"status": "found", "source": "crossref", "count": 1},
         "references": [{"position": 1, "number": 1}],
         "report": {"internal": True}}


class TestReferenceData:
    def test_the_list_is_copied_into_the_block(self):
        block = build_citations_block({}, "x.pdf", "online", "cli", reference_data=FOUND)
        assert block["reference_list"] == FOUND["reference_list"]
        assert block["references"] == FOUND["references"]
        assert block["markers"] == []

    def test_the_report_never_leaks(self):
        block = build_citations_block({}, "x.pdf", "online", "cli", reference_data=FOUND)
        assert "report" not in block and "internal" not in json.dumps(block)

    def test_the_block_holds_copies(self):
        block = build_citations_block({}, "x.pdf", "online", "cli", reference_data=FOUND)
        block["references"].append({"position": 2})
        block["reference_list"]["count"] = 99
        assert len(FOUND["references"]) == 1 and FOUND["reference_list"]["count"] == 1

    @pytest.mark.parametrize("data", [None, {}, {"references": [1]}, {"reference_list": "x"}])
    def test_without_usable_data_the_block_is_as_before(self, data):
        block = build_citations_block({}, "x.pdf", "local", "cli", reference_data=data)
        assert block["reference_list"] == {"status": "not_requested"}
        assert block["references"] == []

    def test_reference_data_of_reads_a_block_back(self):
        block = build_citations_block({}, "x.pdf", "online", "cli", reference_data=FOUND)
        assert reference_data_of(block) == {"reference_list": FOUND["reference_list"],
                                            "references": FOUND["references"],
                                            "markers": []}

    def test_reference_data_of_is_none_when_not_requested(self):
        assert reference_data_of(build_citations_block({}, "x.pdf")) is None
        assert reference_data_of(None) is None
        assert reference_data_of({"reference_list": "bad"}) is None


class TestTheResultCarriesTheList:
    def test_the_list_reaches_the_citations_block(self, tmp_path):
        profile = _profile(tmp_path, title="T", authors=["Jordi Bella"], year=2016)
        profile.processing_result.citation_level = "online"
        profile.processing_result.citation_references = FOUND
        doc = build_document_json(profile, elapsed=1.0)
        assert doc["citations"]["references"] == FOUND["references"]

    def test_the_list_is_not_dumped_twice(self, tmp_path):
        """`profile` holds the whole result; the list must appear once, in `citations`."""
        profile = _profile(tmp_path)
        profile.processing_result.citation_references = FOUND
        assert "citation_references" not in profile.processing_result.model_dump()
        assert "citation_references" not in profile.model_dump(mode="json")["processing_result"]
        doc = build_document_json(profile, elapsed=1.0)
        assert "citation_references" not in doc["profile"]["processing_result"]


class TestTheRunnerAttachesTheList:
    class Finder:
        def __init__(self, raises=False):
            self.raises, self.calls = raises, []

        def find(self, doi, pdf_path):
            self.calls.append((doi, pdf_path))
            if self.raises:
                raise RuntimeError("network on fire")
            return FOUND

    def _attach(self, tmp_path, monkeypatch, level="online", document_type=None, finder=None):
        from src.processing.runner import LOG, _attach_reference_list

        finder = finder or self.Finder()
        monkeypatch.setattr("src.processing.runner.reference_finder_for",
                            lambda lvl, config, logger=None: finder if lvl == "online" else None)
        profile = _profile(tmp_path, doi="10.1042/BJ20151169")
        if document_type:
            profile.document_type = document_type
        _attach_reference_list(profile, {}, level, LOG)
        return profile.processing_result, finder, profile

    def test_an_article_at_online_gets_the_list(self, tmp_path, monkeypatch):
        result, finder, profile = self._attach(tmp_path, monkeypatch)
        assert result.citation_references == FOUND
        assert finder.calls == [("10.1042/BJ20151169", profile.pdf_path)]

    def test_a_book_is_skipped(self, tmp_path, monkeypatch):
        result, finder, _ = self._attach(tmp_path, monkeypatch, document_type=DocumentType.BOOK)
        assert result.citation_references is None and finder.calls == []

    @pytest.mark.parametrize("level", ["local", "zotero"])
    def test_other_levels_are_skipped(self, tmp_path, monkeypatch, level):
        result, finder, _ = self._attach(tmp_path, monkeypatch, level=level)
        assert result.citation_references is None and finder.calls == []

    def test_a_finder_that_raises_costs_nothing(self, tmp_path, monkeypatch, caplog):
        with caplog.at_level("ERROR"):
            result, _, _ = self._attach(tmp_path, monkeypatch, finder=self.Finder(raises=True))
        assert result.citation_references is None
        assert "network on fire" in caplog.text
