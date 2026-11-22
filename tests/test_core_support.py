#!/usr/bin/env python3
"""
Tests for the ported core support layer: models, ledger, metadata merger,
anchor extractor, router client.

These are the modules the strategy pipeline stands on. Ledger and merger are
ported verbatim from V5 (no fitz, no model calls); models.py is modernized to
pydantic 2; anchor_extractor is rewritten onto the PdfBackend seam and its
tests pin the anchor *format*, which the OlmOCR models were trained against.
"""

import pytest

from src.processing.extractors.metadata_extractor import MetadataExtractor
from src.processing.ledger.csv_backend import CsvLedgerBackend
from src.processing.ledger.semantic_ledger import LedgerRecord
from src.processing.models import (
    ContentPage,
    DocumentProfile,
    ExtractedMetadata,
    ProcessingResult,
)
from src.processing.utils.anchor_extractor import extract_document_anchoring
from src.processing.utils.author_utils import (
    canonicalize_author,
    merge_author_lists,
    strip_affiliation_markers,
)
from src.processing.utils.metadata_merger import merge_metadata, normalize_doi
from src.processing.utils.model_swap import RouterClient
from tests.fakes import FakePage, word_row


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

class TestExtractedMetadata:
    def test_doi_url_prefix_is_stripped(self):
        m = ExtractedMetadata(doi="https://doi.org/10.1016/j.cellbi.2005.09.009")
        assert m.doi == "10.1016/j.cellbi.2005.09.009"

    def test_malformed_doi_is_rejected(self):
        with pytest.raises(Exception):
            ExtractedMetadata(doi="not-a-doi")

    def test_year_range_is_enforced(self):
        assert ExtractedMetadata(year=2006).year == 2006
        with pytest.raises(Exception):
            ExtractedMetadata(year=1200)
        with pytest.raises(Exception):
            ExtractedMetadata(year=3000)

    def test_extra_strategy_fields_are_allowed(self):
        m = ExtractedMetadata(title="T", doi_source="text_layer")
        assert m.doi_source == "text_layer"


class TestDocumentProfile:
    def _profile(self, **overrides) -> DocumentProfile:
        defaults = dict(pdf_path="paper.pdf", page_count=8, file_size=1024)
        defaults.update(overrides)
        return DocumentProfile(**defaults)

    def test_pdf_doc_is_excluded_from_serialization(self):
        profile = self._profile(pdf_doc=object())
        assert "pdf_doc" not in profile.model_dump()

    def test_progressive_enrichment_validates(self):
        profile = self._profile()
        result = ProcessingResult(metadata=ExtractedMetadata(title="T"),
                                  pages=[ContentPage(page_number=1, content="x")])
        profile.processing_result = result
        assert profile.processing_result.pages[0].page_number == 1
        with pytest.raises(Exception):
            profile.doi_confidence = 2.0  # out of range, validate_assignment fires


# ---------------------------------------------------------------------------
# ledger (round trip through the CSV backend)
# ---------------------------------------------------------------------------

class TestLedgerRoundTrip:
    def test_update_commit_reload_preserves_record(self, tmp_path):
        path = tmp_path / "ledger.csv"
        ledger = CsvLedgerBackend(file_path=str(path))
        record = LedgerRecord(
            document_id="abc123", pdf_filename="paper.pdf",
            doi="10.1016/j.cellbi.2005.09.009",
            title="An NMR method", authors=["A. Author", "B. Author"],
            journal="Cell Biology International", year=2006,
            validation_confidence=0.95, validation_source="zotero_doi_lookup",
            last_updated="2026-08-06T00:00:00", source_hash="abc123",
        )
        ledger.update_record(record)
        ledger.commit()

        reloaded = CsvLedgerBackend(file_path=str(path)).get_record("abc123")
        assert reloaded is not None
        assert reloaded.doi == record.doi
        assert reloaded.authors == ["A. Author", "B. Author"]  # list survives CSV
        assert reloaded.year == 2006

    def test_missing_record_is_none(self, tmp_path):
        ledger = CsvLedgerBackend(file_path=str(tmp_path / "ledger.csv"))
        assert ledger.get_record("nope") is None

    def test_get_all_records_returns_everything_added(self, tmp_path):
        ledger = CsvLedgerBackend(file_path=str(tmp_path / "ledger.csv"))
        for doc_id in ("a", "b"):
            ledger.update_record(LedgerRecord(
                document_id=doc_id, pdf_filename=f"{doc_id}.pdf",
                validation_confidence=0.9, validation_source="test",
                last_updated="2026-08-11T00:00:00", source_hash=doc_id,
            ))
        assert {r.document_id for r in ledger.get_all_records()} == {"a", "b"}

    def test_delete_record_removes_it_and_reports_success(self, tmp_path):
        path = tmp_path / "ledger.csv"
        ledger = CsvLedgerBackend(file_path=str(path))
        ledger.update_record(LedgerRecord(
            document_id="gone", pdf_filename="gone.pdf",
            validation_confidence=0.9, validation_source="test",
            last_updated="2026-08-11T00:00:00", source_hash="gone",
        ))
        assert ledger.delete_record("gone") is True
        assert ledger.get_record("gone") is None

        ledger.commit()
        reloaded = CsvLedgerBackend(file_path=str(path))
        assert reloaded.get_record("gone") is None

    def test_delete_record_missing_id_returns_false(self, tmp_path):
        ledger = CsvLedgerBackend(file_path=str(tmp_path / "ledger.csv"))
        assert ledger.delete_record("nope") is False

    def test_review_needed_fields_survives_commit_reload(self, tmp_path):
        # review_needed_fields was missing from commit()'s JSON-serialize
        # list, so csv.DictWriter fell back to str(list) — Python repr, not
        # JSON. An empty list's repr ("[]") happens to equal its JSON form,
        # which is exactly why every record in the 11 August gate run (0 of
        # 9 flagged) never exposed this: only a NON-empty list reveals it,
        # and _load()'s json.loads on "['doi', 'journal']" raises and is
        # swallowed, silently losing the field. Found live building S9's
        # Review Queue, the first real consumer of a flagged record's list.
        path = tmp_path / "ledger.csv"
        ledger = CsvLedgerBackend(file_path=str(path))
        ledger.update_record(LedgerRecord(
            document_id="flagged", pdf_filename="flagged.pdf",
            review_needed_fields=["doi", "journal"], requires_review=True,
            validation_confidence=0.4, validation_source="test",
            last_updated="2026-08-11T00:00:00", source_hash="flagged",
        ))
        ledger.commit()

        reloaded = CsvLedgerBackend(file_path=str(path)).get_record("flagged")
        assert reloaded.review_needed_fields == ["doi", "journal"]


# ---------------------------------------------------------------------------
# metadata merger
# ---------------------------------------------------------------------------

class TestMetadataMerger:
    @pytest.mark.parametrize("raw, expected", [
        ("10.1016/j.cellbi.2005.09.009", "10.1016/j.cellbi.2005.09.009"),
        ("https://doi.org/10.1016/j.cellbi.2005.09.009", "10.1016/j.cellbi.2005.09.009"),
        ("doi:10.1016/j.cellbi.2005.09.009", "10.1016/j.cellbi.2005.09.009"),
        ("", None),
        (None, None),
    ])
    def test_normalize_doi(self, raw, expected):
        assert normalize_doi(raw) == expected

    def test_zotero_doi_wins_over_vlm(self):
        merged = merge_metadata(
            {"doi": "10.9999/wrong", "title": "T", "authors": []},
            {"doi": "10.1016/j.cellbi.2005.09.009", "title": "T", "authors": []},
            title_page_text="T",
        )
        assert merged["doi"] == "10.1016/j.cellbi.2005.09.009"

    def test_year_parsed_from_messy_string(self):
        merged = merge_metadata(
            {"year": "Published Jan 2006", "title": "T", "authors": []},
            {}, title_page_text="T",
        )
        assert merged["year"] == 2006


# ---------------------------------------------------------------------------
# anchor extractor (format pinned: "[x,y]words", bottom-left origin)
# ---------------------------------------------------------------------------

class TestAnchorExtractor:
    def test_anchor_format_and_y_flip(self):
        # One line near the top of a 612x792 page: y0=72 -> anchor y = 792-72 = 720.
        page = FakePage(words=word_row(["Materials", "and", "methods"], y=72.0, x=100.0))
        anchor = extract_document_anchoring(page)
        assert anchor.startswith("[100,720]Materials and methods")

    def test_samples_at_most_eight_words_per_block(self):
        texts = [f"word{i}" for i in range(15)]
        page = FakePage(words=word_row(texts, y=100.0))
        anchor = extract_document_anchoring(page)
        assert "word7" in anchor and "word8" not in anchor

    def test_vertical_gap_starts_a_new_block(self):
        words = word_row(["First", "paragraph"], y=100.0) + \
                word_row(["Second", "paragraph"], y=300.0)
        page = FakePage(words=words)
        anchor = extract_document_anchoring(page)
        assert anchor.count("\n") == 2  # two anchor lines, one per block

    def test_empty_page_falls_back_to_dimensions(self):
        anchor = extract_document_anchoring(FakePage())
        assert anchor.startswith("Page dimensions:")

    def test_char_cap_is_respected(self):
        # 120 well-separated rows -> 120 blocks, far more than 500 chars' worth.
        long_page = FakePage(words=[w for i in range(120)
                                    for w in word_row([f"t{i}"] * 8, y=float(i * 50))])
        anchor = extract_document_anchoring(long_page, max_chars=500)
        assert len(anchor) <= 500


# ---------------------------------------------------------------------------
# router client (HTTP mocked — no server)
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class TestRouterClient:
    def test_loaded_models_reads_router_status_shape(self, monkeypatch):
        payload = {"data": [
            {"id": "glm-ocr", "status": {"value": "loaded"}},
            {"id": "olmocr-2-7b", "status": {"value": "loading"}},
            {"id": "plain-build", "status": "loaded"},
        ]}
        monkeypatch.setattr("src.processing.utils.model_swap.requests.get",
                            lambda url, timeout: _FakeResponse(payload))
        client = RouterClient("http://localhost:8080")
        assert client.loaded_models() == {"glm-ocr", "plain-build"}

    def test_unreachable_server_is_empty_set(self, monkeypatch):
        import requests as requests_module

        def boom(url, timeout):
            raise requests_module.ConnectionError("down")

        monkeypatch.setattr("src.processing.utils.model_swap.requests.get", boom)
        assert RouterClient("http://localhost:8080").loaded_models() == set()

    def test_set_resident_unloads_unwanted_then_loads_wanted(self, monkeypatch):
        client = RouterClient("http://localhost:8080")
        calls = []
        monkeypatch.setattr(client, "loaded_models", lambda: {"olmocr-2-7b"})
        monkeypatch.setattr(client, "unload", lambda m, timeout=60: calls.append(("unload", m)) or True)
        monkeypatch.setattr(client, "load", lambda m, timeout=180: calls.append(("load", m)) or True)
        assert client.set_resident(["glm-ocr", "deepseek-ocr-v2"]) is True
        assert ("unload", "olmocr-2-7b") in calls
        assert ("load", "glm-ocr") in calls and ("load", "deepseek-ocr-v2") in calls


class TestWrappedDoiExtraction:
    """
    DOIs broken across a line break by a narrow margin column.

    Nature Reviews sets the DOI down the bottom-left margin, so the text
    layer holds 'https://doi.org/10.1038/\ns41578-020-00251-2'. Every DOI
    pattern needs a character after the '/', so the wrapped form matched
    nothing and the paper looked DOI-less — costing a VLM call to recover
    something already present in the text layer.
    """

    def test_url_doi_wrapped_after_slash_is_rejoined(self):
        text = "REVIEWS\nhttps://doi.org/10.1038/\ns41578-020-00251-2\n264 | March 2021"
        assert MetadataExtractor().extract_doi_from_text(text) == (
            "10.1038/s41578-020-00251-2", 0.95)

    def test_crlf_and_indentation_are_absorbed(self):
        text = "https://doi.org/10.1007/\r\n   978-3-030-80614-9"
        doi, _ = MetadataExtractor().extract_doi_from_text(text)
        assert doi == "10.1007/978-3-030-80614-9"

    def test_labeled_doi_wrapped_after_slash_keeps_its_confidence(self):
        doi, conf = MetadataExtractor().extract_doi_from_text("doi: 10.1016/\nj.tig.2003.11.004")
        assert (doi, conf) == ("10.1016/j.tig.2003.11.004", 0.98)

    def test_unwrapped_dois_are_untouched(self):
        """The repair must be a no-op on every DOI that already parsed."""
        for text, expected in [
            ("doi: 10.1038/s41578-020-00251-2", ("10.1038/s41578-020-00251-2", 0.98)),
            ("https://doi.org/10.1016/j.matt.2024.01.023", ("10.1016/j.matt.2024.01.023", 0.95)),
            ("see 10.1021/acs.jpcb.9b07555 for detail", ("10.1021/acs.jpcb.9b07555", 0.85)),
        ]:
            assert MetadataExtractor().extract_doi_from_text(text) == expected

    def test_a_hyphen_wrap_is_not_joined(self):
        """
        We refuse the ambiguous case on purpose. A real hyphen and a
        hyphenation hyphen are indistinguishable in a text layer, so joining
        would risk INVENTING a DOI. A missed DOI is recoverable downstream;
        a fabricated one silently corrupts the record.
        """
        doi, _ = MetadataExtractor().extract_doi_from_text("10.1234/abc-\ndef")
        assert doi == "10.1234/abc"  # truncated, never "10.1234/abc-def"

    def test_a_bare_line_break_creates_no_doi(self):
        assert MetadataExtractor().extract_doi_from_text(
            "page 10.\n2020 results") == (None, 0.0)


class TestCanonicalDocumentJson:
    """
    run_prajnya.py must emit the shape the parity harness reads.

    The harness (tests/parity.py) speaks V5's serialized document format —
    top-level 'pages', plus 'document_info' and 'metadata_final'. V6's
    DocumentProfile nests all of that under 'processing_result'. The per-page
    and per-figure records already match exactly, so only this envelope stands
    between a V6 run and the gate. It was missed because the harness had never
    been pointed at V6 output until session 7; when it was, every check read
    None and looked like a total pipeline failure.
    """

    @staticmethod
    def _profile(tmp_path, errors=None):
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        profile = DocumentProfile(pdf_path=pdf, page_count=2, file_size=9)
        profile.processing_result = ProcessingResult(
            metadata=ExtractedMetadata(title="A paper", doi="10.1016/j.test.2026.01.001"),
            pages=[ContentPage(page_number=1, content="one",
                               metadata={"figure_count": 2}),
                   ContentPage(page_number=2, content="two",
                               metadata={"figure_count": 1})],
            errors=errors or [],
        )
        return profile

    def test_envelope_carries_the_keys_the_harness_reads(self, tmp_path):
        from run_prajnya import build_document_json

        doc = build_document_json(self._profile(tmp_path), elapsed=12.5)

        assert doc["document_info"]["filename"] == "paper.pdf"
        assert doc["document_info"]["processing_time"] == 12.5
        assert doc["document_info"]["success"] is True
        assert doc["metadata_final"]["doi"] == "10.1016/j.test.2026.01.001"
        # pages must be TOP-LEVEL, not nested under processing_result
        assert [p["page_number"] for p in doc["pages"]] == [1, 2]
        assert doc["summary"]["errors"] == []

    def test_figure_count_is_summed_across_pages(self, tmp_path):
        from run_prajnya import build_document_json

        doc = build_document_json(self._profile(tmp_path), elapsed=1.0)
        assert doc["summary"]["figure_count"] == 3

    def test_errors_flip_success_false(self, tmp_path):
        from run_prajnya import build_document_json

        doc = build_document_json(self._profile(tmp_path, errors=["boom"]), elapsed=1.0)
        assert doc["document_info"]["success"] is False
        assert doc["summary"]["errors"] == ["boom"]

    def test_timings_are_recorded_when_supplied(self, tmp_path):
        """The harness prefers document_info['timings'] over anything it can reconstruct."""
        from run_prajnya import build_document_json

        doc = build_document_json(self._profile(tmp_path), elapsed=9.0,
                                  timings={"title": 3.0, "reads": 6.0})
        assert doc["document_info"]["timings"] == {"title": 3.0, "reads": 6.0}

    def test_nothing_is_lost_by_the_rewrap(self, tmp_path):
        """The full profile stays available under 'profile'."""
        from run_prajnya import build_document_json

        doc = build_document_json(self._profile(tmp_path), elapsed=1.0)
        assert doc["profile"]["page_count"] == 2

    def test_the_harness_can_read_the_result(self, tmp_path):
        """End to end: the envelope survives the harness's own accessors."""
        from run_prajnya import build_document_json
        from tests.parity import all_figures, get_pages

        doc = build_document_json(self._profile(tmp_path), elapsed=1.0)
        assert len(get_pages(doc)) == 2
        assert all_figures(doc) == []  # no figure records, only counts


class TestAffiliationMarkers:
    """
    Journals hang footnote markers off author names; they are not names.

    Found on the real corpus: the VLM read "Jordi Bella*1" from the page, and
    because canonicalize_author keys on the string, that gave 'bella*1, J'
    while Zotero's canonical "Jordi Bella" gave 'bella, J'. Two keys for one
    person, so the dedupe could not pair them and the dirty VLM form won on
    pool order - then got stamped validation_source=zotero_api, claiming a
    canonical provenance the value did not have.
    """

    @pytest.mark.parametrize("raw, expected", [
        ("Jordi Bella*1", "Jordi Bella"),            # the observed case
        ("Michael J. Mienaltowski1,2", "Michael J. Mienaltowski"),
        ("A. Author†", "A. Author"),
        ("B. Author‡,§", "B. Author"),
        ("C. Author 1", "C. Author"),
        ("D. Author¹", "D. Author"),
    ])
    def test_markers_are_stripped(self, raw, expected):
        assert strip_affiliation_markers(raw) == expected

    @pytest.mark.parametrize("name", [
        "Gary D. Fullerton",     # clean names are untouched
        "Jordi Bella",
        "Smith III",             # roman numerals are LETTERS, not markers
        "O'Brien",
        "Jean-Luc Picard",
    ])
    def test_clean_names_are_untouched(self, name):
        assert strip_affiliation_markers(name) == name

    def test_a_name_is_never_reduced_to_nothing(self):
        """Stripping that would empty the string returns the original."""
        assert strip_affiliation_markers("12345") == "12345"

    def test_only_trailing_markers_go(self):
        """A digit inside the name is not a footnote pointer."""
        assert strip_affiliation_markers("Author 3rd") == "Author 3rd"

    def test_the_two_forms_now_share_a_canonical_key(self):
        assert canonicalize_author("Jordi Bella*1") == canonicalize_author("Jordi Bella")

    def test_zotero_clean_name_survives_the_merge(self):
        """The whole point: the record must not carry the marker."""
        merged, suspicious = merge_author_lists(
            ["Jordi Bella*1"], ["Jordi Bella"],
            "Collagen structure\nJordi Bella*1\n1 Faculty of Life Sciences")
        assert merged == ["Jordi Bella"]
        assert suspicious == []

    def test_one_author_is_not_split_into_two(self):
        """Different keys for one person used to duplicate them."""
        merged, _ = merge_author_lists(
            ["Michael J. Mienaltowski1,2", "Nicole L. Gonzales3"],
            ["Michael J. Mienaltowski", "Nicole L. Gonzales"],
            "Michael J. Mienaltowski1,2 and Nicole L. Gonzales3")
        assert merged == ["Michael J. Mienaltowski", "Nicole L. Gonzales"]

    def test_benchmark_authors_are_unchanged_and_in_order(self):
        """The parity gate: 6 authors, published order, must not move."""
        nmr = ["Gary D. Fullerton", "Elena Nes", "Maxwell Amurao",
               "Andres Rahal", "Lada Krasnosselskaia", "Ivan Cameron"]
        merged, _ = merge_author_lists(nmr, nmr, " ".join(nmr))
        assert merged == nmr

    def test_markers_are_stripped_without_a_title_page_referee(self):
        """The no-referee early return must clean too, not just the main path."""
        merged, _ = merge_author_lists(["Jordi Bella*1"], ["Jordi Bella"], "")
        assert merged == ["Jordi Bella"]
