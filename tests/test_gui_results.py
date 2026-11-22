#!/usr/bin/env python3
"""
Results viewer — the data layer, tested against a synthetic run tree.

For biology researchers: `mount()` itself needs a live NiceGUI page context
(same reason as `library.py`'s tests) and was verified by hand this session
against the real 11 August gate run — 9 documents indexed correctly with no
false review flags, a 345-page book loaded in under 20ms, figure/table
manifests resolved and every crop confirmed present on disk, and the
`/run_outputs` static route served real images with HTTP 200. What's
covered here is the part that can run in CI without that live tree: the
pure functions that turn files on disk into what the page renders.

The one thing worth protecting on purpose: `safe_doc_id` is the SAME
function the pipeline uses when it WRITES the figures/tables directory
name (`figure_extractor.py`), imported here rather than re-derived — a
document whose PDF filename has spaces or punctuation is exactly the case
that would silently break a hand-rolled sanitiser and this project's own
corpus, so a document with a "hard" filename is tested explicitly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.gui.pages import results
from src.processing.utils.figure_extractor import safe_doc_id


@pytest.fixture
def output_root(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "DEFAULT_OUTPUT_ROOT", tmp_path)
    return tmp_path


def _write_document(output_root: Path, filename: str, *, title: str = "A Paper",
                    document_type: str = "research_article",
                    review_status: str = "reviewed") -> Path:
    documents_dir = output_root / "documents"
    documents_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).stem
    doc = {
        "document_info": {"filename": filename, "document_type": document_type},
        "metadata_final": {"title": title, "authors": ["A. Author"], "doi": None,
                           "journal": "Journal", "year": 2020,
                           "validation_source": "zotero_api", "validation_confidence": 0.9},
        "pages": [{"page_number": 1, "content": "hello", "quality_score": 0.8}],
        "summary": {"errors": [], "page_count": 1, "figure_count": 0},
        "review_status": review_status,
        "review_needed_fields": [] if review_status == "reviewed" else ["doi"],
    }
    path = documents_dir / f"{stem}.v6.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _write_manifest(output_root: Path, kind: str, stem: str, list_key: str,
                    items: list) -> None:
    directory = output_root / kind / safe_doc_id(stem)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"document": stem, f"{list_key[:-1]}_count": len(items), list_key: items}
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class TestIndexDocuments:
    def test_no_documents_directory_is_an_empty_list(self, output_root):
        assert results._index_documents() == []

    def test_lists_every_document_with_title_and_type(self, output_root):
        _write_document(output_root, "paper one.pdf", title="Paper One")
        _write_document(output_root, "paper_two.pdf", title="Paper Two",
                        document_type="book")

        index = results._index_documents()
        labels = {entry["label"] for entry in index}
        assert any("Paper One" in label and "research_article" in label for label in labels)
        assert any("Paper Two" in label and "book" in label for label in labels)

    def test_a_provisional_document_is_flagged_in_its_label(self, output_root):
        _write_document(output_root, "flagged.pdf", review_status="provisional")
        index = results._index_documents()
        assert "⚠" in index[0]["label"]

    def test_an_unreadable_document_does_not_break_the_whole_index(self, output_root):
        documents_dir = output_root / "documents"
        documents_dir.mkdir(parents=True)
        (documents_dir / "corrupt.v6.json").write_text("{not valid json", encoding="utf-8")
        _write_document(output_root, "good.pdf")

        index = results._index_documents()
        assert len(index) == 2
        assert any("unreadable" in entry["label"] for entry in index)


class TestFiguresAndTablesResolveTheSameWayThePipelineWritesThem:
    """The directory a figure lives under is a SANITISED version of the PDF
    stem, not the stem itself — this is the one thing this page has to get
    exactly right or every gallery for a document with spaces or
    punctuation in its filename comes back empty."""

    def test_a_filename_with_spaces_and_punctuation_still_resolves(self, output_root):
        stem = "Collagen structure- new tricks from a very old dog"
        figure = {"figure_id": "p2_fig1", "image_path": f"figures/{safe_doc_id(stem)}/p2_fig1.png",
                  "page": 2, "printed_label": "Figure 1", "label_key": "figure 1",
                  "caption": "A figure."}
        _write_manifest(output_root, "figures", stem, "figures", [figure])

        found = results._figures_for(stem)
        assert found == [figure]

    def test_no_manifest_is_an_empty_list_not_an_error(self, output_root):
        assert results._figures_for("nothing processed yet") == []
        assert results._tables_for("nothing processed yet") == []

    def test_tables_use_the_tables_manifest_not_the_figures_one(self, output_root):
        stem = "paper"
        table = {"table_id": "p3_tab1", "image_path": f"tables/{safe_doc_id(stem)}/p3_tab1.png",
                 "page": 3, "caption": "Table 1.", "grid": "<table></table>"}
        _write_manifest(output_root, "tables", stem, "tables", [table])

        assert results._tables_for(stem) == [table]
        assert results._figures_for(stem) == []


class TestGroupBy:
    def test_groups_preserve_insertion_order_within_a_group(self):
        items = [{"label_key": "figure 1", "n": 1}, {"label_key": "figure 2", "n": 2},
                 {"label_key": "figure 1", "n": 3}]
        groups = results._group_by(items, "label_key")
        assert [item["n"] for item in groups["figure 1"]] == [1, 3]
        assert [item["n"] for item in groups["figure 2"]] == [2]

    def test_a_missing_key_groups_under_the_empty_string_not_none(self):
        groups = results._group_by([{"other": "x"}], "label_key")
        assert list(groups.keys()) == [""]


class TestFigureGroupOrder:
    def test_groups_follow_the_printed_numbers_not_detection_order(self):
        keys = ["figure 6", "figure 10", "", "figure 5", "figure 2"]
        assert sorted(keys, key=results._label_order) == \
            ["figure 2", "figure 5", "figure 6", "figure 10", ""]
