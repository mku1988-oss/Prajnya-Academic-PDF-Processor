"""
T39, batch by model: the title phase, the saved title pages, the body phase.

For biology researchers: a folder run reads every title page of a chunk with
the title model, then every body with the body model, so the GPU swaps models
twice per chunk instead of twice per paper. These tests pin the ORDER of the
work, the files that carry a title page between the phases, and what happens
after a crash. The gate (eval/batch_by_model/GATE.md) checks the real outputs
on the GPU; this file checks the plumbing with fakes, no model involved.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.processing import runner
from src.processing.batch import PreflightEntry
from src.processing.models import ContentPage, DocumentType
from src.processing.run_request import RunRequest
from src.processing.title_page_store import TITLE_PAGE_FOLDER, TitlePageStore


# --- fakes ------------------------------------------------------------------

class FakeArticle:
    """Stands in for ResearchArticleStrategy; writes what it does to `calls`."""

    def __init__(self, calls, key="settings-1", crash_on=None, fail_title=()):
        self.calls, self.key = calls, key
        self.crash_on, self.fail_title = crash_on, set(fail_title)

    def title_settings_key(self):
        return self.key

    def read_title_page(self, profile):
        name = profile.pdf_path.name
        self.calls.append(("title", name))
        if name in self.fail_title:
            raise RuntimeError("the title model fell over")
        return ContentPage(page_number=1, content=f"title of {name}",
                           metadata={"shot1_time": 1.5, "authors": ["A", "B"]})

    def process(self, profile, test_pages_override=None, title_page=None):
        name = profile.pdf_path.name
        if name == self.crash_on:
            raise KeyboardInterrupt          # a crash, not a document error
        if title_page is None:
            self.fail_title.discard(name)    # the second try succeeds
            title_page = self.read_title_page(profile)
        self.calls.append(("body", name))
        profile.processing_result = SimpleNamespace(
            metadata=None, pages=[title_page], errors=[])
        return profile


class FakeBook:
    """A book: no title phase (no `title_settings_key`)."""

    def __init__(self, calls):
        self.calls = calls

    def process(self, profile, test_pages_override=None):
        self.calls.append(("book", profile.pdf_path.name))
        profile.processing_result = SimpleNamespace(
            metadata=None, pages=[ContentPage(page_number=1, content="book")], errors=[])
        return profile


def fake_profile(path, backend=None, front_matter_pages=None):
    kind = DocumentType.BOOK if path.stem.startswith("book") else DocumentType.RESEARCH_ARTICLE
    return SimpleNamespace(pdf_path=path, document_type=kind, page_count=3, doi=None,
                           doi_confidence=0.0, isbn=None, issn=None,
                           identifier_pages=None, classification_confidence=1.0,
                           title_page_index=0, processing_result=None)


@pytest.fixture
def batch(tmp_path, monkeypatch):
    """A runner wired to fakes; returns (run, calls, written)."""
    calls, written = [], {}

    def install(**strategy_options):
        def build(document_type, *args, **kwargs):
            if document_type is DocumentType.BOOK:
                return FakeBook(calls)
            return FakeArticle(calls, **strategy_options)

        def write(profile, out_path, elapsed, provenance=None):
            written[profile.pdf_path.name] = profile.processing_result.pages[0].content

        monkeypatch.setattr(runner, "build_strategy", build)
        monkeypatch.setattr(runner, "build_document_profile", fake_profile)
        monkeypatch.setattr(runner, "get_backend", lambda name=None: object())
        monkeypatch.setattr(runner, "write_result", write)
        monkeypatch.setattr(runner, "run_provenance", lambda root, overrides=None: {})
        monkeypatch.setattr(runner, "_stamp_citation_level", lambda *a: None)
        monkeypatch.setattr(runner, "_run_citation_level", lambda *a: None)
        monkeypatch.setattr(runner, "_result_for",
                            lambda outcome, ledger, log: {"error": outcome.error})

    def run(names, chunk_size, **strategy_options):
        install(**strategy_options)
        entries = []
        for name in names:
            path = tmp_path / name
            if not path.exists():
                path.write_bytes(f"%PDF {name}".encode())
            entries.append(PreflightEntry(path=path, document_type=None,
                                          confidence=1.0, page_count=3))
        request = RunRequest(paths=[e.path for e in entries], out_root=tmp_path / "out",
                             force=True, chunk_size=chunk_size)
        return runner.process_batch(entries, request, ledger=None)

    return run, calls, written


# --- the order of the work --------------------------------------------------

class TestTheOrder:
    def test_a_chunk_reads_every_title_page_before_any_body(self, batch):
        run, calls, _ = batch
        run(["a.pdf", "b.pdf", "c.pdf"], chunk_size=25)
        assert calls == [("title", "a.pdf"), ("title", "b.pdf"), ("title", "c.pdf"),
                         ("body", "a.pdf"), ("body", "b.pdf"), ("body", "c.pdf")]

    def test_chunks_alternate_the_two_phases(self, batch):
        run, calls, _ = batch
        run(["a.pdf", "b.pdf", "c.pdf"], chunk_size=2)
        assert calls == [("title", "a.pdf"), ("title", "b.pdf"),
                         ("body", "a.pdf"), ("body", "b.pdf"),
                         ("title", "c.pdf"), ("body", "c.pdf")]

    def test_chunk_size_one_is_title_then_body_per_document(self, batch):
        run, calls, _ = batch
        run(["a.pdf", "b.pdf"], chunk_size=1)
        assert calls == [("title", "a.pdf"), ("body", "a.pdf"),
                         ("title", "b.pdf"), ("body", "b.pdf")]

    def test_the_body_phase_gets_the_title_page_the_model_returned(self, batch):
        run, _, written = batch
        run(["a.pdf", "b.pdf"], chunk_size=25)
        assert written == {"a.pdf": "title of a.pdf", "b.pdf": "title of b.pdf"}

    def test_a_book_has_no_title_phase(self, batch):
        run, calls, _ = batch
        run(["a.pdf", "book.pdf", "b.pdf"], chunk_size=25)
        assert ("title", "book.pdf") not in calls
        assert calls[-3:] == [("body", "a.pdf"), ("book", "book.pdf"), ("body", "b.pdf")]

    def test_results_keep_the_preflight_order(self, batch):
        run, _, _ = batch
        results = run(["c.pdf", "a.pdf", "b.pdf"], chunk_size=2)
        assert [Path(key).name for key in results] == ["c.pdf", "a.pdf", "b.pdf"]


# --- failures, crashes, clean-up (gate marks 4 and 5) -------------------------

class TestFailuresAndCrashes:
    def test_a_title_phase_failure_is_read_again_with_its_body(self, batch):
        run, calls, written = batch
        results = run(["a.pdf", "b.pdf"], chunk_size=25, fail_title={"a.pdf"})
        assert calls == [("title", "a.pdf"), ("title", "b.pdf"),
                         ("title", "a.pdf"), ("body", "a.pdf"), ("body", "b.pdf")]
        assert written["a.pdf"] == "title of a.pdf"
        assert all(not line.get("error") for line in results.values())

    def test_the_saved_title_pages_are_removed_after_a_full_run(self, batch, tmp_path):
        run, _, _ = batch
        run(["a.pdf", "b.pdf"], chunk_size=25)
        assert not (tmp_path / "out" / TITLE_PAGE_FOLDER).exists()

    def test_after_a_crash_they_stay_and_the_rerun_reuses_them(self, batch, tmp_path):
        run, calls, written = batch
        with pytest.raises(KeyboardInterrupt):
            run(["a.pdf", "b.pdf", "c.pdf"], chunk_size=25, crash_on="b.pdf")
        folder = tmp_path / "out" / TITLE_PAGE_FOLDER
        assert sorted(p.name for p in folder.iterdir()) == [
            "a.title.json", "b.title.json", "c.title.json"]

        calls.clear()
        written.clear()
        run(["a.pdf", "b.pdf", "c.pdf"], chunk_size=25)
        assert not any(kind == "title" for kind, _ in calls)   # no title model call
        assert written == {"a.pdf": "title of a.pdf", "b.pdf": "title of b.pdf",
                           "c.pdf": "title of c.pdf"}
        assert not folder.exists()

    def test_a_saved_page_from_other_settings_is_read_again(self, batch, tmp_path):
        run, calls, _ = batch
        with pytest.raises(KeyboardInterrupt):
            run(["a.pdf"], chunk_size=25, crash_on="a.pdf")
        calls.clear()
        run(["a.pdf"], chunk_size=25, key="settings-2")
        assert calls == [("title", "a.pdf"), ("body", "a.pdf")]


# --- the store itself (gate mark 6 and the trust rules) -----------------------

class TestTheStore:
    @staticmethod
    def saved(tmp_path, page=None):
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF one")
        store = TitlePageStore(tmp_path / "out")
        page = page or ContentPage(page_number=1, content="Title\n\nAbstract ü",
                                   quality_score=0.85,
                                   metadata={"shot1_time": 37.34, "nested": {"k": [1, 2]}})
        store.save(pdf, "key", page, elapsed=40.0)
        return store, pdf, page

    def test_a_reading_comes_back_unchanged(self, tmp_path):
        store, pdf, page = self.saved(tmp_path)
        loaded = store.load(pdf, "key")
        assert loaded.page.model_dump() == page.model_dump()
        assert loaded.elapsed == 40.0

    def test_the_round_trip_is_logged(self, tmp_path, caplog):
        caplog.set_level("INFO")
        self.saved(tmp_path)
        assert "round trip identical" in caplog.text

    def test_other_settings_are_not_trusted(self, tmp_path):
        store, pdf, _ = self.saved(tmp_path)
        assert store.load(pdf, "another key") is None

    def test_a_changed_pdf_is_not_trusted(self, tmp_path):
        store, pdf, _ = self.saved(tmp_path)
        pdf.write_bytes(b"%PDF two")
        assert store.load(pdf, "key") is None

    def test_another_file_format_is_not_trusted(self, tmp_path):
        store, pdf, _ = self.saved(tmp_path)
        path = store.folder / "paper.title.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["format"] = 0
        path.write_text(json.dumps(record), encoding="utf-8")
        assert store.load(pdf, "key") is None

    def test_a_damaged_file_is_not_trusted(self, tmp_path):
        store, pdf, _ = self.saved(tmp_path)
        (store.folder / "paper.title.json").write_text("{not json", encoding="utf-8")
        assert store.load(pdf, "key") is None

    def test_discard_removes_the_file_and_the_empty_folder(self, tmp_path):
        store, pdf, _ = self.saved(tmp_path)
        store.discard(pdf)
        assert not store.folder.exists()


# --- where the chunk size comes from ------------------------------------------

class TestTheChunkSize:
    def test_the_command_line_wins(self):
        request = RunRequest(chunk_size=3)
        assert runner._chunk_size(request, runner.LOG) == 3

    def test_the_config_decides_otherwise(self):
        assert runner._chunk_size(RunRequest(), runner.LOG) == 25

    def test_it_is_never_below_one(self):
        assert runner._chunk_size(RunRequest(chunk_size=0), runner.LOG) == 25
        assert runner._chunk_size(RunRequest(chunk_size=-4), runner.LOG) == 1

    def test_it_travels_to_a_worker(self):
        assert RunRequest.from_dict(RunRequest(chunk_size=7).to_dict()).chunk_size == 7
