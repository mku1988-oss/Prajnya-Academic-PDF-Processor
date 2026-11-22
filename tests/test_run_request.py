#!/usr/bin/env python3
"""
The run's order form, and the one sentinel that must survive a round trip.

For biology researchers: a `RunRequest` is the description of what a run was
asked to do — which PDFs, where the output goes, which pages. The command line
builds one; the graphical interface will build one; and a worker process will
rebuild one from a file after it has been written out as JSON.

That last part is why these tests exist. **`--chapters` carries a sentinel that
a careless test for emptiness destroys:**

- `chapters=None` -> read no chapter bodies. Seconds, no GPU. The safe default.
- `chapters=[]`   -> read EVERY chapter. For a 345-page volume, hours of GPU.

Those are opposite instructions, and one of them is falsy. This project has
already been bitten by exactly that collision once: a 345-page book "finished"
in 1.5 seconds having read nothing at all, and exited 0, because `[]` meant
"every chapter" in one place and "no chapters" in another (see
`tests/test_all_chapters.py`). JSON round-trips `null` and `[]` distinctly, so
the codec is not the risk — the risk is a truthiness test creeping into new
code. Hence the source-inspection guard at the bottom.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from src.processing.paths import CONFIG_ROOT, DEFAULT_OUTPUT_ROOT
from src.processing.run_request import DocumentOutcome, RunRequest


def make_args(**overrides) -> Namespace:
    """A parsed command line with everything at its default."""
    values = dict(pdf="D:/library/paper.pdf", out=None, backend=None, pages=None,
                  chapters=None, max_pages=None, front_matter_pages=10,
                  force=False, recursive=False, noninteractive=False,
                  verbose=False)
    values.update(overrides)
    return Namespace(**values)


class TestTheChaptersSentinel:
    """
    `None` and `[]` are opposite instructions. All four cases, both directions.
    """

    @pytest.mark.parametrize("chapters", [None, [], [3], [3, 4, 5]])
    def test_it_survives_the_round_trip_unchanged(self, chapters):
        request = RunRequest(chapters=chapters)
        restored = RunRequest.from_dict(request.to_dict())
        assert restored.chapters == chapters

    def test_none_and_empty_stay_distinguishable_after_the_round_trip(self):
        """The one that matters: hours of GPU versus seconds of CPU."""
        no_bodies = RunRequest.from_dict(RunRequest(chapters=None).to_dict())
        every_chapter = RunRequest.from_dict(RunRequest(chapters=[]).to_dict())

        assert no_bodies.chapters is None
        assert every_chapter.chapters == []
        assert no_bodies.chapters != every_chapter.chapters

    def test_the_key_is_always_written_even_when_empty(self):
        """
        A key omitted because its value was falsy reads back as None — which
        here means the OPPOSITE of what was asked for.
        """
        assert "chapters" in RunRequest(chapters=[]).to_dict()
        assert RunRequest(chapters=[]).to_dict()["chapters"] == []

    def test_to_dict_contains_no_truthiness_test_on_chapters(self):
        """
        A guard against the defect rather than against one instance of it.

        The three ways `chapters` gets tested for truthiness instead of for
        presence are all silent: each turns "read every chapter" into "read
        none", and each passes every value-based test above right up until
        someone writes it. The test the code must use is `is not None`.
        """
        import inspect

        source = inspect.getsource(RunRequest.to_dict)
        for dangerous in ("if self.chapters:", "if self.chapters else",
                          "if not self.chapters"):
            assert dangerous not in source, f"truthiness test on the sentinel: {dangerous}"
        assert "self.chapters is not None" in source


class TestTheRoundTripKeepsEverythingElse:
    def test_every_field_survives(self):
        request = RunRequest(
            paths=[Path("D:/a.pdf"), Path("D:/b.pdf")],
            config_root=Path("D:/conf"), out_root=Path("D:/out"),
            out_document=Path("D:/out/one.json"), backend="pymupdf",
            pages=[1, 4, 7], chapters=[2], max_pages=40, front_matter_pages=12,
            force=True, recursive=True, noninteractive=True, verbose=True,
            ledger_path=Path("D:/led.csv"), book_overrides={"policy": "all"})

        restored = RunRequest.from_dict(request.to_dict())
        assert restored == request

    def test_pages_none_is_not_an_empty_list(self):
        """`--pages` unset means every page, which is not the same as no pages."""
        assert RunRequest.from_dict(RunRequest(pages=None).to_dict()).pages is None
        assert RunRequest.from_dict(RunRequest(pages=[]).to_dict()).pages == []

    def test_absent_optional_paths_do_not_become_the_current_directory(self):
        """`Path(None)` raises and `Path("")` is `.` — both would be wrong."""
        restored = RunRequest.from_dict(RunRequest().to_dict())
        assert restored.out_document is None
        assert restored.ledger_path is None

    def test_a_bare_dict_missing_keys_still_builds(self):
        """A worker's job file written by an older version must not crash."""
        restored = RunRequest.from_dict({})
        assert restored.paths == []
        assert restored.chapters is None
        assert restored.config_root == CONFIG_ROOT


class TestFromArgs:
    def test_a_single_pdf_becomes_a_one_path_request(self):
        request = RunRequest.from_args(make_args())
        assert request.paths == [Path("D:/library/paper.pdf")]
        assert request.out_root == DEFAULT_OUTPUT_ROOT

    def test_a_folder_carries_the_paths_the_caller_found(self):
        paths = [Path("D:/library/a.pdf"), Path("D:/library/b.pdf")]
        request = RunRequest.from_args(make_args(pdf="D:/library"), paths=paths)
        assert request.paths == paths

    def test_an_out_file_for_one_pdf_makes_its_folder_the_run_root(self):
        """
        `--out paper.json` names a FILE, so its FOLDER is the run root.

        The file itself cannot be the root — figures would land inside a path
        that is a JSON file. Its directory can, and should: that is what puts
        one document's crops beside the document. See
        `tests/test_output_anchoring.py` for the behaviour this enables.
        """
        request = RunRequest.from_args(make_args(out="D:/elsewhere/mine.json"))
        assert request.out_document == Path("D:/elsewhere/mine.json")
        assert request.out_root == Path("D:/elsewhere")

    def test_without_out_a_single_pdf_uses_the_repository_default(self):
        """The default is what must never move: `<repo>/run_outputs`."""
        assert RunRequest.from_args(make_args()).out_root == DEFAULT_OUTPUT_ROOT

    def test_flags_are_carried_verbatim(self):
        request = RunRequest.from_args(make_args(force=True, recursive=True,
                                                 noninteractive=True, verbose=True,
                                                 pages=[2, 3], max_pages=40,
                                                 backend="pymupdf",
                                                 front_matter_pages=4))
        assert (request.force, request.recursive, request.noninteractive,
                request.verbose) == (True, True, True, True)
        assert request.pages == [2, 3]
        assert request.max_pages == 40
        assert request.backend == "pymupdf"
        assert request.front_matter_pages == 4

    def test_chapters_with_no_numbers_reaches_the_request_as_empty_not_none(self):
        """`--chapters` with no numbers is a deliberate request for everything."""
        assert RunRequest.from_args(make_args(chapters=[])).chapters == []
        assert RunRequest.from_args(make_args(chapters=None)).chapters is None


class TestDocumentOutcome:
    def test_a_finished_document_succeeded(self):
        outcome = DocumentOutcome(path=Path("a.pdf"), output=Path("a.json"),
                                  elapsed=12.0)
        assert outcome.succeeded

    def test_a_failed_document_did_not(self):
        assert not DocumentOutcome(path=Path("a.pdf"), error="Boom: no").succeeded

    def test_a_skipped_document_did_not(self):
        assert not DocumentOutcome(path=Path("a.pdf"),
                                   skipped="already in the ledger").succeeded
