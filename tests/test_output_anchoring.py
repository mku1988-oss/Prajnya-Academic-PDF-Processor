#!/usr/bin/env python3
"""
Figures and tables land with their documents — not wherever you were standing.

For biology researchers: a run produces three kinds of file — the document
JSON, the figure crops, and the table crops. They belong together. Until this
change only the JSON honoured `--out`; **figures and tables were written to a
path relative to the current working directory**, so:

    cd D:/somewhere/else
    run_prajnya.py --pdf paper.pdf --out D:/foo

put the documents in `D:/foo` and the figures in `D:/somewhere/else/
run_outputs/figures`. One run, two unrelated places, and nothing said so. The
keys that could have overridden it (`figures.output_dir`, `figures.path_root`,
`tables.output_dir`) are absent from `config/research_article_strategy.yaml`,
so the CWD-relative defaults always applied.

**This is a deliberate behaviour change, not only a fix.** `--out D:/foo` now
puts figures at `D:/foo/figures`. What has NOT changed is the default: with no
`--out`, everything still goes to `<repo>/run_outputs`, which is the same
absolute place a run launched from the repository root always produced.

The `monkeypatch.chdir` tests are the ones that matter — they are what proves
the working directory can no longer move anything.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from src.processing.paths import (
    DEFAULT_OUTPUT_ROOT,
    REPO_ROOT,
    anchor_output_path,
    resolve_batch_output,
)
from src.processing.run_request import RunRequest


class TestNothingConfiguredMeansTheRunRoot:
    """The case that always applies today: the keys are absent from the YAML."""

    def test_a_name_becomes_a_subdirectory_of_the_root(self):
        assert (anchor_output_path(None, Path("D:/foo"), "figures")
                == Path("D:/foo/figures"))

    def test_no_name_means_the_root_itself(self):
        """`figures.path_root` — manifests keep image paths relative to it."""
        assert anchor_output_path(None, Path("D:/foo")) == Path("D:/foo")

    def test_an_empty_string_is_treated_as_absent(self):
        assert (anchor_output_path("", Path("D:/foo"), "tables")
                == Path("D:/foo/tables"))

    def test_no_root_falls_back_to_the_repository_default(self):
        """A caller that never passes a root must still write somewhere sane."""
        assert (anchor_output_path(None, None, "figures")
                == DEFAULT_OUTPUT_ROOT / "figures")


class TestConfiguredPaths:
    def test_a_relative_path_resolves_against_the_run_root(self):
        """The fix: relative means 'under this run', never 'under the CWD'."""
        assert (anchor_output_path("crops", Path("D:/foo"), "figures")
                == Path("D:/foo/crops"))

    def test_an_absolute_path_is_taken_exactly_as_given(self):
        """Someone who names a whole path means it."""
        assert (anchor_output_path("E:/shared/figures", Path("D:/foo"), "figures")
                == Path("E:/shared/figures"))


class TestTheWorkingDirectoryCannotMoveAnything:
    """
    The gotcha, killed. Each of these would have failed before the change.
    """

    def test_the_default_root_ignores_the_working_directory(self, monkeypatch,
                                                            tmp_path):
        monkeypatch.chdir(tmp_path)
        assert anchor_output_path(None, None, "figures").is_absolute()
        assert tmp_path not in anchor_output_path(None, None, "figures").parents

    def test_a_relative_config_value_ignores_the_working_directory(self,
                                                                   monkeypatch,
                                                                   tmp_path):
        monkeypatch.chdir(tmp_path)
        resolved = anchor_output_path("run_outputs/figures", Path("D:/foo"),
                                      "figures")
        assert resolved == Path("D:/foo/run_outputs/figures")
        assert tmp_path not in resolved.parents

    def test_the_batch_default_ignores_the_working_directory(self, monkeypatch,
                                                             tmp_path):
        monkeypatch.chdir(tmp_path)
        assert resolve_batch_output(None) == DEFAULT_OUTPUT_ROOT
        assert resolve_batch_output(None) != tmp_path / "run_outputs"


class TestOneRunLandsInOnePlace:
    """
    Documents, figures and tables share a root — for every shape of run.
    """

    def test_a_batch_without_out_uses_the_repository_default(self):
        request = _request(pdf="D:/library", paths=[Path("D:/library/a.pdf")])
        assert request.out_root == DEFAULT_OUTPUT_ROOT
        assert (anchor_output_path(None, request.out_root, "figures")
                == DEFAULT_OUTPUT_ROOT / "figures")

    def test_a_batch_with_out_moves_figures_too(self):
        """The live bug: --out used to move documents/ and leave figures behind."""
        request = _request(pdf="D:/library", out="D:/runs/tonight",
                           paths=[Path("D:/library/a.pdf")])
        document = request.output_for(Path("D:/library/a.pdf"))
        figures = anchor_output_path(None, request.out_root, "figures")
        tables = anchor_output_path(None, request.out_root, "tables")

        assert document == Path("D:/runs/tonight/documents/a.v6.json")
        assert figures == Path("D:/runs/tonight/figures")
        assert tables == Path("D:/runs/tonight/tables")
        assert document.parent.parent == figures.parent == tables.parent

    def test_a_single_pdf_with_an_out_file_keeps_its_figures_beside_it(self):
        """`--out D:/elsewhere/mine.json` -> figures at D:/elsewhere/figures."""
        request = _request(pdf="D:/library/paper.pdf", out="D:/elsewhere/mine.json")
        assert request.out_root == Path("D:/elsewhere")
        assert (anchor_output_path(None, request.out_root, "figures")
                == Path("D:/elsewhere/figures"))

    def test_a_single_pdf_without_out_uses_the_repository_default(self):
        """The default is unchanged — this is what stops the fix moving things."""
        request = _request(pdf="D:/library/paper.pdf")
        assert request.out_root == DEFAULT_OUTPUT_ROOT
        assert (request.output_for(Path("D:/library/paper.pdf"))
                == DEFAULT_OUTPUT_ROOT / "documents" / "paper.v6.json")
        assert (anchor_output_path(None, request.out_root, "figures")
                == DEFAULT_OUTPUT_ROOT / "figures")


class TestTheChainIsActuallyWired:
    """
    A resolver nothing calls is decoration — and this project has shipped a
    whole pass that was never imported while 916 tests stayed green.
    """

    def test_the_router_accepts_an_output_root(self):
        import inspect

        from src.processing.strategies.router import build_strategy

        assert "output_root" in inspect.signature(build_strategy).parameters

    def test_the_body_pipeline_accepts_an_output_root(self):
        import inspect

        from src.processing.strategies.research_article.body_pipeline import (
            BodyPipeline,
        )

        assert "output_root" in inspect.signature(BodyPipeline.from_config).parameters

    def test_both_strategies_accept_an_output_root(self):
        import inspect

        from src.processing.strategies.book_chapter.strategy import (
            BookChapterStrategy,
        )
        from src.processing.strategies.research_article.strategy import (
            ResearchArticleStrategy,
        )

        for cls in (ResearchArticleStrategy, BookChapterStrategy):
            assert "output_root" in inspect.signature(cls.__init__).parameters

    def test_the_body_pipeline_no_longer_holds_a_cwd_relative_default(self):
        """
        The defect itself, pinned. `Path("run_outputs/figures")` is relative,
        so it followed whoever launched the run.
        """
        import inspect

        from src.processing.strategies.research_article import body_pipeline

        source = inspect.getsource(body_pipeline.BodyPipeline.from_config)
        assert '"run_outputs/figures"' not in source
        assert '"run_outputs/tables"' not in source
        assert source.count("anchor_output_path(") == 3

    def test_the_runner_hands_the_request_root_to_the_strategy(self):
        import inspect

        from src.processing.runner import process_one

        assert "output_root=request.out_root" in inspect.getsource(process_one)


class TestTheRepositoryOwnsItsOwnAssets:
    """
    Three more paths that followed the working directory, found by running the
    entrypoint from another folder — not by any unit test.

    The ONNX model one is loud: the run dies with "layout model not found".
    The other two are silent and worse. A CWD-relative ledger means a run
    launched from elsewhere builds a SECOND record book there, so "have I
    already processed this?" answers no, every document is re-processed, and
    the review queue splits in two with neither half aware of the other.
    Nothing fails — the library just stops remembering.
    """

    def test_the_layout_model_is_anchored_to_the_repository(self):
        from src.processing.strategies.research_article.body_pipeline import (
            DEFAULT_ONNX_MODEL_PATH,
        )

        assert DEFAULT_ONNX_MODEL_PATH.is_absolute()
        assert DEFAULT_ONNX_MODEL_PATH == (
            REPO_ROOT / "models" / "layout" / "pp-doclayoutv3.onnx")

    def test_the_ledger_and_cache_are_anchored_to_the_repository(self):
        from src.processing.paths import (
            DEFAULT_LEDGER_PATH,
            DEFAULT_ZOTERO_CACHE_PATH,
        )

        for value in (DEFAULT_LEDGER_PATH, DEFAULT_ZOTERO_CACHE_PATH):
            assert Path(value).is_absolute(), f"{value} follows the working directory"
            assert REPO_ROOT in Path(value).parents

    def test_the_working_directory_cannot_fork_the_ledger(self, monkeypatch,
                                                          tmp_path):
        """The silent one, pinned where it would actually bite."""
        from src.processing.paths import DEFAULT_LEDGER_PATH

        monkeypatch.chdir(tmp_path)
        assert Path(DEFAULT_LEDGER_PATH).resolve() != (
            tmp_path / "persistence" / "semantic_ledger.csv")

    def test_the_strategy_uses_the_shared_constant_not_its_own_copy(self):
        """
        It held a second literal `"persistence/semantic_ledger.csv"`. Two
        definitions of one path is how half a fix ships.
        """
        import inspect

        from src.processing.strategies.research_article import strategy

        source = inspect.getsource(strategy)
        assert '"persistence/semantic_ledger.csv"' not in source
        assert '"persistence/zotero_cache"' not in source
        assert "DEFAULT_LEDGER_PATH" in source


def _request(pdf, out=None, paths=None) -> RunRequest:
    """A RunRequest built the way the command line builds one."""
    args = Namespace(pdf=pdf, out=out, backend=None, pages=None, chapters=None,
                     max_pages=None, front_matter_pages=10, force=False,
                     recursive=False, noninteractive=False, verbose=False)
    return RunRequest.from_args(args, paths=paths)
