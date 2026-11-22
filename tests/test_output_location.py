#!/usr/bin/env python3
"""
Where a run's results land when nobody says where.

For biology researchers: this is about not losing your own output. Before
this, the default was *relative to the input*:

- one PDF wrote `<pdf>.v6.json` **beside the PDF**, scattering generated JSON
  through a reference library that should hold only papers;
- a folder run created a `run_outputs/` **inside the scanned folder**, so the
  same corpus processed from two different places left two unrelated trees
  and no way to tell which one was current.

Both now resolve to ONE place anchored to the repository —
`src/v6/run_outputs` — which is also where figures, tables and the batch
summary already go. `--out` still overrides, and overriding is the only thing
that changes the location.

The anchor is `Path(__file__).resolve().parent`, not the working directory:
a run launched from anywhere (a scheduler, a GUI, another drive) must write
to the same place, or "where did my output go" becomes a real question.
"""

from pathlib import Path

import run_prajnya
from run_prajnya import resolve_batch_output, resolve_document_output


REPO_ROOT = Path(run_prajnya.__file__).resolve().parent


class TestTheDefaultIsAnchoredToTheRepository:
    def test_the_root_sits_inside_the_v6_tree(self):
        assert run_prajnya.DEFAULT_OUTPUT_ROOT == REPO_ROOT / "run_outputs"

    def test_it_is_absolute_so_the_working_directory_cannot_move_it(self):
        """
        A relative default would follow whoever launched the run. This one
        does not — which is the whole point.
        """
        assert run_prajnya.DEFAULT_OUTPUT_ROOT.is_absolute()

    def test_it_is_the_same_place_figures_and_tables_already_go(self):
        assert run_prajnya.DEFAULT_OUTPUT_ROOT.name == "run_outputs"


class TestASingleDocument:
    def test_it_no_longer_writes_beside_the_pdf(self):
        """THE DEFECT: a reference library must not accumulate our JSON."""
        pdf = Path("D:/some/reference/library/paper.pdf")
        out = resolve_document_output(pdf, None)

        assert out.parent != pdf.parent
        assert pdf.parent not in out.parents

    def test_it_lands_in_the_run_directory_under_its_own_name(self):
        out = resolve_document_output(Path("D:/library/paper.pdf"), None)
        assert out == run_prajnya.DEFAULT_OUTPUT_ROOT / "documents" / "paper.v6.json"

    def test_the_pdf_name_survives_spaces_and_punctuation(self):
        """Real filenames in this corpus have both."""
        out = resolve_document_output(
            Path("D:/library/Silberbush-2019-Fig2a,3.pdf"), None)
        assert out.name == "Silberbush-2019-Fig2a,3.v6.json"

    def test_an_explicit_out_still_wins(self):
        out = resolve_document_output(Path("D:/library/paper.pdf"),
                                      "D:/elsewhere/mine.json")
        assert out == Path("D:/elsewhere/mine.json")


class TestAFolder:
    def test_it_no_longer_writes_inside_the_scanned_folder(self):
        folder = Path("D:/some/reference/library")
        assert resolve_batch_output(None) != folder / "run_outputs"

    def test_it_lands_in_the_repository_run_directory(self):
        assert resolve_batch_output(None) == run_prajnya.DEFAULT_OUTPUT_ROOT

    def test_an_explicit_out_still_wins(self):
        assert resolve_batch_output("D:/runs/tonight") == Path("D:/runs/tonight")

    def test_both_entry_points_agree_on_the_root(self):
        """
        One PDF and a folder full of PDFs must not disagree about where
        results live — that is how a run goes missing.
        """
        single = resolve_document_output(Path("D:/library/paper.pdf"), None)
        assert resolve_batch_output(None) in single.parents


class TestTheCallersActuallyUseThis:
    """
    A resolver nothing calls is decoration. These pin the wiring.

    Both resolvers are now called from ONE place — `RunRequest.from_args` —
    rather than from two sibling functions in the entrypoint. That is the point
    of the runner seam: a single PDF and a folder of them cannot drift apart
    about where results go, because the same function decides for both. The
    GUI will build its own `RunRequest` and inherit the same answer.
    """

    def test_one_function_decides_the_location_for_both_shapes_of_run(self):
        import inspect

        from src.processing.run_request import RunRequest

        source = inspect.getsource(RunRequest.from_args)
        assert "resolve_batch_output(" in source
        assert "resolve_document_output(" in source

    def test_a_batch_request_puts_documents_under_the_run_root(self):
        """The behaviour the batch assertion used to stand in for."""
        request = _request(pdf="D:/library", out=None,
                           paths=[Path("D:/library/paper.pdf")])
        assert request.out_root == run_prajnya.DEFAULT_OUTPUT_ROOT
        assert (request.output_for(Path("D:/library/paper.pdf"))
                == run_prajnya.DEFAULT_OUTPUT_ROOT / "documents" / "paper.v6.json")

    def test_a_single_document_honours_an_explicit_out_file(self):
        """`--out` for one PDF names the JSON itself, exactly as given."""
        request = _request(pdf="D:/library/paper.pdf", out="D:/elsewhere/mine.json")
        assert request.output_for(Path("D:/library/paper.pdf")) == Path(
            "D:/elsewhere/mine.json")

    def test_a_single_document_without_out_lands_in_the_default_tree(self):
        request = _request(pdf="D:/library/paper.pdf", out=None)
        assert (request.output_for(Path("D:/library/paper.pdf"))
                == run_prajnya.DEFAULT_OUTPUT_ROOT / "documents" / "paper.v6.json")

    def test_an_explicit_run_directory_moves_a_batch(self):
        request = _request(pdf="D:/library", out="D:/runs/tonight",
                           paths=[Path("D:/library/paper.pdf")])
        assert (request.output_for(Path("D:/library/paper.pdf"))
                == Path("D:/runs/tonight") / "documents" / "paper.v6.json")


def _request(pdf, out, paths=None):
    """A RunRequest built the way the command line builds one."""
    from argparse import Namespace

    from src.processing.run_request import RunRequest

    args = Namespace(pdf=pdf, out=out, backend=None, pages=None, chapters=None,
                     max_pages=None, front_matter_pages=10, force=False,
                     recursive=False, noninteractive=False, verbose=False)
    return RunRequest.from_args(args, paths=paths)
