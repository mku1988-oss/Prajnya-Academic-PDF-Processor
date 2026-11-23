"""
Tests for audit L6: worker job files are removed after a clean result.

For biology researchers: each document of a run writes a small job file so a
failure can be re-run by hand. Those files were never deleted. Now a job file
is removed once its document finished cleanly and kept when it failed.
"""
from pathlib import Path

from src.processing.batch import PreflightEntry
from src.processing.run_request import DocumentOutcome, RunRequest
from src.processing.worker import discard_job_if_done, write_job_file


def job(tmp_path: Path) -> Path:
    entry = PreflightEntry(path=tmp_path / "paper.pdf")
    return write_job_file(entry, RunRequest(out_root=tmp_path), directory=tmp_path / ".jobs")


class TestJobFiles:
    def test_a_clean_result_removes_its_job_file(self, tmp_path):
        path = job(tmp_path)
        discard_job_if_done(path, DocumentOutcome(path=tmp_path / "paper.pdf"))
        assert not path.exists()

    def test_a_failed_document_keeps_its_job_file_for_the_repro(self, tmp_path):
        path = job(tmp_path)
        discard_job_if_done(path, DocumentOutcome(path=tmp_path / "paper.pdf", error="WorkerFailed: x"))
        assert path.exists()

    def test_a_job_file_already_gone_is_not_an_error(self, tmp_path):
        path = job(tmp_path)
        path.unlink()
        discard_job_if_done(path, DocumentOutcome(path=tmp_path / "paper.pdf"))
