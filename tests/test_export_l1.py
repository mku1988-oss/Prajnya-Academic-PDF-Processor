"""
Tests for audit L1: a portable export of an output file.

For biology researchers: the shared copy must hold only the PDF's file name
in `profile.pdf_path`, change nothing else, and still validate against the
output contract.
"""
import json
from pathlib import Path

import jsonschema

from src.processing.export import default_destination, export_document, main, portable_copy
from src.processing.paths import REPO_ROOT

EXAMPLE = next((REPO_ROOT / "contract" / "examples").rglob("*.v6.json"))
SCHEMA = json.loads((REPO_ROOT / "contract" / "prajnya_document.schema.json").read_text(encoding="utf-8"))


class TestPortableCopy:
    def test_a_windows_path_becomes_its_file_name(self):
        doc = {"profile": {"pdf_path": r"D:\Lab\Projects\paper one.pdf", "processing_result": {}}}
        assert portable_copy(doc)["profile"]["pdf_path"] == "paper one.pdf"

    def test_a_posix_path_becomes_its_file_name(self):
        assert portable_copy({"profile": {"pdf_path": "/home/lab/paper.pdf"}})["profile"]["pdf_path"] == "paper.pdf"

    def test_the_input_is_not_changed(self):
        doc = {"profile": {"pdf_path": r"D:\Lab\paper.pdf"}}
        portable_copy(doc)
        assert doc["profile"]["pdf_path"] == r"D:\Lab\paper.pdf"

    def test_the_real_example_still_validates_and_nothing_else_changes(self):
        original = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        shared = portable_copy(original)
        jsonschema.validate(shared, SCHEMA)
        assert shared["profile"]["pdf_path"] == Path(original["profile"]["pdf_path"].replace("\\", "/")).name
        shared["profile"]["pdf_path"] = original["profile"]["pdf_path"]
        assert shared == original


class TestExportFile:
    def test_the_copy_is_written_beside_the_original(self, tmp_path):
        source = tmp_path / "paper.v6.json"
        source.write_text(json.dumps({"profile": {"pdf_path": r"C:\Data\me\paper.pdf"}}), encoding="utf-8")
        written = export_document(source)
        assert written == tmp_path / "paper.portable.json"
        assert json.loads(written.read_text(encoding="utf-8"))["profile"]["pdf_path"] == "paper.pdf"
        assert "Data" in source.read_text(encoding="utf-8")       # the original is untouched

    def test_the_default_name_is_not_picked_up_as_an_output(self):
        assert not default_destination(Path("a/paper.v6.json")).name.endswith(".v6.json")

    def test_out_takes_one_document_only(self, tmp_path):
        assert main(["a.v6.json", "b.v6.json", "--out", str(tmp_path / "x.json")]) == 2
