"""
Tests for contract 1.6 (audit D8 + L5): what produced a run, in every output file.

For biology researchers: each output file records the model files, their
quantisation, the library versions and the GPU that produced it, as file
names only. A part that cannot be read on a machine (no GPU, no model
server) is null and never stops a run.
"""
import json
from pathlib import Path

import pytest

from src.processing import run_provenance as provenance_module
from src.processing.document_json import CONTRACT_VERSION, build_document_json
from src.processing.run_provenance import model_records, quantisation_of, run_provenance
from tests.test_output_contract import _errors, _profile


class TestQuantisation:
    @pytest.mark.parametrize("name, expected", [
        ("olmOCR-2-7B-1025.Q5_K_M.gguf", "Q5_K_M"),
        ("GLM-OCR-f16.gguf", "F16"),
        ("deepseek-ocr-q8_0.gguf", "Q8_0"),
        ("Qwen3-35B-A3B-UD-IQ4_XS.gguf", "IQ4_XS"),
        ("model.gguf", None),
    ])
    def test_the_quantisation_is_read_from_the_file_name(self, name, expected):
        assert quantisation_of(name) == expected


class TestModelRecords:
    def config(self, tmp_path: Path) -> Path:
        root = tmp_path / "config"
        (root / "vlm_profiles").mkdir(parents=True)
        (root / "vlm_profiles" / "reader.yaml").write_text('model_name: "reader-ocr"\n', encoding="utf-8")
        (root / "vlm_profiles" / "model.ini").write_text(
            "[reader-ocr]\nmodel = X:/models/Reader-OCR.Q4_K_M.gguf ; the file\nctx-size = 4096\n",
            encoding="utf-8")
        return root

    def test_each_role_names_its_model_file_and_quantisation_never_a_path(self, tmp_path):
        root = self.config(tmp_path)
        strategy = {"profile_registry": {"extraction": "reader", "understanding": "cloud-model"}}
        records = model_records(root, strategy)
        assert records == [
            {"role": "extraction", "model": "reader-ocr", "file": "Reader-OCR.Q4_K_M.gguf",
             "quantisation": "Q4_K_M"},
            {"role": "understanding", "model": "cloud-model"},
        ]


class TestCollected:
    def test_a_probe_that_fails_gives_null_and_never_raises(self, monkeypatch, tmp_path):
        def broken():
            raise OSError("no nvidia-smi here")
        monkeypatch.setattr(provenance_module, "gpu_record", broken)
        provenance_module._cached_provenance.cache_clear()
        record = run_provenance(tmp_path)          # an empty config tree: everything optional
        provenance_module._cached_provenance.cache_clear()
        assert record["gpu"] is None and record["python"]
        assert set(record) == {"python", "os", "packages", "pdfium", "layout_model", "models",
                               "llama_server", "gpu"}

    def test_no_full_path_is_recorded(self):
        record = run_provenance(Path(__file__).resolve().parent.parent / "config")
        text = json.dumps(record)
        assert ":/" not in text and ":\\\\" not in text


class TestInTheDocument:
    def test_a_written_document_carries_it_and_is_valid(self, tmp_path):
        record = {"python": "3.10.16", "os": "Windows 10", "packages": {"numpy": "1.26.4"},
                  "pdfium": None, "layout_model": None, "models": [], "llama_server": None, "gpu": None}
        document = build_document_json(_profile(tmp_path), 1.0, provenance=record)
        assert CONTRACT_VERSION == "1.6"
        assert document["document_info"]["provenance"] == record
        assert _errors(document) == []

    def test_unknown_provenance_is_written_as_null_and_is_valid(self, tmp_path):
        document = build_document_json(_profile(tmp_path), 1.0)
        assert document["document_info"]["provenance"] is None
        assert _errors(document) == []

    def test_a_1_6_document_without_the_key_is_refused(self, tmp_path):
        document = build_document_json(_profile(tmp_path), 1.0)
        del document["document_info"]["provenance"]
        assert _errors(document) != []


class TestTheRunnerWritesIt:
    def test_process_one_writes_the_provenance_of_its_run(self, tmp_path, monkeypatch):
        """Through the real runner, with the strategy stubbed (no model needed)."""
        from src.processing import runner
        from src.processing.batch import PreflightEntry
        from src.processing.run_request import RunRequest

        profile = _profile(tmp_path)

        class StubStrategy:
            def process(self, given, test_pages_override=None):
                return profile

        monkeypatch.setattr(runner, "build_strategy", lambda *args, **kwargs: StubStrategy())
        request = RunRequest(out_root=tmp_path / "out")
        outcome = runner.process_one(PreflightEntry(path=profile.pdf_path), request, profile=profile)
        assert outcome.error is None
        written = json.loads(request.output_for(profile.pdf_path).read_text(encoding="utf-8"))
        info = written["document_info"]
        assert info["contract_version"] == "1.6"
        assert info["provenance"]["python"] and isinstance(info["provenance"]["packages"], dict)
        assert _errors(written) == []
