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
                               "llama_server", "gpu", "source", "config_sha256"}

    def test_no_full_path_is_recorded(self):
        record = run_provenance(Path(__file__).resolve().parent.parent / "config")
        text = json.dumps(record)
        assert ":/" not in text and ":\\\\" not in text


class TestInTheDocument:
    def test_a_written_document_carries_it_and_is_valid(self, tmp_path):
        record = {"python": "3.10.16", "os": "Windows 10", "packages": {"numpy": "1.26.4"},
                  "pdfium": None, "layout_model": None, "models": [], "llama_server": None, "gpu": None,
                  "source": None, "config_sha256": None}
        document = build_document_json(_profile(tmp_path), 1.0, provenance=record)
        assert CONTRACT_VERSION == "1.7"
        assert document["document_info"]["provenance"] == record
        assert _errors(document) == []

    def test_unknown_provenance_is_written_as_null_and_is_valid(self, tmp_path):
        document = build_document_json(_profile(tmp_path), 1.0)
        assert document["document_info"]["provenance"] is None
        assert _errors(document) == []

    def test_a_1_7_provenance_without_the_commit_or_hash_is_refused(self, tmp_path):
        record = {"python": "3.10.16", "os": "Windows 10", "packages": {}, "pdfium": None,
                  "layout_model": None, "models": [], "llama_server": None, "gpu": None}
        document = build_document_json(_profile(tmp_path), 1.0, provenance=record)
        assert _errors(document) != []

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
        assert info["contract_version"] == "1.7"
        assert len(info["provenance"]["config_sha256"]) == 64
        assert info["provenance"]["python"] and isinstance(info["provenance"]["packages"], dict)
        assert _errors(written) == []


class TestContract17SourceAndConfigHash:
    """Contract 1.7 (audit 2.4): the git commit and one hash of the effective settings."""

    def tree(self, tmp_path: Path, model_path: str = "X:/models/Reader.Q4_K_M.gguf") -> Path:
        root = tmp_path / "config"
        (root / "vlm_profiles").mkdir(parents=True)
        (root / "factory_defaults").mkdir()
        (root / "research_article_strategy.yaml").write_text(
            "# a comment\ncitations:\n  level: local\nkey: ${SECRET_KEY}\n", encoding="utf-8")
        (root / "vlm_profiles" / "model.ini").write_text(
            f"[reader]\nmodel = {model_path}\nubatch-size = 1024\n", encoding="utf-8")
        (root / "factory_defaults" / "research_article_strategy.yaml").write_text(
            "anything: 1\n", encoding="utf-8")
        return root

    def config_hash(self, root: Path, overrides=None) -> str:
        from src.processing.run_provenance import config_tree_digest, effective_config_hash
        return effective_config_hash(config_tree_digest(root), overrides or {})

    def test_comments_line_endings_and_model_folders_do_not_change_it(self, tmp_path):
        first = self.config_hash(self.tree(tmp_path / "a"))
        other = self.tree(tmp_path / "b", model_path="D:/elsewhere/Reader.Q4_K_M.gguf")
        strategy = other / "research_article_strategy.yaml"
        strategy.write_bytes(strategy.read_bytes().replace(b"# a comment", b"# other")
                             .replace(b"\n", b"\r\n"))
        assert self.config_hash(other) == first

    def test_a_setting_or_an_override_changes_it(self, tmp_path):
        root = self.tree(tmp_path)
        before = self.config_hash(root)
        assert self.config_hash(root, {"citations": "online"}) != before
        (root / "vlm_profiles" / "model.ini").write_text(
            "[reader]\nmodel = X:/models/Reader.Q4_K_M.gguf\nubatch-size = 2048\n", encoding="utf-8")
        assert self.config_hash(root) != before

    def test_secrets_are_never_expanded_and_factory_defaults_are_ignored(self, tmp_path, monkeypatch):
        from src.processing.run_provenance import config_tree_digest
        root = self.tree(tmp_path)
        monkeypatch.setenv("SECRET_KEY", "abc123")
        digest = config_tree_digest(root)
        assert digest["research_article_strategy.yaml"]["key"] == "${SECRET_KEY}"
        assert not any(path.startswith("factory_defaults") for path in digest)
        assert digest["vlm_profiles/model.ini"]["reader"]["model"] == "Reader.Q4_K_M.gguf"

    def test_the_record_carries_the_commit_and_the_hash(self, tmp_path):
        record = run_provenance(self.tree(tmp_path), {"citations": "local"})
        assert len(record["config_sha256"]) == 64
        source = record["source"]
        assert source is None or (len(source["commit"]) == 40 and "dirty" in source)

    def test_outside_a_git_checkout_the_source_is_null(self, monkeypatch):
        monkeypatch.setattr(provenance_module, "_git", lambda repo, *arguments: None)
        assert provenance_module.source_record() is None
