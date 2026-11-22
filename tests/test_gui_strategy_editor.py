#!/usr/bin/env python3
"""Tests for src/gui/strategy_editor.py — S12's config-tree editor."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.gui import strategy_editor as se

MODEL_INI_TEXT = """[glm-ocr]
model = D:/models/glm-ocr.gguf
mmproj = D:/models/glm-ocr.mmproj.gguf
ctx-size = 8192
n-gpu-layers = 99

[olmocr-2-7b]
model = D:/models/olmocr.gguf
ctx-size = 16384
n-gpu-layers = 99
"""

STRATEGY_YAML_TEXT = """# a leading comment untouched by anything
top_level_before:
  keep: this
profile_registry:
  default: olmocr-2-7b
  title_detection: olmocr-2-7b
  # --- Role-based offline roster ---
  # a longer explanation spanning
  # several lines, must survive.
  extraction: glm-ocr          # reads the pixels
  layout_detector: onnx        # PP-DocLayoutV3 on CPU

  figure_escalation: deepseek-ocr-v2
task_registry:
  raw_ocr: raw_ocr_extraction
  structured_title: structured_analysis_title_page
"""


# --- model.ini ---------------------------------------------------------

def test_read_model_ini_sections(tmp_path: Path):
    ini_path = tmp_path / "model.ini"
    ini_path.write_text(MODEL_INI_TEXT, encoding="utf-8")

    parser = se.read_model_ini(ini_path)

    assert se.model_ini_sections(parser) == ["glm-ocr", "olmocr-2-7b"]
    assert se.model_ini_section_dict(parser, "glm-ocr")["n-gpu-layers"] == "99"


def test_write_model_ini_section_updates_and_backs_up(tmp_path: Path):
    ini_path = tmp_path / "model.ini"
    ini_path.write_text(MODEL_INI_TEXT, encoding="utf-8")
    parser = se.read_model_ini(ini_path)

    backup = se.write_model_ini_section(
        parser, "glm-ocr", {"model": "D:/models/glm-ocr.gguf", "ctx-size": "16384"},
        path=ini_path,
    )

    assert backup.exists()
    assert "8192" in backup.read_text(encoding="utf-8")  # old value preserved in backup

    reparsed = se.read_model_ini(ini_path)
    section = se.model_ini_section_dict(reparsed, "glm-ocr")
    assert section["ctx-size"] == "16384"
    assert "mmproj" not in section  # dropped keys are actually removed, not left stale


def test_model_ini_text_round_trip_preserves_comments(tmp_path: Path):
    ini_path = tmp_path / "model.ini"
    text_with_comment = MODEL_INI_TEXT.replace(
        "[glm-ocr]", "; a hand-written comment\n[glm-ocr]"
    )
    ini_path.write_text(text_with_comment, encoding="utf-8")

    text = se.read_model_ini_text(ini_path)
    assert "; a hand-written comment" in text

    edited = text.replace("ctx-size = 8192", "ctx-size = 16384")
    backup = se.save_model_ini_text(edited, path=ini_path)

    assert backup.exists()
    assert "8192" in backup.read_text(encoding="utf-8")
    saved = se.read_model_ini_text(ini_path)
    assert "; a hand-written comment" in saved  # survived verbatim
    assert "16384" in saved


def test_save_model_ini_text_rejects_invalid_ini(tmp_path: Path):
    ini_path = tmp_path / "model.ini"
    ini_path.write_text(MODEL_INI_TEXT, encoding="utf-8")

    with pytest.raises(Exception):
        se.save_model_ini_text("not valid ini [[[", path=ini_path)
    assert se.read_model_ini_text(ini_path) == MODEL_INI_TEXT  # unchanged


def test_write_model_ini_section_can_add_a_new_section(tmp_path: Path):
    ini_path = tmp_path / "model.ini"
    ini_path.write_text(MODEL_INI_TEXT, encoding="utf-8")
    parser = se.read_model_ini(ini_path)

    se.write_model_ini_section(parser, "new-model", {"model": "D:/models/new.gguf"}, path=ini_path)

    reparsed = se.read_model_ini(ini_path)
    assert "new-model" in se.model_ini_sections(reparsed)


# --- vlm_profiles/<model>.yaml ------------------------------------------

def _write_profile(root: Path, name: str, data: dict) -> Path:
    path = root / f"{name}.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_list_and_load_vlm_profile(tmp_path: Path):
    _write_profile(tmp_path, "glm-ocr", {"model_name": "glm-ocr", "prompt_mode": "fixed"})
    _write_profile(tmp_path, "olmocr-2-7b", {"model_name": "olmocr-2-7b"})

    assert se.list_vlm_profiles(tmp_path) == ["glm-ocr", "olmocr-2-7b"]
    loaded = se.load_vlm_profile("glm-ocr", root=tmp_path)
    assert loaded["prompt_mode"] == "fixed"


def test_save_vlm_profile_backs_up_and_round_trips(tmp_path: Path):
    path = _write_profile(tmp_path, "glm-ocr", {"model_name": "glm-ocr", "prompt_mode": "fixed"})

    updated = se.load_vlm_profile("glm-ocr", root=tmp_path)
    updated["prompt_mode"] = "instruction"
    backup = se.save_vlm_profile("glm-ocr", updated, root=tmp_path)

    assert backup.exists()
    assert yaml.safe_load(backup.read_text(encoding="utf-8"))["prompt_mode"] == "fixed"
    assert se.load_vlm_profile("glm-ocr", root=tmp_path)["prompt_mode"] == "instruction"


def test_preview_vlm_profile_diff_does_not_write(tmp_path: Path):
    _write_profile(tmp_path, "glm-ocr", {"model_name": "glm-ocr"})
    before_text = se.load_vlm_profile_text("glm-ocr", root=tmp_path)

    updated = se.load_vlm_profile("glm-ocr", root=tmp_path)
    updated["model_name"] = "glm-ocr-v2"
    diff = se.preview_vlm_profile_diff("glm-ocr", updated, root=tmp_path)

    assert "glm-ocr-v2" in diff
    assert se.load_vlm_profile_text("glm-ocr", root=tmp_path) == before_text  # unchanged on disk


def test_save_vlm_profile_text_preserves_untouched_comments(tmp_path: Path):
    path = tmp_path / "glm-ocr.yaml"
    path.write_text("# a hand-written comment\nmodel_name: glm-ocr\nprompt_mode: fixed\n",
                    encoding="utf-8")

    edited = path.read_text(encoding="utf-8").replace("fixed", "instruction")
    backup = se.save_vlm_profile_text("glm-ocr", edited, root=tmp_path)

    assert backup.exists()
    saved = path.read_text(encoding="utf-8")
    assert "# a hand-written comment" in saved
    assert "instruction" in saved


def test_save_vlm_profile_text_rejects_invalid_yaml(tmp_path: Path):
    path = _write_profile(tmp_path, "glm-ocr", {"model_name": "glm-ocr"})
    original = path.read_text(encoding="utf-8")

    with pytest.raises(yaml.YAMLError):
        se.save_vlm_profile_text("glm-ocr", "key: [unclosed", root=tmp_path)
    assert path.read_text(encoding="utf-8") == original  # unchanged


def test_render_profile_yaml_round_trips():
    text = se.render_profile_yaml({"model_name": "new-model", "prompt_mode": "instruction"})
    assert yaml.safe_load(text) == {"model_name": "new-model", "prompt_mode": "instruction"}


def test_duplicate_vlm_profile_clones_and_renames(tmp_path: Path):
    _write_profile(tmp_path, "glm-ocr", {
        "model_name": "glm-ocr", "prompt_mode": "fixed",
        "resource_requirements": {"vram_per_instance_gb": 2.5},
    })

    cloned = se.duplicate_vlm_profile("glm-ocr", "glm-ocr-v2", "glm-ocr-v2", root=tmp_path)

    assert cloned["model_name"] == "glm-ocr-v2"
    assert cloned["prompt_mode"] == "fixed"  # shape carried over
    assert not (tmp_path / "glm-ocr-v2.yaml").exists()  # caller must still save() explicitly


def test_duplicate_vlm_profile_rejects_existing_name(tmp_path: Path):
    _write_profile(tmp_path, "glm-ocr", {"model_name": "glm-ocr"})
    _write_profile(tmp_path, "glm-ocr-v2", {"model_name": "glm-ocr-v2"})

    with pytest.raises(ValueError):
        se.duplicate_vlm_profile("glm-ocr", "glm-ocr-v2", "glm-ocr-v2", root=tmp_path)


def test_duplicate_vlm_profile_missing_source_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        se.duplicate_vlm_profile("nonexistent", "new", "new", root=tmp_path)


# --- profile_registry (surgical edit) ------------------------------------

@pytest.fixture
def strategy_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "research_article_strategy.yaml"
    path.write_text(STRATEGY_YAML_TEXT, encoding="utf-8")
    return path


def test_read_profile_registry(strategy_yaml: Path):
    roles = se.read_profile_registry(strategy_yaml)
    assert roles == {
        "default": "olmocr-2-7b",
        "title_detection": "olmocr-2-7b",
        "extraction": "glm-ocr",
        "layout_detector": "onnx",
        "figure_escalation": "deepseek-ocr-v2",
    }


def test_write_profile_registry_roles_changes_value_keeps_comment(strategy_yaml: Path):
    se.write_profile_registry_roles({"extraction": "deepseek-ocr-v2"}, strategy_yaml)

    text = strategy_yaml.read_text(encoding="utf-8")
    assert "extraction: deepseek-ocr-v2          # reads the pixels" in text


def test_write_profile_registry_roles_preserves_everything_outside_the_block(strategy_yaml: Path):
    se.write_profile_registry_roles({"extraction": "deepseek-ocr-v2"}, strategy_yaml)

    text = strategy_yaml.read_text(encoding="utf-8")
    assert "# a leading comment untouched by anything" in text
    assert "top_level_before:\n  keep: this" in text
    assert "task_registry:\n  raw_ocr: raw_ocr_extraction" in text
    assert "# --- Role-based offline roster ---" in text
    assert "several lines, must survive." in text


def test_write_profile_registry_roles_can_delete_an_optional_role(strategy_yaml: Path):
    se.write_profile_registry_roles({"figure_escalation": None}, strategy_yaml)

    text = strategy_yaml.read_text(encoding="utf-8")
    assert "figure_escalation" not in text
    roles = se.read_profile_registry(strategy_yaml)
    assert "figure_escalation" not in roles


def test_write_profile_registry_roles_can_add_a_new_role(strategy_yaml: Path):
    se.write_profile_registry_roles({"understanding": "olmocr-2-7b"}, strategy_yaml)

    roles = se.read_profile_registry(strategy_yaml)
    assert roles["understanding"] == "olmocr-2-7b"
    # the file must still parse as valid YAML afterwards
    parsed = yaml.safe_load(strategy_yaml.read_text(encoding="utf-8"))
    assert parsed["profile_registry"]["understanding"] == "olmocr-2-7b"


def test_write_profile_registry_roles_backs_up_first(strategy_yaml: Path):
    original = strategy_yaml.read_text(encoding="utf-8")
    backup = se.write_profile_registry_roles({"extraction": "olmocr-2-7b"}, strategy_yaml)

    assert backup.exists()
    assert backup.read_text(encoding="utf-8") == original


def test_preview_profile_registry_diff_does_not_write(strategy_yaml: Path):
    original = strategy_yaml.read_text(encoding="utf-8")
    diff = se.preview_profile_registry_diff({"extraction": "olmocr-2-7b"}, strategy_yaml)

    assert "extraction: olmocr-2-7b" in diff
    assert strategy_yaml.read_text(encoding="utf-8") == original


def test_write_profile_registry_roles_missing_block_raises(tmp_path: Path):
    path = tmp_path / "no_registry.yaml"
    path.write_text("something_else:\n  key: value\n", encoding="utf-8")

    with pytest.raises(ValueError):
        se.write_profile_registry_roles({"extraction": "glm-ocr"}, path)


# --- validators -----------------------------------------------------------

def test_validate_model_name_matches_ini_ok():
    error = se.validate_model_name_matches_ini({"model_name": "glm-ocr"}, ["glm-ocr", "olmocr-2-7b"])
    assert error is None


def test_validate_model_name_matches_ini_mismatch():
    error = se.validate_model_name_matches_ini({"model_name": "typo-name"}, ["glm-ocr"])
    assert error is not None
    assert "typo-name" in error


def test_validate_model_name_matches_ini_missing_name():
    error = se.validate_model_name_matches_ini({}, ["glm-ocr"])
    assert error is not None


def test_validate_model_name_matches_ini_exempts_cloud_profiles():
    profile = {"model_name": "gemini-2.5-flash", "model_type": "cloud_api_gemini"}
    assert se.validate_model_name_matches_ini(profile, ["glm-ocr"]) is None


def test_validate_fixed_prompts_for_role_ok():
    profile = {"prompt_mode": "fixed", "fixed_prompts": {"raw_ocr_extraction": "Text Recognition:"}}
    assert se.validate_fixed_prompts_for_role("extraction", profile) == []


def test_validate_fixed_prompts_for_role_missing_task():
    profile = {"prompt_mode": "fixed", "fixed_prompts": {}}
    missing = se.validate_fixed_prompts_for_role("extraction", profile)
    assert missing == ["raw_ocr_extraction"]


def test_validate_fixed_prompts_for_role_instruction_mode_skips_check():
    profile = {"prompt_mode": "instruction", "fixed_prompts": {}}
    assert se.validate_fixed_prompts_for_role("extraction", profile) == []


def test_validate_fixed_prompts_for_role_unchecked_role_is_always_clean():
    profile = {"prompt_mode": "fixed", "fixed_prompts": {}}
    assert se.validate_fixed_prompts_for_role("layout_detector", profile) == []


def test_compute_vram_warnings_flags_an_unfit_pair():
    # extraction + figure_escalation: both real body-phase roles (same
    # ROLE_PHASE_GROUPS entry), so a same-phase clash IS expected to fire.
    role_profiles = {
        "extraction": {"model_name": "glm-ocr",
                       "resource_requirements": {"vram_per_instance_gb": 6.0, "co_resident_with": []}},
        "figure_escalation": {"model_name": "deepseek-ocr-v2",
                              "resource_requirements": {"vram_per_instance_gb": 7.6, "co_resident_with": []}},
    }
    warnings = se.compute_vram_warnings(role_profiles, card_vram_gb=8.0)
    assert len(warnings) == 1
    assert "glm-ocr" in warnings[0] and "deepseek-ocr-v2" in warnings[0]


def test_compute_vram_warnings_excludes_the_unwired_understanding_role():
    # understanding has no caller in src/ today (session 11's design) — a
    # role that never runs cannot conflict with anything it's nominally
    # assigned alongside, even if the raw VRAM numbers would not fit.
    role_profiles = {
        "extraction": {"model_name": "glm-ocr",
                       "resource_requirements": {"vram_per_instance_gb": 6.0, "co_resident_with": []}},
        "understanding": {"model_name": "olmocr-2-7b",
                          "resource_requirements": {"vram_per_instance_gb": 7.6, "co_resident_with": []}},
    }
    assert se.compute_vram_warnings(role_profiles, card_vram_gb=8.0) == []


def _glm_and_deepseek():
    """The REAL shipped body-phase pair, at their declared VRAM figures —
    which agree with the 2026-07-10 measurements (glm 2535 MiB = 2.48 GB,
    deepseek delta 4929 MiB = 4.81 GB)."""
    return {
        "extraction": {"model_name": "glm-ocr",
                       "resource_requirements": {"vram_per_instance_gb": 2.5,
                                                 "co_resident_with": ["deepseek-ocr-v2"]}},
        "figure_escalation": {"model_name": "deepseek-ocr-v2",
                              "resource_requirements": {"vram_per_instance_gb": 4.9,
                                                        "co_resident_with": []}},
    }


def test_compute_vram_warnings_clean_for_the_real_pair_on_the_real_card():
    """2.5 + 4.9 + 0.5 overhead = 7.9 GB on an 8 GB card. Measured peak was
    7940 MiB of 8188 (248 MiB headroom), so this must stay silent — it is the
    project's own shipped, verified configuration."""
    assert se.compute_vram_warnings(_glm_and_deepseek(), card_vram_gb=8.0) == []


def test_compute_vram_warnings_declared_co_residency_does_not_override_a_smaller_card():
    """`co_resident_with` records a pair VERIFIED to co-reside on the 8188 MiB
    card. It is a statement about two models, not a promise about every GPU.
    On a 6 GB card the same pair still does not fit, and the old behaviour —
    trusting the declaration unconditionally — hid that on exactly the
    hardware most likely to hit it. Reversed after the measurements were
    recovered from SLM's `default` profile (session 17)."""
    warnings = se.compute_vram_warnings(_glm_and_deepseek(), card_vram_gb=6.0)

    assert len(warnings) == 1
    assert "declared co-resident" in warnings[0]
    assert "verified on a larger card" in warnings[0]


def test_compute_vram_warnings_counts_the_measured_vision_overhead():
    """Weights are not peak. glm + deepseek weigh 7464 MiB but peaked at 7940
    MiB under interleaved inference — 476 MiB of transient vision buffers.
    A pair summing to exactly the card size therefore does NOT fit."""
    role_profiles = {
        "extraction": {"model_name": "a",
                       "resource_requirements": {"vram_per_instance_gb": 4.0, "co_resident_with": []}},
        "figure_escalation": {"model_name": "b",
                              "resource_requirements": {"vram_per_instance_gb": 4.0, "co_resident_with": []}},
    }
    # 4.0 + 4.0 = 8.0 exactly — the OLD sum-of-weights check passed this.
    warnings = se.compute_vram_warnings(role_profiles, card_vram_gb=8.0)

    assert len(warnings) == 1
    assert "vision buffers" in warnings[0]


def test_compute_vram_warnings_overhead_constant_matches_the_measurement():
    """7940 MiB peak - 7464 MiB weights = 476 MiB = 0.465 GB, rounded up to
    0.5. Pinned so the constant cannot drift away from its evidence."""
    measured_gb = (7940 - 7464) / 1024
    assert measured_gb < se.VISION_TRANSIENT_OVERHEAD_GB <= 0.6


def test_compute_vram_warnings_no_card_detected_means_no_warnings():
    role_profiles = {
        "extraction": {"model_name": "glm-ocr",
                       "resource_requirements": {"vram_per_instance_gb": 99.0, "co_resident_with": []}},
    }
    assert se.compute_vram_warnings(role_profiles, card_vram_gb=None) == []


def test_compute_vram_warnings_ignores_cross_phase_pairs():
    # default/title_detection (title phase) vs extraction (body phase) — a
    # deliberate, already-budgeted swap between phases (_reserve_gpu_for),
    # not a same-phase co-residency failure. This is the exact pair the
    # live S12 click-through over-flagged before ROLE_PHASE_GROUPS existed.
    role_profiles = {
        "default": {"model_name": "olmocr-2-7b",
                   "resource_requirements": {"vram_per_instance_gb": 7.6, "co_resident_with": []}},
        "extraction": {"model_name": "glm-ocr",
                       "resource_requirements": {"vram_per_instance_gb": 2.5, "co_resident_with": []}},
    }
    assert se.compute_vram_warnings(role_profiles, card_vram_gb=8.0) == []


def test_compute_vram_warnings_same_model_twice_is_never_a_pair():
    role_profiles = {
        "default": {"model_name": "olmocr-2-7b",
                   "resource_requirements": {"vram_per_instance_gb": 7.6, "co_resident_with": []}},
        "title_detection": {"model_name": "olmocr-2-7b",
                            "resource_requirements": {"vram_per_instance_gb": 7.6, "co_resident_with": []}},
    }
    assert se.compute_vram_warnings(role_profiles, card_vram_gb=8.0) == []


# --- factory defaults (the idiot-proof reset) ------------------------------

@pytest.fixture
def config_tree(tmp_path: Path):
    """A live config tree and a factory-default tree that starts IDENTICAL
    to it — callers mutate the live side to simulate drift, then assert the
    restore functions bring it back."""
    live_root = tmp_path / "live"
    factory_root = tmp_path / "factory"
    (live_root / "vlm_profiles").mkdir(parents=True)
    (factory_root / "vlm_profiles").mkdir(parents=True)

    ini_text = MODEL_INI_TEXT
    (live_root / "model.ini").write_text(ini_text, encoding="utf-8")
    (factory_root / "model.ini").write_text(ini_text, encoding="utf-8")

    strategy_text = STRATEGY_YAML_TEXT
    (live_root / "strategy.yaml").write_text(strategy_text, encoding="utf-8")
    (factory_root / "strategy.yaml").write_text(strategy_text, encoding="utf-8")

    profile = {"model_name": "glm-ocr", "prompt_mode": "fixed"}
    _write_profile(live_root / "vlm_profiles", "glm-ocr", profile)
    _write_profile(factory_root / "vlm_profiles", "glm-ocr", profile)

    return live_root, factory_root


def test_has_factory_default_profile(config_tree):
    live_root, factory_root = config_tree
    assert se.has_factory_default_profile("glm-ocr", factory_root=factory_root / "vlm_profiles")
    assert not se.has_factory_default_profile("user-clone", factory_root=factory_root / "vlm_profiles")


def test_restore_model_ini_default_reverts_a_drift(config_tree):
    live_root, factory_root = config_tree
    live_ini, factory_ini = live_root / "model.ini", factory_root / "model.ini"

    live_ini.write_text(live_ini.read_text(encoding="utf-8").replace("8192", "16384"), encoding="utf-8")
    assert se.preview_model_ini_restore_diff(live_ini, factory_ini).strip()

    backup = se.restore_model_ini_default(live_ini, factory_ini)

    assert backup.exists()
    assert "16384" in backup.read_text(encoding="utf-8")  # the drifted value, preserved in the backup
    assert live_ini.read_text(encoding="utf-8") == factory_ini.read_text(encoding="utf-8")


def test_restore_model_ini_default_noop_when_already_matching(config_tree):
    live_root, factory_root = config_tree
    live_ini, factory_ini = live_root / "model.ini", factory_root / "model.ini"
    assert se.preview_model_ini_restore_diff(live_ini, factory_ini) == ""


def test_restore_vlm_profile_default_reverts_a_drift(config_tree):
    live_root, factory_root = config_tree
    live_profiles, factory_profiles = live_root / "vlm_profiles", factory_root / "vlm_profiles"

    updated = se.load_vlm_profile("glm-ocr", root=live_profiles)
    updated["prompt_mode"] = "instruction"
    se.save_vlm_profile("glm-ocr", updated, root=live_profiles)
    assert se.preview_vlm_profile_restore_diff("glm-ocr", root=live_profiles,
                                               factory_root=factory_profiles).strip()

    se.restore_vlm_profile_default("glm-ocr", root=live_profiles, factory_root=factory_profiles)

    assert se.load_vlm_profile("glm-ocr", root=live_profiles)["prompt_mode"] == "fixed"


def test_restore_vlm_profile_default_raises_for_a_profile_with_no_baseline(config_tree):
    live_root, factory_root = config_tree
    with pytest.raises(FileNotFoundError):
        se.preview_vlm_profile_restore_diff("user-clone", root=live_root / "vlm_profiles",
                                            factory_root=factory_root / "vlm_profiles")
    with pytest.raises(FileNotFoundError):
        se.restore_vlm_profile_default("user-clone", root=live_root / "vlm_profiles",
                                       factory_root=factory_root / "vlm_profiles")


def test_restore_profile_registry_default_reverts_a_changed_role(config_tree):
    live_root, factory_root = config_tree
    live_strategy, factory_strategy = live_root / "strategy.yaml", factory_root / "strategy.yaml"

    se.write_profile_registry_roles({"extraction": "deepseek-ocr-v2"}, live_strategy)
    changes = se._profile_registry_restore_changes(live_strategy, factory_strategy)
    assert changes == {"extraction": "glm-ocr"}

    se.restore_profile_registry_default(live_strategy, factory_strategy)

    assert se.read_profile_registry(live_strategy) == se.read_profile_registry(factory_strategy)
    # surgical: comments outside the changed line still present
    assert "# --- Role-based offline roster ---" in live_strategy.read_text(encoding="utf-8")


def test_restore_profile_registry_default_restores_a_role_deleted_live(config_tree):
    live_root, factory_root = config_tree
    live_strategy, factory_strategy = live_root / "strategy.yaml", factory_root / "strategy.yaml"

    se.write_profile_registry_roles({"figure_escalation": None}, live_strategy)
    assert "figure_escalation" not in se.read_profile_registry(live_strategy)

    se.restore_profile_registry_default(live_strategy, factory_strategy)

    assert se.read_profile_registry(live_strategy)["figure_escalation"] == "deepseek-ocr-v2"


def test_restore_profile_registry_default_removes_a_role_added_live(config_tree):
    live_root, factory_root = config_tree
    live_strategy, factory_strategy = live_root / "strategy.yaml", factory_root / "strategy.yaml"

    se.write_profile_registry_roles({"understanding": "olmocr-2-7b"}, live_strategy)
    assert "understanding" in se.read_profile_registry(live_strategy)

    se.restore_profile_registry_default(live_strategy, factory_strategy)

    assert "understanding" not in se.read_profile_registry(live_strategy)


def test_restore_all_defaults_reverts_everything_drifted(config_tree):
    live_root, factory_root = config_tree
    live_ini, factory_ini = live_root / "model.ini", factory_root / "model.ini"
    live_profiles, factory_profiles = live_root / "vlm_profiles", factory_root / "vlm_profiles"
    live_strategy, factory_strategy = live_root / "strategy.yaml", factory_root / "strategy.yaml"

    live_ini.write_text(live_ini.read_text(encoding="utf-8").replace("8192", "16384"), encoding="utf-8")
    updated = se.load_vlm_profile("glm-ocr", root=live_profiles)
    updated["prompt_mode"] = "instruction"
    se.save_vlm_profile("glm-ocr", updated, root=live_profiles)
    se.write_profile_registry_roles({"extraction": "deepseek-ocr-v2"}, live_strategy)

    backups = se.restore_all_defaults(
        root=live_profiles, factory_root=factory_profiles,
        model_ini_path=live_ini, factory_model_ini_path=factory_ini,
        strategy_path=live_strategy, factory_strategy_path=factory_strategy,
    )

    assert set(backups) == {"model.ini", "glm-ocr.yaml", "profile_registry"}
    restored_ini = se.read_model_ini(live_ini)
    assert se.model_ini_section_dict(restored_ini, "glm-ocr")["ctx-size"] == "8192"
    assert se.load_vlm_profile("glm-ocr", root=live_profiles)["prompt_mode"] == "fixed"
    assert se.read_profile_registry(live_strategy)["extraction"] == "glm-ocr"


def test_restore_all_defaults_is_a_noop_when_nothing_drifted(config_tree):
    live_root, factory_root = config_tree
    backups = se.restore_all_defaults(
        root=live_root / "vlm_profiles", factory_root=factory_root / "vlm_profiles",
        model_ini_path=live_root / "model.ini", factory_model_ini_path=factory_root / "model.ini",
        strategy_path=live_root / "strategy.yaml", factory_strategy_path=factory_root / "strategy.yaml",
    )
    assert backups == {}


# --- line endings (the LF -> CRLF trap) ------------------------------------
#
# Python's text mode translates "\n" to os.linesep on write, so on Windows
# every config save silently rewrote all ~129 line endings of a file the user
# had barely touched. Nothing broke (YAML/INI parse either ending), which is
# exactly why it went unnoticed: `git status` called the file modified while
# `git diff` showed nothing. These tests pin BOTH directions — Python must not
# ADD a CRLF, and must not PASS ONE THROUGH from a browser textarea (HTML
# specifies textarea contents are submitted with CRLF line breaks).


def _no_crlf(path: Path) -> bool:
    return b"\r\n" not in path.read_bytes()


def test_save_model_ini_text_writes_lf_endings(config_tree):
    live_root, _ = config_tree
    path = live_root / "model.ini"
    se.save_model_ini_text(MODEL_INI_TEXT, path=path)
    assert _no_crlf(path)


def test_save_model_ini_text_normalizes_crlf_from_a_textarea(config_tree):
    live_root, _ = config_tree
    path = live_root / "model.ini"
    se.save_model_ini_text(MODEL_INI_TEXT.replace("\n", "\r\n"), path=path)
    assert _no_crlf(path)
    # content survives the normalization intact
    assert se.model_ini_section_dict(se.read_model_ini(path), "glm-ocr")["ctx-size"] == "8192"


def test_save_vlm_profile_text_normalizes_crlf_from_a_textarea(config_tree):
    live_root, _ = config_tree
    profiles = live_root / "vlm_profiles"
    text = se.load_vlm_profile_text("glm-ocr", root=profiles).replace("\n", "\r\n")
    se.save_vlm_profile_text("glm-ocr", text, root=profiles)
    assert _no_crlf(profiles / "glm-ocr.yaml")
    assert se.load_vlm_profile("glm-ocr", root=profiles)["model_name"] == "glm-ocr"


def test_save_vlm_profile_writes_lf_endings(config_tree):
    live_root, _ = config_tree
    profiles = live_root / "vlm_profiles"
    se.save_vlm_profile("glm-ocr", {"model_name": "glm-ocr", "prompt_mode": "fixed"}, root=profiles)
    assert _no_crlf(profiles / "glm-ocr.yaml")


def test_write_model_ini_section_writes_lf_endings(config_tree):
    live_root, _ = config_tree
    path = live_root / "model.ini"
    parser = se.read_model_ini(path)
    se.write_model_ini_section(parser, "glm-ocr", {"ctx-size": "4096"}, path=path)
    assert _no_crlf(path)
    assert se.model_ini_section_dict(se.read_model_ini(path), "glm-ocr")["ctx-size"] == "4096"


def test_write_profile_registry_roles_writes_lf_endings(config_tree):
    live_root, _ = config_tree
    path = live_root / "strategy.yaml"
    se.write_profile_registry_roles({"extraction": "olmocr-2-7b"}, path)
    assert _no_crlf(path)
    assert se.read_profile_registry(path)["extraction"] == "olmocr-2-7b"


def test_write_profile_registry_roles_normalizes_a_crlf_source_file(config_tree):
    """The surgical registry edit splits the file into lines and rejoins them,
    so a file that ARRIVES as CRLF (e.g. checked out before .gitattributes
    existed) is normalized on the way out rather than left half-converted."""
    live_root, _ = config_tree
    path = live_root / "strategy.yaml"
    path.write_bytes(STRATEGY_YAML_TEXT.replace("\n", "\r\n").encode("utf-8"))
    se.write_profile_registry_roles({"extraction": "olmocr-2-7b"}, path)
    assert _no_crlf(path)
    assert se.read_profile_registry(path)["extraction"] == "olmocr-2-7b"


def test_gitattributes_pins_lf_for_the_repository():
    """The OTHER half of the fix. `_write_text` stops Python introducing CRLF;
    this stops `git checkout` doing it, which is where the 73 CRLF files in
    the working tree actually came from (Git for Windows ships a system-wide
    `core.autocrlf = true`). Deleting this file quietly reopens the problem
    from the git side, so it is asserted rather than assumed."""
    from src.processing.paths import REPO_ROOT
    text = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "* text=auto eol=lf" in text


# =============================================================================
# The citation level (citation levels, S0b): one line of the strategy YAML
# =============================================================================

@pytest.fixture
def live_strategy_copy(tmp_path: Path) -> Path:
    """A copy of the REAL strategy YAML: verified against the live file, not a fixture."""
    path = tmp_path / "research_article_strategy.yaml"
    path.write_bytes(se.STRATEGY_YAML_PATH.read_bytes())
    return path


def test_read_citation_level_reads_the_live_file(live_strategy_copy: Path):
    assert se.read_citation_level(live_strategy_copy) in ("local", "zotero")


def test_write_citation_level_changes_exactly_one_line(live_strategy_copy: Path):
    before = live_strategy_copy.read_text(encoding="utf-8").splitlines()
    current = se.read_citation_level(live_strategy_copy)
    other = "zotero" if current == "local" else "local"

    se.write_citation_level(other, live_strategy_copy)

    after = live_strategy_copy.read_text(encoding="utf-8").splitlines()
    changed = [(old, new) for old, new in zip(before, after) if old != new]
    assert len(before) == len(after)
    assert changed == [(f"  level: {current}", f"  level: {other}")]
    assert se.read_citation_level(live_strategy_copy) == other


def test_write_citation_level_keeps_lf_line_endings(live_strategy_copy: Path):
    se.write_citation_level("zotero", live_strategy_copy)
    assert b"\r\n" not in live_strategy_copy.read_bytes()


def test_write_citation_level_backs_up_first(live_strategy_copy: Path):
    original = live_strategy_copy.read_bytes()
    backup = se.write_citation_level("zotero", live_strategy_copy)
    assert backup.read_bytes() == original


def test_a_level_not_built_yet_changes_nothing_and_leaves_no_backup(live_strategy_copy: Path,
                                                                    monkeypatch):
    monkeypatch.setattr("src.processing.citations.level.BUILT_LEVELS", ("local", "zotero"))
    original = live_strategy_copy.read_bytes()
    with pytest.raises(NotImplementedError):
        se.write_citation_level("online", live_strategy_copy)
    assert live_strategy_copy.read_bytes() == original
    assert list(live_strategy_copy.parent.glob("*.bak-*")) == []


def test_a_misspelt_level_is_refused(live_strategy_copy: Path):
    with pytest.raises(ValueError):
        se.write_citation_level("everything", live_strategy_copy)


def test_preview_citation_level_diff_does_not_write(live_strategy_copy: Path):
    original = live_strategy_copy.read_bytes()
    diff = se.preview_citation_level_diff("zotero", live_strategy_copy)
    assert live_strategy_copy.read_bytes() == original
    assert "level: zotero" in diff or diff == ""


def test_an_older_file_with_no_citations_block_gets_one(strategy_yaml: Path):
    """A config written before the setting existed; its comments must survive."""
    before = strategy_yaml.read_text(encoding="utf-8")
    assert se.read_citation_level(strategy_yaml) == "local"       # nothing said -> local

    se.write_citation_level("zotero", strategy_yaml)

    after = strategy_yaml.read_text(encoding="utf-8")
    assert after.startswith(before)
    assert after.endswith("citations:\n  level: zotero\n")
    assert se.read_citation_level(strategy_yaml) == "zotero"
