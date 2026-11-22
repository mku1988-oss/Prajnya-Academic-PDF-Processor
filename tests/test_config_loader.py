#!/usr/bin/env python3
"""
Tests for config loading: YAML, environment interpolation, registry references.

For biology researchers: this is where a wrong answer is quietest. A bad model
name raises an error you can see; a credential that silently resolves to the
literal text "${ZOTERO_API_KEY}" produces a pipeline that runs to completion
and returns a paper with worse metadata. That exact failure existed in V5, so
several tests below exist specifically to pin the fix.

Everything here runs on temporary files and a patched environment — no network,
no real config tree.
"""

from pathlib import Path

import pytest
import yaml

from src.processing.config import (
    ConfigLoader,
    interpolate_env,
    load_config,
    resolve_reference,
    resolve_references,
)

#: A registry shaped like V5's, so reference resolution is tested against the
#: real structure rather than a toy one.
REGISTRY_CONFIG = {
    "profile_registry": {
        "default": "olmocr-2-7b",
        "extraction": "glm-ocr",
        "layout_detector": "deepseek-ocr-v2",
    },
    "task_registry": {
        "raw_ocr": "raw_ocr_extraction",
        "structured_title": "structured_analysis_title_page",
    },
}


def write_yaml(path: Path, data) -> Path:
    """Write a config file, accepting either a dict or literal YAML text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = data if isinstance(data, str) else yaml.safe_dump(data)
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Environment interpolation — the V5 bug this module exists to fix
# ---------------------------------------------------------------------------

def test_a_set_variable_is_substituted(monkeypatch):
    monkeypatch.setenv("ZOTERO_API_KEY", "secret-key-value")
    assert interpolate_env("${ZOTERO_API_KEY}") == "secret-key-value"


def test_an_unset_variable_resolves_to_none_not_the_literal_text():
    """
    The V5 trap, closed.

    V5 never interpolated, so `api_key: ${ZOTERO_API_KEY}` sat in the config as
    that literal string. Nothing broke, because V5 loads `.env` at startup and
    every consumer reads the environment itself first
    (`os.getenv(...) or config.get(...)`) — the literal was dead config.

    But that safety depended on each consumer double-reading the variable. A
    config key whose consumer lacks that fallback would receive the literal,
    and a truthy literal walks past an `if not value:` guard. Resolving to None
    makes the guard fire instead.
    """
    resolved = interpolate_env("${DEFINITELY_NOT_SET_ANYWHERE}")
    assert resolved is None
    assert resolved != "${DEFINITELY_NOT_SET_ANYWHERE}"
    assert not resolved, "must be falsy, or downstream `if not key:` guards stay asleep"


def test_an_unset_variable_can_declare_a_fallback():
    assert interpolate_env("${NOT_SET_EITHER:-persistence/ledger.csv}") == "persistence/ledger.csv"


def test_a_set_variable_wins_over_its_fallback(monkeypatch):
    monkeypatch.setenv("PRAJNYA_LEDGER_PATH", "D:/runs/ledger.csv")
    assert interpolate_env("${PRAJNYA_LEDGER_PATH:-persistence/ledger.csv}") == "D:/runs/ledger.csv"


def test_an_empty_fallback_is_respected():
    """`${VAR:-}` means "empty string", which is different from None."""
    assert interpolate_env("${STILL_NOT_SET:-}") == ""


def test_a_placeholder_embedded_in_text_is_substituted_in_place(monkeypatch):
    monkeypatch.setenv("PRAJNYA_HOST", "localhost")
    assert interpolate_env("http://${PRAJNYA_HOST}:8080/v1") == "http://localhost:8080/v1"


def test_an_unset_embedded_placeholder_contributes_nothing(monkeypatch):
    monkeypatch.delenv("PRAJNYA_HOST", raising=False)
    assert interpolate_env("http://${PRAJNYA_HOST}:8080") == "http://:8080"


def test_interpolation_reaches_nested_dicts_and_lists(monkeypatch):
    monkeypatch.setenv("PRAJNYA_MODEL", "glm-ocr")
    config = {
        "vlm_provider": {"models": ["${PRAJNYA_MODEL}", "olmocr-2-7b"]},
        "zotero": {"api_key": "${NOT_SET_AT_ALL}"},
    }
    result = interpolate_env(config)
    assert result["vlm_provider"]["models"] == ["glm-ocr", "olmocr-2-7b"]
    assert result["zotero"]["api_key"] is None


def test_non_string_values_pass_through_untouched():
    """Ports, booleans, and None must not be stringified on the way past."""
    config = {"port": 8080, "enabled": True, "timeout": 2.5, "unset": None}
    assert interpolate_env(config) == config


def test_ordinary_strings_are_left_alone():
    assert interpolate_env("http://localhost:8080/v1") == "http://localhost:8080/v1"
    assert interpolate_env("$NOT_BRACED") == "$NOT_BRACED"
    assert interpolate_env("costs ${5.00}") == "costs ${5.00}"


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

def test_load_config_reads_yaml_and_interpolates(tmp_path, monkeypatch):
    monkeypatch.setenv("ZOTERO_LIBRARY_ID", "123456")
    path = write_yaml(tmp_path / "strategy.yaml", {
        "strategy_name": "research_article",
        "zotero": {"library_id": "${ZOTERO_LIBRARY_ID}", "enabled": True},
    })
    config = load_config(path)
    assert config["strategy_name"] == "research_article"
    assert config["zotero"]["library_id"] == "123456"
    assert config["zotero"]["enabled"] is True


def test_load_config_reads_utf8_regardless_of_platform_default(tmp_path):
    """
    Windows defaults to cp1252, which raises on the first accented author name
    or typographic dash in a prompt. The encoding is pinned explicitly.
    """
    path = write_yaml(tmp_path / "profile.yaml", {
        "authors": ["Müller", "Ferré", "Škrlj"],
        "prompt": "Read the page — all of it.",
    })
    config = load_config(path)
    assert config["authors"] == ["Müller", "Ferré", "Škrlj"]
    assert "—" in config["prompt"]


def test_a_missing_config_file_raises_rather_than_falling_back(tmp_path):
    """
    V5 answered a missing strategy config with a hardcoded fallback, which hid
    genuine path bugs behind a silently different pipeline — the worst outcome
    for a benchmark whose gates are numeric.
    """
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "absent.yaml")


def test_malformed_yaml_raises(tmp_path):
    path = write_yaml(tmp_path / "broken.yaml", "key: [unclosed\n")
    with pytest.raises(yaml.YAMLError):
        load_config(path)


def test_an_empty_config_file_is_an_empty_mapping(tmp_path):
    assert load_config(write_yaml(tmp_path / "empty.yaml", "")) == {}


def test_a_yaml_list_at_the_root_is_rejected(tmp_path):
    path = write_yaml(tmp_path / "list.yaml", "- one\n- two\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_config(path)


# ---------------------------------------------------------------------------
# resolve_reference — ported from V5 unchanged
# ---------------------------------------------------------------------------

def test_a_registry_reference_resolves():
    assert resolve_reference("@profile_registry.default", REGISTRY_CONFIG) == "olmocr-2-7b"
    assert resolve_reference("@task_registry.raw_ocr", REGISTRY_CONFIG) == "raw_ocr_extraction"


def test_role_indirection_keeps_model_names_out_of_the_code():
    """
    The VLM-agnostic rule in practice: swapping the model behind a ROLE is one
    registry edit. The code asks for `extraction`, never for "glm-ocr".
    """
    assert resolve_reference("@profile_registry.extraction", REGISTRY_CONFIG) == "glm-ocr"

    swapped = {**REGISTRY_CONFIG,
               "profile_registry": {**REGISTRY_CONFIG["profile_registry"],
                                    "extraction": "deepseek-ocr-v2"}}
    assert resolve_reference("@profile_registry.extraction", swapped) == "deepseek-ocr-v2"


@pytest.mark.parametrize("value", ["literal-value", "olmocr-2-7b", 123, 4.5, True, None, ["a"]])
def test_non_references_pass_through_untouched(value):
    """Safe to call on any config value, which is why call sites stay clean."""
    assert resolve_reference(value, REGISTRY_CONFIG) == value


def test_a_reference_without_a_key_is_rejected():
    with pytest.raises(ValueError, match="Expected format"):
        resolve_reference("@profile_registry", REGISTRY_CONFIG)


def test_a_missing_key_names_the_available_ones():
    """The error has to be actionable — a typo should show the real options."""
    with pytest.raises(ValueError, match="Available keys") as error:
        resolve_reference("@profile_registry.defualt", REGISTRY_CONFIG)
    assert "default" in str(error.value)


def test_a_missing_registry_is_rejected():
    with pytest.raises(ValueError, match="not found"):
        resolve_reference("@no_such_registry.key", REGISTRY_CONFIG)


def test_a_reference_through_a_scalar_is_rejected():
    with pytest.raises(ValueError, match="not a mapping"):
        resolve_reference("@profile_registry.default.deeper", REGISTRY_CONFIG)


def test_registry_values_may_not_themselves_be_references():
    """
    Deliberate: chaining would turn a typo into an infinite loop instead of an
    error. V5 refused recursion and so does this.
    """
    chained = {"profile_registry": {"default": "@profile_registry.fallback",
                                    "fallback": "olmocr-2-7b"}}
    with pytest.raises(ValueError, match="cannot themselves be references"):
        resolve_reference("@profile_registry.default", chained)


# ---------------------------------------------------------------------------
# resolve_references — whole-subtree resolution
# ---------------------------------------------------------------------------

def test_a_whole_section_resolves_at_once():
    section = {
        "title_page": {
            "profile": "@profile_registry.default",
            "shot1_task": "@task_registry.raw_ocr",
            "resolution_override": 1288,
        },
        "detectors": ["@profile_registry.layout_detector", "literal"],
    }
    resolved = resolve_references(section, REGISTRY_CONFIG)
    assert resolved["title_page"]["profile"] == "olmocr-2-7b"
    assert resolved["title_page"]["shot1_task"] == "raw_ocr_extraction"
    assert resolved["title_page"]["resolution_override"] == 1288
    assert resolved["detectors"] == ["deepseek-ocr-v2", "literal"]


def test_a_typo_in_an_untaken_branch_surfaces_immediately():
    """
    V5 resolved one value at a time at each call site, so a bad reference in a
    rarely-taken branch stayed invisible until that branch ran — possibly
    mid-document. Resolving the section up front fails fast instead.
    """
    section = {"common": "@profile_registry.default",
               "rare_branch": {"profile": "@profile_registry.typo"}}
    with pytest.raises(ValueError, match="typo"):
        resolve_references(section, REGISTRY_CONFIG)


# ---------------------------------------------------------------------------
# ConfigLoader
# ---------------------------------------------------------------------------

@pytest.fixture
def config_root(tmp_path: Path) -> Path:
    """A config tree shaped like the one the strategy will ship with."""
    root = tmp_path / "config"
    write_yaml(root / "research_article_strategy.yaml", {
        "strategy_name": "research_article",
        "profile_registry": {"extraction": "glm-ocr"},
    })
    write_yaml(root / "vlm_profiles" / "glm-ocr.yaml",
               {"model_name": "glm-ocr", "default_resolution": 1024})
    write_yaml(root / "vlm_profiles" / "olmocr-2-7b.yaml", {"model_name": "olmocr-2-7b"})
    write_yaml(root / "task_definitions" / "raw_ocr_extraction.yaml",
               {"task_name": "raw_ocr_extraction"})
    return root


def test_loader_reads_each_kind_of_config(config_root):
    loader = ConfigLoader(config_root)
    assert loader.strategy()["strategy_name"] == "research_article"
    assert loader.vlm_profile("glm-ocr")["default_resolution"] == 1024
    assert loader.task_definition("raw_ocr_extraction")["task_name"] == "raw_ocr_extraction"


def test_the_config_root_is_explicit_not_derived_from_module_location(config_root, tmp_path):
    """
    V5 derived every config path from `Path(__file__)` of the loading module,
    so moving a module — exactly what plan.md's decomposition does — silently
    broke its lookups. plan.md flags this as a Stage 4 risk; an explicit root
    removes it, and two loaders can point at different trees.
    """
    relocated = tmp_path / "elsewhere"
    write_yaml(relocated / "research_article_strategy.yaml", {"strategy_name": "relocated"})

    assert ConfigLoader(config_root).strategy()["strategy_name"] == "research_article"
    assert ConfigLoader(relocated).strategy()["strategy_name"] == "relocated"


def test_a_missing_profile_lists_what_is_available(config_root):
    """A typo should print the real profile names, not just a path."""
    with pytest.raises(FileNotFoundError) as error:
        ConfigLoader(config_root).vlm_profile("glm-ocrr")
    message = str(error.value)
    assert "glm-ocr" in message and "olmocr-2-7b" in message
    assert "VLM profile" in message


def test_a_missing_task_definition_directory_is_reported_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="directory does not exist"):
        ConfigLoader(tmp_path).task_definition("raw_ocr_extraction")


def test_loader_interpolates_the_environment_too(config_root, monkeypatch):
    """The fix must apply to every config file, not only the strategy one."""
    monkeypatch.setenv("PRAJNYA_ENDPOINT", "http://localhost:8080/v1")
    write_yaml(config_root / "vlm_profiles" / "remote.yaml",
               {"model_name": "remote", "api_endpoint": "${PRAJNYA_ENDPOINT}",
                "api_key": "${NEVER_SET_IN_TESTS}"})

    profile = ConfigLoader(config_root).vlm_profile("remote")
    assert profile["api_endpoint"] == "http://localhost:8080/v1"
    assert profile["api_key"] is None
