#!/usr/bin/env python3
"""
Tests for the ported V6 config tree (config/).

These are different from test_config_loader.py, which tests the *loader
mechanics* against synthetic fixtures. This file tests the *real shipped
config*: that the strategy YAML loads, that every registry entry points at a
file that actually exists, and that the title-page task is still the
ported-as-is V5 version (the Stage 0 prompt fix is a deliberate, separate
change — see plan.md).

For biology researchers: the config tree is data, but broken data fails at
runtime just like broken code. A registry entry naming a profile file that was
never copied would only surface mid-run; these tests surface it in the suite.
"""

from pathlib import Path

import pytest

from src.processing.config.loader import ConfigLoader, resolve_reference

CONFIG_ROOT = Path(__file__).parent.parent / "config"


@pytest.fixture(scope="module")
def loader() -> ConfigLoader:
    return ConfigLoader(CONFIG_ROOT)


@pytest.fixture(scope="module")
def strategy(loader) -> dict:
    return loader.strategy()


class TestStrategyConfig:
    def test_loads_and_identifies_itself(self, strategy):
        assert strategy["strategy_name"] == "research_article"
        assert strategy["version"] == "2.0"

    def test_delete_list_is_really_deleted(self, strategy):
        """The 5-tier machinery and the concurrency block must not come back."""
        body = strategy["vlm_provider"]["body_pages"]
        for relic in ("tier1_pymupdf", "tier2_surgical_vlm", "tier3_single_shot",
                      "tier4_two_shot", "tier5_fallback", "adaptive_routing_rules"):
            assert relic not in body, f"V5 relic '{relic}' reappeared in body_pages"
        assert "concurrency" not in strategy["vlm_provider"]

    def test_glm_twophase_roster_present(self, strategy):
        registry = strategy["profile_registry"]
        assert registry["extraction"] == "glm-ocr"
        assert registry["understanding"] == "olmocr-2-7b"
        assert registry["layout_detector"] == "onnx"
        assert strategy["vlm_provider"]["local_server"]["enabled"] is True

    def test_title_page_references_resolve(self, strategy):
        title = strategy["vlm_provider"]["title_page"]
        assert resolve_reference(title["profile"], strategy) == "olmocr-2-7b"
        assert resolve_reference(title["shot1_task"], strategy) == "raw_ocr_extraction"
        assert resolve_reference(title["shot2_task"], strategy) == "structured_analysis_title_page"
        assert title["resolution_override"] == 1288

    def test_zotero_keys_are_interpolated_not_literal(self, strategy):
        """${ZOTERO_API_KEY} must never survive as a literal string (the V5 trap)."""
        for key in ("api_key", "library_id"):
            value = strategy["zotero"][key]
            assert value is None or "${" not in str(value)


class TestRegistryFilesExist:
    def test_every_task_registry_entry_has_a_file(self, loader, strategy):
        for role, task_name in strategy["task_registry"].items():
            definition = loader.task_definition(task_name)  # raises if missing
            assert definition.get("prompt") or definition.get("fixed_prompts"), \
                f"task '{task_name}' (role '{role}') has no prompt"

    def test_every_model_profile_entry_has_a_file(self, loader, strategy):
        for role, profile_name in strategy["profile_registry"].items():
            if profile_name == "onnx":
                continue  # the CPU detector is code, not a VLM profile
            profile = loader.vlm_profile(profile_name)  # raises if missing
            assert profile, f"profile '{profile_name}' (role '{role}') is empty"

    def test_glm_is_fixed_prompt(self, loader):
        """GLM is recognition-style: its profile must carry its own prompts."""
        glm = loader.vlm_profile("glm-ocr")
        assert glm.get("prompt_mode") == "fixed"
        assert isinstance(glm.get("fixed_prompts"), dict) and glm["fixed_prompts"]


class TestStage0Fixes:
    """
    Stage 0 fixes, each replacing a pinned ported-as-is defect.

    These tests used to assert the *bug*, so that fixing it would show up as a
    deliberate test change rather than unnoticed drift. Each flip below was
    made with the parity harness green before and after.
    """

    def test_no_shipped_prompt_asks_for_a_placeholder_nothing_substitutes(self):
        """
        The generalized guard — the defect class, not the one file.

        `{base_text}` is the only name `resolve_prompt_text` fills in. Any
        other `{placeholder}` in a prompt reaches the model as a literal
        string, usually under a heading promising real content, and the model
        quietly works around it. That is exactly how V5's `{raw_text}` went
        unnoticed for a whole pipeline generation.

        Braces inside an example JSON structure are legitimate, so only
        lowercase snake_case placeholder shapes are checked.
        """
        import re

        placeholder = re.compile(r'\{[a-z][a-z0-9_]*\}')
        offenders = {}

        for path in sorted((CONFIG_ROOT / "task_definitions").glob("*.yaml")):
            prompt = (ConfigLoader(CONFIG_ROOT)
                      .task_definition(path.stem).get("prompt") or "")
            unknown = {name for name in placeholder.findall(prompt)
                       if name != "{base_text}"}
            if unknown:
                offenders[path.name] = sorted(unknown)

        assert not offenders, (
            "these prompts ask for placeholders the provider never fills: "
            f"{offenders}")

    def test_the_title_task_uses_the_placeholder_the_provider_substitutes(self):
        """
        Shot 2's prompt must say {base_text}, not V5's {raw_text}.

        `{base_text}` is the ONLY name `prompts.resolve_prompt_text`
        substitutes. With V5's `{raw_text}` nothing was replaced, so shot 2
        received the literal string "{raw_text}" beneath a heading promising
        the transcription — and had to re-read the metadata from pixels while
        the transcription it had just produced went unused.
        """
        text = (CONFIG_ROOT / "task_definitions" /
                "structured_analysis_title_page.yaml").read_text(encoding="utf-8")
        assert "{base_text}" in text
        # Only the explanatory comment may still mention the old name.
        prompt_body = text.split("prompt: |", 1)[1]
        assert "{raw_text}" not in prompt_body

    def test_the_placeholder_is_actually_interpolated(self):
        """
        The end-to-end check the YAML assertion alone cannot make.

        A placeholder is only useful if the provider fills it, so this drives
        the real profile and the real task through `resolve_prompt_text` and
        asserts the transcription arrives in the prompt.
        """
        from src.llm.providers.prompts import resolve_prompt_text
        from src.llm.providers.schemas import ModelProfile, TaskDefinition

        loader = ConfigLoader(CONFIG_ROOT)
        profile = ModelProfile(**loader.vlm_profile("olmocr-2-7b"))
        task = TaskDefinition(**loader.task_definition("structured_analysis_title_page"))

        transcription = "Gary D. Fullerton, Ivan Cameron — Cell Biology International"
        prompt = resolve_prompt_text(profile, task, base_text=transcription)

        assert transcription in prompt
        assert "{base_text}" not in prompt
        assert "{raw_text}" not in prompt
