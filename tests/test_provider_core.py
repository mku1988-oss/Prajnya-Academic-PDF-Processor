#!/usr/bin/env python3
"""
Tests for the pure layer of the VLM provider: schemas, prompts, parsing, quality.

For biology researchers: none of this needs a GPU, a model server, or even a
PDF. That is the reason for splitting these pieces out of V5's single
1654-line provider class — in V5 you could not check any of this logic without
standing up a model. Here it is all plain functions over plain data.

Where a number or a shape is asserted below, it was **measured by running V5's
code directly** and pinned here, so the port is verified against the real
baseline rather than against my reading of it. Two places where V6 knowingly
differs are marked and explained.
"""

import pytest

from src.llm.providers.output_parser import post_process_output
from src.llm.providers.prompts import (
    build_message_content,
    lookup_fixed_prompt,
    resolve_prompt_text,
    task_identifier,
)
from src.llm.providers.quality import (
    calculate_page_quality_score,
    title_page_metadata_is_complete,
)
from src.llm.providers.schemas import (
    APIParameters,
    ModelParameters,
    ModelProfile,
    TaskDefinition,
    UniversalConfig,
)

JSON_TASK = TaskDefinition(task_name="t", output_format={"type": "json"})
TEXT_TASK = TaskDefinition(task_name="t", output_format={"type": "markdown"})


def instruction_model(**kwargs) -> ModelProfile:
    """An instruction-following profile (OlmOCR/Gemini family)."""
    return ModelProfile(model_name="olmocr-2-7b", model_type="vlm", **kwargs)


def recognition_model(**kwargs) -> ModelProfile:
    """A recognition-only profile (GLM family) — fixed prompts, no instructions."""
    kwargs.setdefault("fixed_prompts", {"raw_ocr_extraction": "Text Recognition:"})
    kwargs.setdefault("ineligible_tasks", ["figure_description"])
    return ModelProfile(model_name="glm-ocr", model_type="ocr",
                        prompt_mode="fixed", **kwargs)


# ---------------------------------------------------------------------------
# schemas
# ---------------------------------------------------------------------------

def test_api_parameters_are_preferred_over_the_legacy_field():
    profile = ModelProfile(
        model_name="m", model_type="t",
        api_parameters=APIParameters(temperature=0.1, max_tokens=4096),
        parameters=ModelParameters(temperature=0.9),
    )
    assert profile.get_api_params() == {"temperature": 0.1, "max_tokens": 4096}


def test_legacy_parameters_still_load_when_there_is_no_new_field():
    profile = ModelProfile(model_name="m", model_type="t",
                           parameters=ModelParameters(temperature=0.9, top_p=0.8))
    assert profile.get_api_params() == {"temperature": 0.9, "top_p": 0.8}


def test_a_profile_with_no_parameters_sends_none():
    assert ModelProfile(model_name="m", model_type="t").get_api_params() == {}


def test_unset_parameters_are_omitted_rather_than_sent_as_null():
    """An absent field means "server default", which is not the same as null."""
    profile = ModelProfile(model_name="m", model_type="t",
                           api_parameters=APIParameters(temperature=0.0))
    assert profile.get_api_params() == {"temperature": 0.0}


def test_profiles_keep_unknown_fields():
    """
    Model-specific notes must survive, or annotating a YAML would mean editing
    Python — the coupling the VLM-agnostic rule exists to prevent.
    """
    profile = ModelProfile(model_name="m", model_type="t",
                           biology_optimizations={"stain": "H&E"}, benchmark_notes="fast")
    assert profile.biology_optimizations == {"stain": "H&E"}


def test_out_of_range_parameters_are_rejected():
    """The point of validating at all: catch a typo before the GPU run, not during."""
    with pytest.raises(ValueError):
        APIParameters(temperature=5.0)
    with pytest.raises(ValueError):
        APIParameters(max_tokens=0)


def test_is_fixed_prompt_reads_the_profile():
    assert recognition_model().is_fixed_prompt()
    assert not instruction_model().is_fixed_prompt()


def test_api_endpoint_keeps_a_stable_string_type():
    """
    V5 declared this `HttpUrl`, which yields `str` when it comes from the
    default but a non-str `HttpUrl` when it comes from a config file — which is
    why V5 wraps the field in `str(...)` at four call sites. Measured on
    pydantic 2.11. A plain `str` is the same value with one type.
    """
    from_default = UniversalConfig().api_endpoint
    from_config = UniversalConfig(api_endpoint="http://localhost:9931/v1/chat/completions").api_endpoint

    assert isinstance(from_default, str) and isinstance(from_config, str)
    assert from_default == from_config == "http://localhost:9931/v1/chat/completions"


# ---------------------------------------------------------------------------
# prompts — the recognition vs instruction seam
# ---------------------------------------------------------------------------

def test_an_instruction_model_gets_the_tasks_own_prompt():
    task = TaskDefinition(task_name="t", prompt="Extract the DOI and authors as JSON.")
    assert resolve_prompt_text(instruction_model(), task) == "Extract the DOI and authors as JSON."


def test_base_text_anchoring_fills_the_placeholder():
    task = TaskDefinition(task_name="t", prompt="Use this text: {base_text}")
    assert resolve_prompt_text(instruction_model(), task, base_text="PAGE WORDS") == \
        "Use this text: PAGE WORDS"


def test_a_prompt_full_of_json_braces_survives_anchoring():
    """
    The trap that kept V5's `{raw_text}` typo alive.

    A prompt asking for structured output shows the model an example of that
    structure, so it is full of literal JSON braces. Substituting with
    `str.format` treats each of them as a field and raises
    KeyError: '\\n  "metadata"' — so correcting the placeholder made the call
    crash, and the broken name stayed. Substitution must touch the
    placeholder and nothing else.
    """
    task = TaskDefinition(
        task_name="t",
        prompt='Return {\n  "metadata": {"title": "..."}\n}\nTEXT:\n{base_text}')
    prompt = resolve_prompt_text(instruction_model(), task, base_text="PAGE WORDS")

    assert prompt.endswith("TEXT:\nPAGE WORDS")
    assert '"metadata": {"title": "..."}' in prompt   # the example is untouched


def test_base_text_is_ignored_when_the_prompt_does_not_ask_for_it():
    task = TaskDefinition(task_name="t", prompt="Just read the page.")
    assert resolve_prompt_text(instruction_model(), task, base_text="PAGE WORDS") == \
        "Just read the page."


def test_a_recognition_model_gets_its_own_fixed_prompt_not_ours():
    """
    The substitution that makes recognition models usable. Passing our rich
    instruction through would return transcription dressed up as an answer.
    """
    task = TaskDefinition(task_name="raw_ocr_extraction",
                          prompt="Extract the DOI and authors as JSON.")
    resolved = resolve_prompt_text(recognition_model(), task, task_type="raw_ocr_extraction")
    assert resolved == "Text Recognition:"


def test_a_recognition_model_refuses_a_task_it_cannot_do():
    """
    THE guard. A recognition model answers every prompt with transcription, so
    an impossible task returns plausible-looking text rather than an error.
    Accepting that would put unflagged, wrong content into a literature review.
    """
    task = TaskDefinition(task_name="figure_description")
    with pytest.raises(ValueError, match="requires an instruction-following"):
        resolve_prompt_text(recognition_model(), task, task_type="figure_description")


def test_a_recognition_model_refuses_a_task_with_no_fixed_prompt():
    task = TaskDefinition(task_name="some_new_task")
    with pytest.raises(ValueError, match="No fixed prompt defined") as error:
        lookup_fixed_prompt(recognition_model(), task, task_type="some_new_task")
    # The message must say how to fix it, both ways.
    assert "fixed_prompts" in str(error.value) and "ineligible_tasks" in str(error.value)


def test_the_provider_task_type_wins_when_looking_up_a_fixed_prompt():
    """Profiles write `fixed_prompts` against the task YAML's filename."""
    task = TaskDefinition(task_name="name_field", task_type="type_field")
    assert task_identifier(task, task_type="provider_value") == "provider_value"
    assert task_identifier(task) == "type_field"
    assert task_identifier(TaskDefinition(task_name="name_field")) == "name_field"


def test_a_recognition_model_receives_the_image_first():
    """
    Not cosmetic: on identical input these models return different output
    depending on ordering, and they were validated image-first.
    """
    content = build_message_content("prompt", "BASE64", model_profile=recognition_model())
    assert [part["type"] for part in content] == ["image_url", "text"]


def test_an_instruction_model_receives_the_text_first():
    """The original order, which existing profiles were tuned against."""
    content = build_message_content("prompt", "BASE64", model_profile=instruction_model())
    assert [part["type"] for part in content] == ["text", "image_url"]


def test_ordering_can_be_forced_by_the_caller():
    """A grounding read overrides this regardless of model family."""
    forced = build_message_content("p", "B", model_profile=instruction_model(), image_first=True)
    assert forced[0]["type"] == "image_url"


def test_the_image_is_sent_as_a_data_url_in_the_declared_format():
    content = build_message_content("p", "BASE64DATA", image_format="jpeg", image_first=True)
    assert content[0]["image_url"]["url"] == "data:image/jpeg;base64,BASE64DATA"


# ---------------------------------------------------------------------------
# output_parser — values measured from V5
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected_content,expected_metadata", [
    ('{"natural_text":"hello","is_table":true}', "hello", {"is_table": True}),
    ('```json\n{"natural_text":"hi","doi":"10.1/x"}\n```', "hi", {"doi": "10.1/x"}),
    ('```\n{"natural_text":"pf"}\n```', "pf", {}),
    ('{"natural_text":"x","is_table":True,"lang":None,"ok":False}', "x",
     {"is_table": True, "lang": None, "ok": False}),
    ('{"metadata":{"doi":"10.1/y"},"natural_text":"outer text"}', "outer text", {"doi": "10.1/y"}),
    ('{"metadata":{"doi":"10.1/z","natural_text":"inner"},"natural_text":"outer"}', "inner",
     {"doi": "10.1/z"}),
    ('{"doi":"10.1/q"}', "", {"doi": "10.1/q"}),
    ("not json at all", "not json at all", {}),
])
def test_parsing_matches_v5_exactly(raw, expected_content, expected_metadata):
    """Each expectation was produced by running V5's `_post_process_output`."""
    result = post_process_output(raw, JSON_TASK, task_type="t")
    assert result["content"] == expected_content
    assert result["metadata"] == expected_metadata


def test_the_raw_output_is_always_preserved():
    """On this slow a pipeline, re-running a page to see what the model said is expensive."""
    raw = "unparseable {{{"
    assert post_process_output(raw, JSON_TASK)["raw_output"] == raw


def test_a_text_task_skips_json_parsing_entirely():
    """
    The input here is deliberately VALID JSON with a `natural_text` key, so the
    two code paths give different answers: parsing would yield "parsed me",
    while the text path returns the document verbatim. An input that merely
    fails to parse would pass either way and prove nothing.
    """
    raw = '{"natural_text":"parsed me","doi":"10.1/x"}'
    result = post_process_output(raw, TEXT_TASK)
    assert result["content"] == raw, "a markdown task must return its text untouched"
    assert result["metadata"] == {}


def test_a_json_task_parses_the_same_input_that_a_text_task_passes_through():
    """The other half of the pair above — same bytes, different task, different result."""
    raw = '{"natural_text":"parsed me","doi":"10.1/x"}'
    assert post_process_output(raw, JSON_TASK)["content"] == "parsed me"


@pytest.mark.parametrize("raw", ["[1,2,3]", '"just a string"', "42"])
def test_valid_json_of_the_wrong_shape_does_not_crash(raw):
    """
    DELIBERATE DEVIATION FROM V5. V5 raised `AttributeError: 'list' object has
    no attribute 'get'` here, taking down the page. A bare list or string is
    valid JSON with no metadata in it, so the text is passed through instead.
    """
    result = post_process_output(raw, JSON_TASK)
    assert result["content"] == raw
    assert result["metadata"] == {}


def test_python_literals_outside_strings_are_repaired():
    result = post_process_output('{"natural_text":"x","flag":True}', JSON_TASK)
    assert result["metadata"] == {"flag": True}


def test_the_word_none_in_the_papers_own_prose_is_not_corrupted():
    """
    DELIBERATE DEVIATION FROM V5, and the reason this module was worth
    extracting. V5 ran three blind regex substitutions over the whole document,
    so a paper reading "None of the samples showed activity" was rewritten to
    "null of the samples showed activity". The page's text travels in this same
    JSON under `natural_text`, so this silently corrupted extracted scientific
    content — inside the path meant to REPAIR output. Measured against V5:
    it returns "null of the samples...".
    """
    raw = '{"natural_text":"None of the samples showed activity.","is_table":False}'
    result = post_process_output(raw, JSON_TASK)
    assert result["content"] == "None of the samples showed activity."
    assert result["metadata"] == {"is_table": False}


def test_true_and_false_in_prose_are_also_left_alone():
    raw = '{"natural_text":"True positives were counted. False negatives were rare."}'
    assert post_process_output(raw, JSON_TASK)["content"] == \
        "True positives were counted. False negatives were rare."


def test_a_literal_embedded_in_a_longer_word_is_untouched():
    """V5 got this one right via word boundaries; keep it right."""
    result = post_process_output('{"natural_text":"x","colour":"TrueBlue"}', JSON_TASK)
    assert result["metadata"] == {"colour": "TrueBlue"}


def test_escaped_quotes_inside_a_string_do_not_break_the_scan():
    raw = r'{"natural_text":"He said \"None\" clearly.","flag":True}'
    result = post_process_output(raw, JSON_TASK)
    assert result["content"] == 'He said "None" clearly.'
    assert result["metadata"] == {"flag": True}


# ---------------------------------------------------------------------------
# quality — scores measured from V5
# ---------------------------------------------------------------------------

TITLE_METADATA = {"title": "A Real Paper Title", "authors": ["A. One", "B. Two"],
                  "doi": "10.1016/j.x.2005.09.009"}
BODY_METADATA = {"primary_language": "en", "is_rotation_valid": True,
                 "rotation_correction": 0, "is_table": False, "is_diagram": False}


@pytest.mark.parametrize("name,task_type,content,metadata,expected", [
    ("title-good", "structured_analysis_title_page", "# Paper\n\nSome prose. " * 40,
     TITLE_METADATA, 0.725800),
    ("title-empty", "structured_analysis_title_page", "", {}, 0.0),
    ("body-good", "structured_analysis_body_page",
     "## Results\n\n* item\n" + "Prose text here. " * 100, BODY_METADATA, 0.881280),
    ("body-empty", "raw_ocr_extraction", "", {}, 0.0),
    ("body-flags-false", "raw_ocr_extraction", "Text. " * 300,
     {**BODY_METADATA, "is_rotation_valid": False}, 0.816000),
])
def test_quality_scores_match_v5_exactly(name, task_type, content, metadata, expected):
    """Each expectation was produced by running V5's `_calculate_page_quality_score`."""
    assert calculate_page_quality_score(content, metadata, task_type=task_type) == \
        pytest.approx(expected, abs=1e-6)


def test_scoring_is_task_aware():
    """
    A body page scored against title-page expectations would look terrible
    despite being a perfect read. That is what "task-aware" is guarding.
    """
    content = "## Results\n\n* item\n" + "Prose text here. " * 100
    as_body = calculate_page_quality_score(content, BODY_METADATA,
                                           task_type="structured_analysis_body_page")
    as_title = calculate_page_quality_score(content, BODY_METADATA,
                                            task_type="structured_analysis_title_page")
    assert as_body > as_title


def test_body_flags_that_are_legitimately_false_still_count_as_present():
    """
    `is_table: False` is an answer, not a missing field — which is why body
    presence tests `is not None` while title presence tests truthiness.
    """
    all_false = {"primary_language": "en", "is_rotation_valid": False,
                 "rotation_correction": 0, "is_table": False, "is_diagram": False}
    assert calculate_page_quality_score("Text. " * 300, all_false,
                                        task_type="raw_ocr_extraction") > 0.8


def test_an_empty_page_scores_zero():
    assert calculate_page_quality_score("", {}, task_type="raw_ocr_extraction") == 0.0


def test_the_score_never_leaves_the_unit_interval():
    """Weights sum to 1.0 and every component is capped, so this must hold."""
    enormous = "# H\n## H2\n* a\n- b\n**bold**\n" + "Long prose sentence here. " * 2000
    score = calculate_page_quality_score(enormous, TITLE_METADATA,
                                         task_type="structured_analysis_title_page")
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# The title-page completeness gate
# ---------------------------------------------------------------------------

def test_complete_title_metadata_passes_the_gate():
    assert title_page_metadata_is_complete(TITLE_METADATA)


@pytest.mark.parametrize("broken,why", [
    ({**TITLE_METADATA, "title": "Short"}, "title of five characters or fewer"),
    ({**TITLE_METADATA, "authors": []}, "no authors"),
    ({**TITLE_METADATA, "doi": "not-a-doi"}, "DOI without a 10. prefix"),
    ({**TITLE_METADATA, "authors": "A. One"}, "authors as a string, not a list"),
    ({}, "nothing at all"),
])
def test_incomplete_title_metadata_fails_the_gate(broken, why):
    """
    Each failure sends the metadata ladder down to its next rung, so these
    checks decide how much VLM time a title page costs.
    """
    assert not title_page_metadata_is_complete(broken), why
