#!/usr/bin/env python3
"""
Tests for the transport layer and the provider that sequences it.

For biology researchers: these check how a request is *built* and what happens
when it *fails*, without a model server or a GPU anywhere. A fake session
records the request that would have been sent and returns a canned reply, so
we can assert on the exact payload — which is how you catch a wrong image
ordering or a missing header before a four-minute GPU run does.

The failure tests matter most. Retry policy is the difference between a
transient blip costing three seconds and costing the whole document, and it is
almost impossible to exercise against a real server on demand.
"""

import logging
from pathlib import Path

import pytest
import requests
from PIL import Image

from src.llm.providers.backends import LocalOpenAIBackend, select_backend_name
from src.llm.providers.backends.base import CLOUD, GEMINI, LOCAL, extract_message_content
from src.llm.providers.backends.gemini import RETRY_DELAYS_SECONDS, is_rate_limit_error
from src.llm.providers.provider import VlmProvider
from src.llm.providers.schemas import (
    APIParameters,
    CloudConfig,
    GlobalSettings,
    Metadata,
    ModelProfile,
    TaskDefinition,
    UniversalConfig,
)

QUIET = logging.getLogger("test-backends")
QUIET.addHandler(logging.NullHandler())
QUIET.propagate = False

JSON_TASK = TaskDefinition(task_name="raw_ocr_extraction", output_format={"type": "json"})


def page() -> Image.Image:
    return Image.new("RGB", (40, 50), "white")


def local_profile(**kwargs) -> ModelProfile:
    kwargs.setdefault("api_parameters", APIParameters(temperature=0.0, max_tokens=2048))
    return ModelProfile(model_name="glm-ocr", model_type="official_specialized_ocr", **kwargs)


# ---------------------------------------------------------------------------
# A fake requests.Session — records what was sent, returns what we tell it
# ---------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, payload=None, status_code=200, text=""):
        self._payload = payload if payload is not None else {}
        self.status_code = status_code
        self.text = text
        self.headers = {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            error = requests.HTTPError(f"HTTP {self.status_code}")
            error.response = self  # type: ignore[attr-defined]
            raise error


class FakeSession:
    """
    Stands in for `requests.Session`.

    `replies` is consumed one per call; a `FakeResponse` is returned and an
    exception instance is raised, which is how the retry tests script a
    sequence like "time out, time out, then succeed".
    """

    def __init__(self, replies=None):
        self.replies = list(replies or [])
        self.requests = []
        self.health_calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.requests.append({"url": url, "headers": headers or {}, "json": json or {},
                              "timeout": timeout})
        reply = self.replies.pop(0) if self.replies else FakeResponse(_ok("text"))
        if isinstance(reply, Exception):
            raise reply
        return reply

    def get(self, url, timeout=None):
        self.health_calls.append(url)
        reply = self.replies.pop(0) if self.replies else FakeResponse({})
        if isinstance(reply, Exception):
            raise reply
        return reply

    @property
    def last_payload(self):
        return self.requests[-1]["json"]


def _ok(content, finish_reason="stop"):
    """An OpenAI-shaped success body."""
    return {"choices": [{"message": {"content": content}, "finish_reason": finish_reason}]}


def make_backend(session=None, profile=None, cloud=False, **kwargs):
    return LocalOpenAIBackend(
        profile or local_profile(),
        UniversalConfig(),
        task_type="raw_ocr_extraction",
        session=session or FakeSession(),
        cloud=cloud,
        logger=QUIET,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

def test_backend_selection_reads_the_profiles_model_type():
    assert select_backend_name(ModelProfile(model_name="g", model_type="gemini_cloud")) == GEMINI
    assert select_backend_name(ModelProfile(model_name="o", model_type="cloud_api_openai")) == CLOUD
    assert select_backend_name(local_profile()) == LOCAL


def test_an_unrecognised_model_type_defaults_to_the_local_server():
    """
    The safe default for an offline-first pipeline: a mistyped `model_type`
    fails loudly against localhost rather than quietly sending a paper to a
    third party.
    """
    assert select_backend_name(ModelProfile(model_name="x", model_type="typo")) == LOCAL


# ---------------------------------------------------------------------------
# Request construction
# ---------------------------------------------------------------------------

def test_the_request_carries_the_model_name_and_its_api_parameters():
    session = FakeSession([FakeResponse(_ok("page text"))])
    make_backend(session).generate(page(), JSON_TASK)

    payload = session.last_payload
    assert payload["model"] == "glm-ocr"
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == 2048


def test_the_reply_is_returned_stripped():
    session = FakeSession([FakeResponse(_ok("  page text\n\n"))])
    assert make_backend(session).generate(page(), JSON_TASK) == "page text"


def test_a_recognition_model_receives_the_image_before_the_text():
    """Pinned end to end, not just in the prompt unit: ordering changes output."""
    session = FakeSession([FakeResponse(_ok("t"))])
    profile = local_profile(prompt_mode="fixed",
                            fixed_prompts={"raw_ocr_extraction": "Text Recognition:"})
    make_backend(session, profile).generate(page(), JSON_TASK)

    content = session.last_payload["messages"][0]["content"]
    assert [part["type"] for part in content] == ["image_url", "text"]
    assert content[1]["text"] == "Text Recognition:"


def test_an_instruction_model_receives_the_text_first_with_its_task_prompt():
    session = FakeSession([FakeResponse(_ok("t"))])
    profile = ModelProfile(model_name="olmocr-2-7b", model_type="vlm")
    task = TaskDefinition(task_name="t", prompt="Read this page.",
                          output_format={"type": "json"})
    make_backend(session, profile).generate(page(), task)

    content = session.last_payload["messages"][0]["content"]
    assert [part["type"] for part in content] == ["text", "image_url"]
    assert content[0]["text"] == "Read this page."


def test_a_schema_constrained_json_task_sets_response_format():
    session = FakeSession([FakeResponse(_ok("{}"))])
    task = TaskDefinition(task_name="t",
                          output_format={"type": "json", "schema": {"type": "object"}})
    make_backend(session).generate(page(), task)

    assert session.last_payload["response_format"] == {
        "type": "json_object", "schema": {"type": "object"},
    }


def test_a_cloud_api_never_gets_response_format():
    """Measured as slower and less reliable there than prompt-based JSON."""
    session = FakeSession([FakeResponse(_ok("{}"))])
    task = TaskDefinition(task_name="t",
                          output_format={"type": "json", "schema": {"type": "object"}})
    make_backend(session, cloud=True, api_key="k").generate(page(), task)

    assert "response_format" not in session.last_payload


def test_a_plain_text_task_sets_no_output_constraints():
    session = FakeSession([FakeResponse(_ok("prose"))])
    make_backend(session).generate(page(), TaskDefinition(
        task_name="t", output_format={"type": "markdown"}))

    assert "response_format" not in session.last_payload
    assert "grammar" not in session.last_payload


def test_a_grammar_file_is_loaded_and_attached(tmp_path: Path):
    (tmp_path / "doc.gbnf").write_text('root ::= "x"', encoding="utf-8")
    session = FakeSession([FakeResponse(_ok("{}"))])
    task = TaskDefinition(task_name="t", output_format={"type": "json"}, grammar_file="doc.gbnf")

    make_backend(session, grammar_root=tmp_path).generate(page(), task)
    assert session.last_payload["grammar"] == 'root ::= "x"'


def test_a_missing_grammar_file_warns_but_does_not_stop_the_run(tmp_path: Path):
    """The JSON repair in output_parser is the safety net, so this degrades."""
    session = FakeSession([FakeResponse(_ok("{}"))])
    task = TaskDefinition(task_name="t", output_format={"type": "json"},
                          grammar_file="absent.gbnf")

    make_backend(session, grammar_root=tmp_path).generate(page(), task)
    assert "grammar" not in session.last_payload


def test_grammar_and_response_format_are_never_sent_together(tmp_path: Path):
    """
    DELIBERATE DEVIATION FROM V5, and a latent bug it closes.

    llama.cpp build 6660+ returns 400 Bad Request for a request carrying both.
    V5 carried a comment saying exactly that, then set the two independently —
    so a task declaring a schema AND a grammar file would have produced a
    guaranteed 400. Nothing hit it because no V5 task definition declares
    `grammar_file` at all (checked across all eleven), so the collision never
    fired. This changes no current behaviour; it makes the code do what V5's
    own comment already claimed.

    The schema wins, for the reason V5 gave: it is sufficient alone, and the
    output parser repairs the rest.
    """
    (tmp_path / "doc.gbnf").write_text('root ::= "x"', encoding="utf-8")
    session = FakeSession([FakeResponse(_ok("{}"))])
    task = TaskDefinition(task_name="t", grammar_file="doc.gbnf",
                          output_format={"type": "json", "schema": {"type": "object"}})

    make_backend(session, grammar_root=tmp_path).generate(page(), task)
    payload = session.last_payload
    assert "response_format" in payload
    assert "grammar" not in payload


def test_a_grammar_is_still_used_when_there_is_no_schema_to_conflict_with(tmp_path: Path):
    """The guard above must not disable grammars generally."""
    (tmp_path / "doc.gbnf").write_text('root ::= "x"', encoding="utf-8")
    session = FakeSession([FakeResponse(_ok("{}"))])
    task = TaskDefinition(task_name="t", grammar_file="doc.gbnf",
                          output_format={"type": "json"})

    make_backend(session, grammar_root=tmp_path).generate(page(), task)
    assert session.last_payload["grammar"] == 'root ::= "x"'


# ---------------------------------------------------------------------------
# Headers and authentication
# ---------------------------------------------------------------------------

def test_a_local_request_sends_no_authorization_header():
    session = FakeSession([FakeResponse(_ok("t"))])
    make_backend(session).generate(page(), JSON_TASK)
    assert "Authorization" not in session.requests[-1]["headers"]


def test_a_cloud_request_sends_a_bearer_token():
    session = FakeSession([FakeResponse(_ok("t"))])
    make_backend(session, cloud=True, api_key="secret").generate(page(), JSON_TASK)
    assert session.requests[-1]["headers"]["Authorization"] == "Bearer secret"


def test_openrouter_gets_its_attribution_headers():
    session = FakeSession([FakeResponse(_ok("t"))])
    backend = LocalOpenAIBackend(
        local_profile(),
        UniversalConfig(api_endpoint="https://openrouter.ai/api/v1/chat/completions"),
        session=session, cloud=True, api_key="k", logger=QUIET,
    )
    backend.generate(page(), JSON_TASK)
    assert "HTTP-Referer" in session.requests[-1]["headers"]


def test_a_cloud_backend_without_a_key_refuses_to_send():
    with pytest.raises(ValueError, match="No API key"):
        make_backend(FakeSession(), cloud=True).generate(page(), JSON_TASK)


# ---------------------------------------------------------------------------
# read_crop
# ---------------------------------------------------------------------------

def test_read_crop_sends_the_exact_prompt_it_was_given():
    """No task resolution — the caller chose the prompt from model config."""
    session = FakeSession([FakeResponse(_ok("| a | b |"))])
    result = make_backend(session).read_crop(page(), "Table Recognition:")

    assert result == "| a | b |"
    content = session.last_payload["messages"][0]["content"]
    assert any(part.get("text") == "Table Recognition:" for part in content)


def test_read_crop_can_override_max_tokens():
    session = FakeSession([FakeResponse(_ok("x"))])
    make_backend(session).read_crop(page(), "p", max_tokens=8192)
    assert session.last_payload["max_tokens"] == 8192


def test_read_crop_can_force_image_first_for_a_grounding_read():
    session = FakeSession([FakeResponse(_ok("x"))])
    profile = ModelProfile(model_name="olmocr-2-7b", model_type="vlm")
    make_backend(session, profile).read_crop(page(), "Locate tables.", image_first=True)
    assert session.last_payload["messages"][0]["content"][0]["type"] == "image_url"


def test_read_crop_returns_empty_string_on_failure():
    """
    V5 behaviour, ported unchanged for parity. It is a known weakness — a
    failed table read looks like an empty table rather than an error — and
    plan.md Stage 0 replaces it with a logged retry. Pinned so that change is
    a deliberate edit here, not an accident.
    """
    session = FakeSession([requests.ConnectionError("server down")])
    assert make_backend(session).read_crop(page(), "Table Recognition:") == ""


def test_read_crop_returns_empty_string_on_a_500():
    session = FakeSession([FakeResponse(status_code=500)])
    assert make_backend(session).read_crop(page(), "p") == ""


def test_read_crop_still_returns_truncated_output_with_a_warning():
    """Truncated is not nothing — partial table text beats no table text."""
    session = FakeSession([FakeResponse(_ok("| a |", finish_reason="length"))])
    assert make_backend(session).read_crop(page(), "p") == "| a |"


# ---------------------------------------------------------------------------
# Response handling and the health check
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("body", [{}, {"choices": []}, {"choices": [{}]}, {"choices": "no"}])
def test_a_malformed_response_raises_rather_than_returning_empty(body):
    with pytest.raises(ValueError, match="Invalid API response"):
        extract_message_content(body)


def test_the_health_check_targets_the_servers_health_path():
    session = FakeSession([FakeResponse({})])
    make_backend(session).check_availability()
    assert session.health_calls == ["http://localhost:9931/health"]


def test_a_failed_health_check_raises_a_clear_connection_error():
    session = FakeSession([requests.ConnectionError("refused")])
    with pytest.raises(ConnectionError, match="Could not connect to VLM server"):
        make_backend(session).check_availability()


# ---------------------------------------------------------------------------
# The provider: sequencing and retry policy
# ---------------------------------------------------------------------------

def make_provider(session, profile=None, **kwargs):
    return VlmProvider(
        profile or local_profile(), JSON_TASK,
        config=UniversalConfig(), task_type="raw_ocr_extraction",
        session=session, logger=QUIET, **kwargs,
    )


def test_process_page_parses_and_scores_a_successful_read():
    session = FakeSession([FakeResponse(_ok('{"natural_text":"Body text.","is_table":false}'))])
    result = make_provider(session).process_page(page())

    assert result["success"]
    assert result["content"] == "Body text."
    assert result["metadata"] == {"is_table": False}
    assert 0.0 <= result["quality_score"] <= 1.0


def test_a_failed_page_returns_a_result_rather_than_raising():
    """One unreadable page must not end an eight-page document."""
    session = FakeSession([requests.ConnectionError("down")] * 3)
    result = make_provider(session).process_page(page())

    assert result["success"] is False
    assert result["content"] == ""
    assert result["quality_score"] == 0.0
    assert "down" in result["error"]


def test_a_timeout_is_retried_and_can_then_succeed(monkeypatch):
    monkeypatch.setattr("src.llm.providers.provider.time.sleep", lambda _: None)
    session = FakeSession([
        requests.Timeout("slow"),
        FakeResponse(_ok('{"natural_text":"recovered"}')),
    ])
    result = make_provider(session).process_page(page())

    assert result["success"] and result["content"] == "recovered"
    assert len(session.requests) == 2


def test_a_server_error_is_retried(monkeypatch):
    monkeypatch.setattr("src.llm.providers.provider.time.sleep", lambda _: None)
    session = FakeSession([FakeResponse(status_code=503),
                           FakeResponse(_ok('{"natural_text":"ok"}'))])
    assert make_provider(session).process_page(page())["success"]
    assert len(session.requests) == 2


def test_a_client_error_is_never_retried():
    """
    A malformed or unauthorized request will be just as wrong next time, and
    each attempt costs real seconds on this pipeline.
    """
    session = FakeSession([FakeResponse(status_code=400)] * 3)
    result = make_provider(session).process_page(page())

    assert result["success"] is False
    assert len(session.requests) == 1, "a 4xx must not be retried"


def test_retries_stop_after_the_configured_maximum(monkeypatch):
    monkeypatch.setattr("src.llm.providers.provider.time.sleep", lambda _: None)
    session = FakeSession([requests.Timeout("slow")] * 10)
    config = UniversalConfig(metadata=Metadata(
        provider_type="local", version="1",
        global_settings=GlobalSettings(max_retries=2)))

    provider = VlmProvider(local_profile(), JSON_TASK, config=config,
                           session=session, logger=QUIET)
    assert provider.process_page(page())["success"] is False
    assert len(session.requests) == 2


def test_backoff_doubles_between_attempts(monkeypatch):
    waits = []
    monkeypatch.setattr("src.llm.providers.provider.time.sleep", waits.append)
    session = FakeSession([requests.Timeout("slow")] * 3)

    make_provider(session).process_page(page())
    assert waits == [1, 2], "exponential backoff: 1s then 2s, none after the last attempt"


def test_a_malformed_response_is_retried_exactly_once(monkeypatch):
    """Usually a truncated reply — occasionally transient, rarely twice."""
    monkeypatch.setattr("src.llm.providers.provider.time.sleep", lambda _: None)
    session = FakeSession([FakeResponse({"unexpected": "shape"}),
                           FakeResponse({"unexpected": "shape"}),
                           FakeResponse(_ok('{"natural_text":"never reached"}'))])

    assert make_provider(session).process_page(page())["success"] is False
    assert len(session.requests) == 2


def test_a_profile_with_cloud_config_overrides_the_local_endpoint():
    """Otherwise every cloud request would be posted to localhost."""
    session = FakeSession([FakeResponse(_ok("{}"))])
    profile = ModelProfile(
        model_name="gpt", model_type="cloud_api_openai",
        cloud_config=CloudConfig(api_endpoint="https://api.example.com/v1/chat/completions",
                                 api_key_env="TEST_KEY_VAR"),
    )
    import os
    os.environ["TEST_KEY_VAR"] = "k"
    try:
        make_provider(session, profile).process_page(page())
    finally:
        del os.environ["TEST_KEY_VAR"]

    assert session.requests[-1]["url"] == "https://api.example.com/v1/chat/completions"


def test_a_cloud_profile_without_its_environment_variable_fails_clearly(monkeypatch):
    monkeypatch.delenv("ABSENT_KEY_VAR", raising=False)
    profile = ModelProfile(
        model_name="gpt", model_type="cloud_api_openai",
        cloud_config=CloudConfig(api_endpoint="https://api.example.com/v1",
                                 api_key_env="ABSENT_KEY_VAR"),
    )
    with pytest.raises(ValueError, match="ABSENT_KEY_VAR"):
        make_provider(FakeSession(), profile)


def test_an_injected_backend_is_used_as_given():
    """The seam that lets a caller substitute a transport wholesale."""
    class StubBackend:
        name = "stub"
        def generate(self, *_args, **_kwargs):
            return '{"natural_text":"from the stub"}'
        def read_crop(self, *_args, **_kwargs):
            return "crop text"

    provider = VlmProvider(local_profile(), JSON_TASK, backend=StubBackend(), logger=QUIET)
    assert provider.process_page(page())["content"] == "from the stub"
    assert provider.read_crop(page(), "p") == "crop text"


def test_the_server_is_not_pinged_unless_validation_is_requested():
    """Providers are built before the server loads in the lazy-launch flow."""
    session = FakeSession()
    make_provider(session)
    assert session.health_calls == []


def test_validation_pings_the_server_when_requested():
    session = FakeSession([FakeResponse({})])
    make_provider(session, validate_server=True)
    assert session.health_calls == ["http://localhost:9931/health"]


# ---------------------------------------------------------------------------
# Gemini: the pure pieces, with no SDK installed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("message", [
    "429 Too Many Requests", "RESOURCE_EXHAUSTED: quota exceeded",
    "Rate limit reached", "503 Service Unavailable", "The model is overloaded",
])
def test_rate_limit_errors_are_recognised(message):
    assert is_rate_limit_error(Exception(message))


@pytest.mark.parametrize("message", [
    "400 Bad Request", "invalid API key", "connection reset", "",
])
def test_other_errors_are_not_treated_as_rate_limits(message):
    """Retrying a genuine failure just burns quota."""
    assert not is_rate_limit_error(Exception(message))


def test_the_retry_schedule_waits_once_for_a_per_minute_quota():
    """
    Four attempts, with a single 60s wait before the third: two quick tries
    for a transient 429, then wait out a per-minute quota, then two more.
    """
    assert RETRY_DELAYS_SECONDS == [0, 0, 60, 0]
    assert sum(1 for delay in RETRY_DELAYS_SECONDS if delay) == 1


def test_the_gemini_sdk_is_not_imported_by_the_package():
    """
    An offline run must never need `google-genai`. V5 imported it at module
    level on every run; here it is imported only when a profile selects it.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; import src.llm.providers; "
         "assert 'google.genai' not in sys.modules, 'SDK imported eagerly'; print('clean')"],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent.parent),
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout
