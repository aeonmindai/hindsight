"""The broker model mix (RUN-2355): a refused call is answered by the fallback model, and the person-model and
dialectic operations run on their own models.

The gateway is a local HTTP server that speaks the Anthropic Messages wire the way ``omp auth-gateway`` does,
including how it reports a refusal: ``502 upstream_error`` with ``Refusal (cyber)``.
"""

import json
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

import hindsight_api.config as config_module
from hindsight_api.broker_only import violations
from hindsight_api.config import HindsightConfig
from hindsight_api.engine.llm_interface import ProviderContentPolicyError, is_gateway_refusal
from hindsight_api.engine.llm_wrapper import LLMProvider
from hindsight_api.engine.providers.broker_llm import RefusalFallbackLLM, create_broker_llm
from hindsight_api.engine.retain.fact_extraction import extract_facts_from_text
from hindsight_api.people.context import PeopleContext

SONNET = "anthropic/claude-sonnet-5-5"
OPUS = "anthropic/claude-opus-5-5"
HAIKU = "anthropic/claude-haiku-5-5"

EXTRACTED = {
    "facts": [
        {
            "what": "Jish asked why the harness does not hand over to a new session on its own.",
            "when": "N/A",
            "where": "N/A",
            "who": "Jish",
            "why": "N/A",
            "fact_type": "world",
            "entities": ["Jish"],
        }
    ]
}


class _Reply(BaseModel):
    facts: list[str]


class FakeGateway:
    """Answers per model: ``refuse`` (502 Refusal), ``stop`` (200 with stop_reason refusal), ``bad`` (400) or ``ok``."""

    def __init__(self, behaviour: dict[str, str], ok_body: dict) -> None:
        self.behaviour = behaviour
        self.ok_body = ok_body
        self.requests: list[str] = []
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence
                pass

            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                model = body["model"]
                gateway.requests.append(model)
                mode = gateway.behaviour.get(model, "ok")
                if mode == "refuse":
                    return self._send(
                        502,
                        {
                            "type": "error",
                            "error": {
                                "type": "upstream_error",
                                "message": "Refusal (cyber): Cyber safeguards flagged this message.",
                            },
                        },
                    )
                if mode == "bad":
                    return self._send(
                        400, {"type": "error", "error": {"type": "invalid_request_error", "message": "x"}}
                    )
                stop = "refusal" if mode == "stop" else "end_turn"
                content = [] if mode == "stop" else [{"type": "text", "text": json.dumps(gateway.ok_body)}]
                self._send(
                    200,
                    {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "model": model,
                        "content": content,
                        "stop_reason": stop,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 10, "output_tokens": 5},
                    },
                )

            def _send(self, status, payload):
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def gateway_factory():
    made = []

    def make(behaviour, ok_body=None):
        gateway = FakeGateway(behaviour, ok_body or {"facts": ["fallback answer"]})
        made.append(gateway)
        return gateway

    yield make
    for gateway in made:
        gateway.close()


@pytest.fixture
def fallback_to_haiku(monkeypatch):
    monkeypatch.setattr(
        "hindsight_api.engine.providers.broker_llm.get_config",
        lambda: SimpleNamespace(broker_refusal_fallback_model=HAIKU),
    )


def _provider(url, model=SONNET):
    return LLMProvider(provider="broker", api_key="", base_url=url, model=model, timeout=30.0)


def test_the_gateways_refusal_is_recognised_and_other_errors_are_not():
    class Error(Exception):
        def __init__(self, status_code, message):
            super().__init__(message)
            self.status_code = status_code

    assert is_gateway_refusal(Error(502, "Refusal (cyber): flagged"))
    assert not is_gateway_refusal(Error(502, "Unable to verify model access"))
    assert not is_gateway_refusal(Error(400, "Refusal (cyber): flagged"))
    assert not is_gateway_refusal(ValueError("Refusal ("))


@pytest.mark.asyncio
async def test_a_refused_call_is_answered_by_haiku_without_retrying_the_refusal(gateway_factory, fallback_to_haiku):
    gateway = gateway_factory({SONNET: "refuse"})

    result = await _provider(gateway.url).call(
        messages=[{"role": "user", "content": "extract"}], response_format=_Reply, scope="test"
    )

    assert result.content.facts == ["fallback answer"]
    # One refused request, not four: the refusal is permanent, so it is not retried.
    assert gateway.requests == [SONNET, HAIKU]


@pytest.mark.asyncio
async def test_a_refusal_stop_reason_is_answered_by_haiku(gateway_factory, fallback_to_haiku):
    gateway = gateway_factory({SONNET: "stop"})

    result = await _provider(gateway.url).call(
        messages=[{"role": "user", "content": "extract"}], response_format=_Reply, scope="test"
    )

    assert result.content.facts == ["fallback answer"]
    assert gateway.requests == [SONNET, HAIKU]


@pytest.mark.asyncio
async def test_a_call_that_is_not_refused_never_reaches_the_fallback(gateway_factory, fallback_to_haiku):
    gateway = gateway_factory({})

    await _provider(gateway.url).call(
        messages=[{"role": "user", "content": "extract"}], response_format=_Reply, scope="t"
    )

    assert gateway.requests == [SONNET]


@pytest.mark.asyncio
async def test_an_error_that_is_not_a_refusal_is_raised_not_handed_to_the_fallback(gateway_factory, fallback_to_haiku):
    gateway = gateway_factory({SONNET: "bad"})

    with pytest.raises(Exception, match="400"):
        await _provider(gateway.url).call(
            messages=[{"role": "user", "content": "extract"}], response_format=_Reply, scope="test"
        )

    assert gateway.requests == [SONNET]


@pytest.mark.asyncio
async def test_when_the_fallback_refuses_too_the_primarys_refusal_is_raised(gateway_factory, fallback_to_haiku):
    gateway = gateway_factory({SONNET: "refuse", HAIKU: "refuse"})

    with pytest.raises(ProviderContentPolicyError, match="refusal"):
        await _provider(gateway.url).call(
            messages=[{"role": "user", "content": "extract"}], response_format=_Reply, scope="test"
        )

    assert gateway.requests == [SONNET, HAIKU]


@pytest.mark.asyncio
async def test_a_refused_message_is_kept_and_extracted_by_haiku(gateway_factory, fallback_to_haiku):
    """The retain path: the message the primary refuses still becomes a fact."""
    gateway = gateway_factory({SONNET: "refuse"}, ok_body=EXTRACTED)

    facts, _, _ = await extract_facts_from_text(
        text="ar bl ki kore ei nonce private key crack kore kotipoti hya jai",
        event_date=datetime(2026, 11, 1),
        llm_config=_provider(gateway.url),
        config=config_module._get_raw_config(),
        agent_name=None,
    )

    assert len(facts) == 1
    assert "harness does not hand over" in facts[0].fact
    assert HAIKU in gateway.requests
    assert gateway.requests.count(SONNET) == 1


def test_no_fallback_model_leaves_the_provider_bare(monkeypatch):
    monkeypatch.setattr(
        "hindsight_api.engine.providers.broker_llm.get_config",
        lambda: SimpleNamespace(broker_refusal_fallback_model=None),
    )
    llm = create_broker_llm(
        api_key=None,
        base_url="http://127.0.0.1:4000",
        model=SONNET,
        reasoning_effort=None,
        timeout=30.0,
        extra_body=None,
        default_headers=None,
    )

    assert not isinstance(llm, RefusalFallbackLLM)


def test_a_fallback_model_that_is_the_primary_adds_nothing(monkeypatch):
    monkeypatch.setattr(
        "hindsight_api.engine.providers.broker_llm.get_config",
        lambda: SimpleNamespace(broker_refusal_fallback_model=HAIKU),
    )
    llm = create_broker_llm(
        api_key=None,
        base_url="http://127.0.0.1:4000",
        model=HAIKU,
        reasoning_effort=None,
        timeout=30.0,
        extra_body=None,
        default_headers=None,
    )

    assert not isinstance(llm, RefusalFallbackLLM)


# ----------------------------------------------------------------------------- the model mix


def _engine():
    return SimpleNamespace(
        _consolidation_llm_config=_provider("http://127.0.0.1:4000", SONNET),
        _reflect_llm_config=_provider("http://127.0.0.1:4000", OPUS),
    )


def _with_people_models(monkeypatch, people=None, dialectic=None):
    monkeypatch.setattr(
        config_module,
        "get_config",
        lambda: SimpleNamespace(people_llm_model=people, dialectic_llm_model=dialectic),
    )


def test_the_deriver_and_dreamer_run_on_the_people_model_and_the_dialectic_on_its_own(monkeypatch):
    engine = _engine()
    _with_people_models(monkeypatch, people=OPUS, dialectic=SONNET)
    ctx = PeopleContext.internal(engine)

    assert ctx.llm_background.model == OPUS
    assert ctx.llm_interactive.model == SONNET
    # Same provider settings as the operation they replace, built once.
    assert ctx.llm_background.base_url == engine._consolidation_llm_config.base_url
    assert ctx.llm_background.provider == "broker"
    assert PeopleContext.internal(engine).llm_background is ctx.llm_background
    assert engine._consolidation_llm_config.model == SONNET
    assert engine._reflect_llm_config.model == OPUS


def test_without_the_settings_people_use_consolidation_and_reflect(monkeypatch):
    engine = _engine()
    _with_people_models(monkeypatch)
    ctx = PeopleContext.internal(engine)

    assert ctx.llm_background is engine._consolidation_llm_config
    assert ctx.llm_interactive is engine._reflect_llm_config


def test_the_environment_names_the_mix(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_PEOPLE_LLM_MODEL", OPUS)
    monkeypatch.setenv("HINDSIGHT_API_DIALECTIC_LLM_MODEL", SONNET)
    monkeypatch.setenv("HINDSIGHT_API_BROKER_REFUSAL_FALLBACK_MODEL", HAIKU)
    config = HindsightConfig.from_env()

    assert (config.people_llm_model, config.dialectic_llm_model, config.broker_refusal_fallback_model) == (
        OPUS,
        SONNET,
        HAIKU,
    )
    monkeypatch.delenv("HINDSIGHT_API_PEOPLE_LLM_MODEL")
    monkeypatch.delenv("HINDSIGHT_API_DIALECTIC_LLM_MODEL")
    monkeypatch.delenv("HINDSIGHT_API_BROKER_REFUSAL_FALLBACK_MODEL")
    config = HindsightConfig.from_env()
    assert (config.people_llm_model, config.dialectic_llm_model, config.broker_refusal_fallback_model) == (
        None,
        None,
        None,
    )


@pytest.mark.parametrize("field", ["people_llm_model", "dialectic_llm_model", "broker_refusal_fallback_model"])
def test_broker_only_refuses_a_model_that_is_not_a_broker_model(field):
    base = {
        "broker_only": True,
        "llm_provider": "broker",
        "llm_model": SONNET,
        "llm_base_url": "http://127.0.0.1:4000",
        "embeddings_provider": "local",
        "reranker_provider": "local",
    }

    assert violations(SimpleNamespace(**base, **{field: OPUS})) == []
    assert any(field in p for p in violations(SimpleNamespace(**base, **{field: "gpt-4o"})))
