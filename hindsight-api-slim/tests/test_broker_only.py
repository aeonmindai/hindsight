"""The broker-only profile and the ``broker`` provider (RUN-2355: memory on the auth broker's logins)."""

from types import SimpleNamespace

import pytest

from hindsight_api.broker_only import enforce, violations
from hindsight_api.config import HindsightConfig
from hindsight_api.engine.llm_wrapper import create_llm_provider
from hindsight_api.engine.providers.broker_llm import broker_model_family, broker_wire, gateway_root

GATEWAY = "http://127.0.0.1:4000"


def _config(**overrides):
    base = {
        "broker_only": True,
        "llm_provider": "broker",
        "llm_model": "anthropic/claude-haiku-5-5",
        "llm_base_url": GATEWAY,
        "retain_llm_provider": None,
        "retain_llm_model": None,
        "retain_llm_base_url": None,
        "vlm_provider": None,
        "reflect_llm_provider": None,
        "consolidation_llm_provider": "broker",
        "consolidation_llm_model": "openai-codex/gpt-6-luna",
        "consolidation_llm_base_url": None,
        "mental_model_refresh_llm_provider": None,
        "decision_model_provider": "none",
        "embeddings_provider": "local",
        "query_embeddings": None,
        "passage_embeddings": None,
        "reranker_provider": "local",
        "webhook_url": None,
        "otel_traces_enabled": False,
        "otel_exporter_otlp_endpoint": None,
    }
    return SimpleNamespace(**{**base, **overrides})


def test_anthropic_and_codex_through_a_gateway_with_local_embeddings_is_allowed():
    assert violations(_config()) == []
    enforce(_config())


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"llm_provider": "openai"}, "llm_provider='openai'"),
        ({"llm_provider": "anthropic"}, "llm_provider='anthropic'"),
        ({"llm_provider": "claude-code"}, "llm_provider='claude-code'"),
        ({"llm_provider": "openai-codex"}, "llm_provider='openai-codex'"),
        ({"consolidation_llm_provider": "gemini"}, "consolidation_llm_provider='gemini'"),
        ({"retain_llm_provider": "groq"}, "retain_llm_provider='groq'"),
        ({"llm_model": "gpt-4o-mini"}, "model 'gpt-4o-mini' is not a broker model"),
        ({"llm_model": "google/gemini-3"}, "model 'google/gemini-3' is not a broker model"),
        ({"llm_model": "anthropic/"}, "is not a broker model"),
        ({"retain_llm_model": "gpt-4o-mini"}, "retain_llm_provider model 'gpt-4o-mini'"),
        ({"consolidation_llm_model": "o3"}, "consolidation_llm_provider model 'o3'"),
        ({"llm_base_url": "https://api.anthropic.com"}, "base URL 'https://api.anthropic.com'"),
        ({"llm_base_url": None}, "base URL None"),
        ({"consolidation_llm_base_url": "https://api.openai.com/v1"}, "consolidation_llm_provider base URL"),
        ({"decision_model_provider": "openai"}, "decision_model_provider='openai'"),
        ({"reflect_decision_model_provider": "typesafe"}, "reflect_decision_model_provider='typesafe'"),
        ({"embeddings_provider": "openai"}, "embeddings provider 'openai'"),
        ({"embeddings_provider": "cohere"}, "embeddings provider 'cohere'"),
        ({"reranker_provider": "cohere"}, "reranker_provider='cohere'"),
        ({"webhook_url": "https://hooks.example.com/x"}, "webhook_url"),
        ({"otel_traces_enabled": True, "otel_exporter_otlp_endpoint": "https://otel.vendor.io"}, "otlp"),
        ({"file_parser_iris_token": "secret"}, "iris"),
        ({"llm_members": [SimpleNamespace(provider="openai")]}, "member provider 'openai'"),
        (
            {"llm_members": [SimpleNamespace(provider="broker", model="gpt-4o", base_url=GATEWAY)]},
            "llm_members member model 'gpt-4o'",
        ),
    ],
)
def test_anything_that_is_not_a_broker_login_is_a_violation(overrides, fragment):
    problems = violations(_config(**overrides))

    assert any(fragment in problem for problem in problems), problems
    with pytest.raises(ValueError, match="HINDSIGHT_API_BROKER_ONLY"):
        enforce(_config(**overrides))


def test_an_operation_that_only_overrides_its_model_is_checked_against_the_global_provider():
    problems = violations(_config(retain_llm_model="gpt-4o-mini"))

    assert any("retain_llm_provider model 'gpt-4o-mini'" in problem for problem in problems), problems


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:4000",
        "http://127.0.0.1:4000",
        "http://10.1.2.3:4000",
        "http://100.101.102.103:4000",
        "http://neo-auth-gateway.tail9b529e.ts.net:4000",
    ],
)
def test_the_gateway_may_be_local_private_or_on_the_tailnet(url):
    assert violations(_config(llm_base_url=url)) == []


def test_a_self_hosted_tei_is_allowed_only_on_a_private_address():
    private = _config(embeddings_provider="tei", embeddings_tei_url="http://10.0.0.5:8080")
    public = _config(embeddings_provider="tei", embeddings_tei_url="https://tei.vendor.io")

    assert violations(private) == []
    assert violations(public) != []


def test_the_profile_off_checks_nothing():
    enforce(_config(broker_only=False, llm_provider="openai", llm_model="gpt-4o", embeddings_provider="openai"))


def test_startup_is_refused_when_the_environment_names_another_provider(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_BROKER_ONLY", "true")
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("HINDSIGHT_API_LLM_API_KEY", "sk-ant-test")
    with pytest.raises(ValueError, match="llm_provider='anthropic'"):
        HindsightConfig.from_env().validate()


def test_the_environment_switch_accepts_broker_models_and_local_embeddings(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_BROKER_ONLY", "true")
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "broker")
    monkeypatch.setenv("HINDSIGHT_API_LLM_MODEL", "anthropic/claude-sonnet-5-5")
    monkeypatch.setenv("HINDSIGHT_API_LLM_BASE_URL", GATEWAY)
    monkeypatch.setenv("HINDSIGHT_API_CONSOLIDATION_LLM_PROVIDER", "broker")
    monkeypatch.setenv("HINDSIGHT_API_CONSOLIDATION_LLM_MODEL", "openai-codex/gpt-6-luna")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_PROVIDER", "local")
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_PROVIDER", "local")
    for name in (
        "HINDSIGHT_API_RETAIN_LLM_PROVIDER",
        "HINDSIGHT_API_REFLECT_LLM_PROVIDER",
        "HINDSIGHT_API_WEBHOOK_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    config = HindsightConfig.from_env()

    assert config.broker_only is True
    config.validate()


def test_the_environment_switch_refuses_a_broker_provider_with_a_vendor_model(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_BROKER_ONLY", "true")
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "broker")
    monkeypatch.setenv("HINDSIGHT_API_LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("HINDSIGHT_API_LLM_BASE_URL", GATEWAY)
    with pytest.raises(ValueError, match="model 'gpt-4o-mini' is not a broker model"):
        HindsightConfig.from_env().validate()


# ----------------------------------------------------------------------------- the provider


def test_the_model_id_names_the_family():
    assert broker_model_family("anthropic/claude-haiku-5-5") == "anthropic"
    assert broker_model_family("openai-codex/gpt-6-luna") == "openai-codex"
    assert broker_model_family("google-antigravity/gemini-4-argon") == "google-antigravity"
    assert broker_model_family("opencode-go/kimi-k3.1-pro") == "opencode-go"
    for bad in ("claude-haiku-5-5", "gpt-4o", "openai/gpt-4o", "anthropic/", "openrouter/anthropic/claude"):
        with pytest.raises(ValueError, match="not a broker model"):
            broker_model_family(bad)


def test_the_gateway_url_loses_a_trailing_v1():
    assert gateway_root("http://127.0.0.1:4000") == "http://127.0.0.1:4000"
    assert gateway_root("http://127.0.0.1:4000/") == "http://127.0.0.1:4000"
    assert gateway_root("http://127.0.0.1:4000/v1") == "http://127.0.0.1:4000"
    with pytest.raises(ValueError, match="needs the auth gateway's URL"):
        gateway_root("")


def _provider(model, base_url=GATEWAY, api_key=None):
    return create_llm_provider(
        provider="broker",
        api_key=api_key,
        base_url=base_url,
        model=model,
        reasoning_effort=None,
        timeout=30.0,
    )


def test_a_claude_model_is_called_as_an_anthropic_messages_request_to_the_gateway():
    llm = _provider("anthropic/claude-haiku-5-5")

    assert type(llm).__name__ == "AnthropicLLM"
    assert llm.model == "anthropic/claude-haiku-5-5"
    assert str(llm._client.base_url).rstrip("/") == GATEWAY
    # The gateway does not act on output_config, so the schema is asked for in the prompt.
    assert llm._output_format_rejected is True


def test_a_codex_model_is_called_as_an_openai_chat_request_to_the_gateway():
    llm = _provider("openai-codex/gpt-6-luna", base_url=GATEWAY + "/v1", api_key="gateway-token")

    assert type(llm).__name__ == "OpenAICompatibleLLM"
    assert llm.model == "openai-codex/gpt-6-luna"
    assert str(llm._client.base_url).rstrip("/") == GATEWAY + "/v1"
    assert llm._client.api_key == "gateway-token"


def test_a_vendor_model_is_refused_by_the_provider_itself():
    with pytest.raises(ValueError, match="not a broker model"):
        _provider("gpt-4o-mini")


def test_the_provider_needs_a_gateway_url():
    with pytest.raises(ValueError, match="needs the auth gateway's URL"):
        _provider("anthropic/claude-haiku-5-5", base_url="")


def test_claude_models_use_the_messages_wire_and_every_other_model_chat_completions():
    assert broker_wire("anthropic/claude-sonnet-5-5") == "anthropic"
    assert broker_wire("google-antigravity/claude-opus-5-5") == "anthropic"
    assert broker_wire("google-antigravity/gemini-4-argon") == "openai"
    assert broker_wire("openai-codex/gpt-6.1-sol") == "openai"
    assert broker_wire("opencode-go/glm-5.3") == "openai"


@pytest.mark.parametrize(
    ("model", "provider_class"),
    [
        ("google-antigravity/claude-sonnet-5-5", "AnthropicLLM"),
        ("google-antigravity/gemini-3.8-flash", "OpenAICompatibleLLM"),
        ("opencode-go/kimi-k3.1-pro", "OpenAICompatibleLLM"),
    ],
)
def test_antigravity_and_opencode_models_reach_the_gateway(model, provider_class):
    llm = _provider(model)

    assert type(llm).__name__ == provider_class
    assert llm.model == model
    assert violations(_config(llm_model=model)) == []
