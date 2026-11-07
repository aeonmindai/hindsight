"""The broker mix shipped in .env.example (RUN-2355) is a valid broker-only configuration, and means what it says."""

from pathlib import Path

import pytest

from hindsight_api.config import HindsightConfig
from hindsight_api.engine.providers.broker_llm import RefusalFallbackLLM, create_broker_llm

ENV_EXAMPLE = Path(__file__).resolve().parents[2] / ".env.example"

SONNET_5 = "anthropic/claude-sonnet-5"
SONNET_55 = "anthropic/claude-sonnet-5-5"
TERRA = "openai-codex/gpt-6.1-terra"


def _active_settings() -> dict[str, str]:
    """The uncommented KEY=value lines of .env.example."""
    settings = {}
    for line in ENV_EXAMPLE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            settings[key] = value.split("  #")[0].strip()
    return settings


@pytest.fixture
def mix_config(monkeypatch):
    import os

    for name in [n for n in os.environ if n.startswith("HINDSIGHT_API_")]:
        monkeypatch.delenv(name)
    for key, value in _active_settings().items():
        monkeypatch.setenv(key, value)
    return HindsightConfig.from_env()


def test_the_example_is_a_valid_broker_only_configuration(mix_config):
    mix_config.validate()

    assert mix_config.broker_only is True
    assert mix_config.llm_provider == "broker"


def test_each_operation_runs_on_the_model_the_benchmark_picked(mix_config):
    assert mix_config.retain_llm_model == SONNET_5
    assert mix_config.consolidation_llm_model == TERRA
    assert mix_config.people_llm_model == TERRA
    assert mix_config.dialectic_llm_model == TERRA
    assert mix_config.reflect_llm_model == SONNET_55
    assert mix_config.mental_model_refresh_llm_model == SONNET_55
    assert mix_config.broker_refusal_fallback_model == TERRA


def test_no_model_in_the_mix_is_haiku_antigravity_or_opencode(mix_config):
    text = " ".join(_active_settings().values())

    for banned in ("haiku", "antigravity", "opencode", "gemini"):
        assert banned not in text.lower()


def test_reasoning_effort_is_left_to_the_provider(mix_config):
    assert not [key for key in _active_settings() if "REASONING_EFFORT" in key]


def test_timeouts_come_from_the_measured_latency(mix_config):
    # Slowest call measured in this mix: 55 s. The global limit keeps 2x headroom; reflect's own
    # per-call limit is raised from 30 s because the dialectic on Terra reached 27 s.
    assert mix_config.llm_timeout == 120
    assert mix_config.reflect_llm_timeout == 60
    assert mix_config.reflect_wall_timeout == 300


def test_the_output_language_keeps_the_speakers_words(mix_config):
    assert mix_config.llm_output_language == "English"
    assert mix_config.llm_output_language_directive == "keep-words"


def test_a_refusal_falls_back_to_terra_for_claude_and_terra_has_no_wrapper(mix_config, monkeypatch):
    monkeypatch.setattr(
        "hindsight_api.engine.providers.broker_llm.get_config",
        lambda: mix_config,
    )

    def provider(model):
        return create_broker_llm(
            api_key=None,
            base_url=mix_config.llm_base_url,
            model=model,
            reasoning_effort=None,
            timeout=30.0,
            extra_body=None,
            default_headers=None,
        )

    claude = provider(SONNET_5)
    assert isinstance(claude, RefusalFallbackLLM)
    assert claude.fallback.model == TERRA
    assert not isinstance(provider(TERRA), RefusalFallbackLLM)
