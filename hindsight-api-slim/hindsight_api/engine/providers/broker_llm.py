"""The ``broker`` provider: model calls signed by the auth broker's subscription logins.

The broker holds our Claude Max and ChatGPT (Codex) OAuth logins. ``omp auth-gateway serve`` is the
only process that signs calls with them: it takes OpenAI chat-completions, Anthropic Messages or
OpenAI Responses requests, adds the Claude Code or Codex headers and the access token, and sends them
to the vendor. This provider never sees a token; it speaks the gateway's wire and names its models
the way the gateway does, ``<provider>/<model>``:

* ``anthropic/claude-haiku-5-5`` goes out as an Anthropic Messages call, the shape the Claude login accepts;
* ``openai-codex/gpt-6-luna`` goes out as an OpenAI chat-completions call, which the gateway sends on to
  the ChatGPT backend with the Codex login.

``base_url`` is the gateway (``http://127.0.0.1:4000`` by default for ``omp auth-gateway serve``);
``api_key`` is the gateway's bearer token, or anything when the gateway runs ``--no-auth`` on loopback.
"""

from __future__ import annotations

from typing import Any

from ...broker_only import BROKER_MODEL_PREFIXES, is_broker_model
from ..llm_interface import LLMInterface

# The SDKs refuse an empty key; a gateway started with --no-auth ignores whatever arrives.
NO_AUTH_PLACEHOLDER_KEY = "broker-gateway"


def broker_model_family(model: str) -> str:
    """``anthropic`` or ``openai-codex`` for a gateway model id; ``ValueError`` for any other model."""
    if is_broker_model(model):
        return next(p.rstrip("/") for p in BROKER_MODEL_PREFIXES if model.startswith(p))
    raise ValueError(
        f"Broker model {model!r} is not a broker model: name it 'anthropic/<model>' or 'openai-codex/<model>' "
        "(the ids that `GET <gateway>/v1/models` lists)."
    )


def gateway_root(base_url: str) -> str:
    """The gateway's address without a trailing ``/v1``, for the Anthropic SDK, which appends ``/v1/messages``."""
    if not base_url:
        raise ValueError(
            "The broker provider needs the auth gateway's URL: set HINDSIGHT_API_LLM_BASE_URL "
            "(for example http://127.0.0.1:4000) or the operation's *_LLM_BASE_URL."
        )
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return root


def create_broker_llm(
    *,
    api_key: str | None,
    base_url: str,
    model: str,
    reasoning_effort: str | None,
    timeout: float | None,
    extra_body: dict[str, Any] | None,
    default_headers: dict[str, str] | None,
    cache_affinity: str | None = None,
    openai_service_tier: str | None = None,
) -> LLMInterface:
    """The provider that reaches ``model`` through the gateway at ``base_url``."""
    family = broker_model_family(model)
    root = gateway_root(base_url)
    key = api_key or NO_AUTH_PLACEHOLDER_KEY

    if family == "anthropic":
        from .anthropic_llm import AnthropicLLM

        llm = AnthropicLLM(
            provider="anthropic",
            api_key=key,
            base_url=root,
            model=model,
            reasoning_effort=reasoning_effort,
            default_headers=default_headers,
            extra_body=extra_body,
            timeout=timeout,
        )
        # The gateway takes an ``output_config`` without acting on it: the reply would come back
        # unconstrained and unreported. Ask for the schema in the prompt from the first call.
        llm._output_format_rejected = True
        return llm

    from .openai_compatible_llm import OpenAICompatibleLLM

    return OpenAICompatibleLLM(
        provider="openai",
        api_key=key,
        base_url=f"{root}/v1",
        model=model,
        reasoning_effort=reasoning_effort,
        openai_service_tier=openai_service_tier,
        extra_body=extra_body,
        default_headers=default_headers,
        cache_affinity=cache_affinity,
        timeout=timeout,
    )
