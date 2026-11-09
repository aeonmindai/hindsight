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

import json
import logging
from typing import Any

from ...broker_only import BROKER_MODEL_PREFIXES, is_broker_model
from ...config import get_config
from ..llm_interface import LLMInterface, ProviderContentPolicyError, is_gateway_refusal

logger = logging.getLogger(__name__)

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


class RefusalFallbackLLM(LLMInterface):
    """A broker model that hands a call it refused, or could not answer in JSON, to a second broker model.

    A refusal (the gateway's ``502 Refusal (cyber)``, or a refusal stop reason) is a function of the
    content: retrying the same model repeats it, and without a fallback the message is never
    extracted. So is a reply that is still not JSON after the provider's own retries (an empty reply or
    prose where the schema asked for JSON, which is what a model that declines a message without saying so
    sends back: 38 retains failed that way on 8 Nov and no retry ever got through). The fallback model sees
    the same request; if it fails the same way, the primary's error is raised. Every other error passes through untouched.
    """

    def __init__(self, primary: LLMInterface, fallback: LLMInterface) -> None:
        super().__init__(
            provider=primary.provider,
            api_key=primary.api_key,
            base_url=primary.base_url,
            model=primary.model,
            reasoning_effort=primary.reasoning_effort,
            timeout=primary.timeout,
        )
        self.primary = primary
        self.fallback = fallback

    async def verify_connection(self) -> None:
        await self.primary.verify_connection()

    async def call(self, messages: list[dict[str, str]], *args: Any, **kwargs: Any) -> Any:
        try:
            return await self.primary.call(messages, *args, **kwargs)
        except Exception as error:
            if not (isinstance(error, (ProviderContentPolicyError, json.JSONDecodeError)) or is_gateway_refusal(error)):
                raise
            scope = kwargs.get("scope", "memory")
            logger.warning(
                "[%s] %s refused the call (%s); answering it with %s",
                scope,
                self.primary.model,
                str(error)[:160],
                self.fallback.model,
            )
            try:
                return await self.fallback.call(messages, *args, **kwargs)
            except Exception as fallback_error:
                if isinstance(fallback_error, (ProviderContentPolicyError, json.JSONDecodeError)) or is_gateway_refusal(fallback_error):
                    raise error from fallback_error
                raise

    async def supports_batch_api(self) -> bool:
        return False

    def supports_vision(self) -> bool | None:
        return self.primary.supports_vision() and self.fallback.supports_vision()

    async def cleanup(self) -> None:
        await self.primary.cleanup()
        await self.fallback.cleanup()


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
    """The provider that reaches ``model`` through the gateway at ``base_url``.

    With ``HINDSIGHT_API_BROKER_REFUSAL_FALLBACK_MODEL`` set, a call the model refuses is answered by
    that model instead (see :class:`RefusalFallbackLLM`).
    """
    kwargs = dict(
        api_key=api_key,
        base_url=base_url,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
        extra_body=extra_body,
        default_headers=default_headers,
        cache_affinity=cache_affinity,
        openai_service_tier=openai_service_tier,
    )
    primary = _create_broker_model(model=model, **kwargs)
    fallback_model = get_config().broker_refusal_fallback_model
    if not fallback_model or fallback_model == model:
        return primary
    return RefusalFallbackLLM(primary, _create_broker_model(model=fallback_model, **kwargs))


def _create_broker_model(
    *,
    api_key: str | None,
    base_url: str,
    model: str,
    reasoning_effort: str | None,
    timeout: float | None,
    extra_body: dict[str, Any] | None,
    default_headers: dict[str, str] | None,
    cache_affinity: str | None,
    openai_service_tier: str | None,
) -> LLMInterface:
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
