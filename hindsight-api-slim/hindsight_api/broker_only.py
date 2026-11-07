"""The broker-only profile: every model call runs on a subscription login the auth broker holds.

Set ``HINDSIGHT_API_BROKER_ONLY=true`` and the configuration is refused at startup (and when a bank
override is validated) unless:

* every language-model provider the server can call, for any operation, is ``broker``: the provider
  that sends the call to an ``omp auth-gateway`` (which signs it with the Claude or Codex login the
  auth broker holds), never to a vendor with an API key;
* the person-model, dialectic and refusal-fallback models, when set, are broker models too;
* every ``broker`` model is an Anthropic or Codex model, named ``anthropic/<model>`` or
  ``openai-codex/<model>`` (the gateway's own model ids);
* every ``broker`` base URL is the gateway: loopback, a private address, or a tailnet name or address;
* embeddings and the reranker run on this machine (``local``, ``onnx``, ``flashrank``, ``jina-mlx``,
  ``rrf``, ``none``; ``tei`` only when its URL is loopback or a private address);
* nothing else dials out by configuration: no decision model, no global webhook, no remote OTLP
  exporter, no hosted file parser.

``violations`` returns the reasons so a test (or an operator) can list them all at once.
"""

from __future__ import annotations

import ipaddress
from typing import Any
from urllib.parse import urlparse

BROKER_LLM_PROVIDER = "broker"
# ``none`` turns a model-backed operation off, so it calls nobody.
ALLOWED_LLM_PROVIDERS = frozenset({BROKER_LLM_PROVIDER, "none", "mock"})
# The gateway's model ids: ``<provider>/<model>``. These are the two subscription families the broker holds.
BROKER_MODEL_PREFIXES = ("anthropic/", "openai-codex/")
LOCAL_EMBEDDINGS_PROVIDERS = frozenset({"local", "onnx"})
LOCAL_RERANKER_PROVIDERS = frozenset({"local", "onnx", "flashrank", "jina-mlx", "rrf", "none"})
# A service that runs on this machine or inside our own network, reached by URL.
SELF_HOSTED_PROVIDERS = frozenset({"tei"})


def is_broker_model(model: str | None) -> bool:
    """True for ``anthropic/<model>`` and ``openai-codex/<model>``: a prefix followed by a model name."""
    return bool(model) and any(model.startswith(p) and len(model) > len(p) for p in BROKER_MODEL_PREFIXES)


# (provider field, model field, base URL field) of each operation's model. An operation that sets none
# of them uses the global ``llm_*`` triple.
_LLM_SLOTS = (
    ("llm_provider", "llm_model", "llm_base_url"),
    ("retain_llm_provider", "retain_llm_model", "retain_llm_base_url"),
    ("vlm_provider", "vlm_model", "vlm_base_url"),
    ("reflect_llm_provider", "reflect_llm_model", "reflect_llm_base_url"),
    ("consolidation_llm_provider", "consolidation_llm_model", "consolidation_llm_base_url"),
    (
        "mental_model_refresh_llm_provider",
        "mental_model_refresh_llm_model",
        "mental_model_refresh_llm_base_url",
    ),
)
# Models named on their own, which run on the provider of the operation they belong to.
_EXTRA_MODEL_FIELDS = ("people_llm_model", "dialectic_llm_model", "broker_refusal_fallback_model")
# Decision models are hosted services (Hindsight Inference, TypeSafe, Cloudflare, OpenAI); only "none" passes.
_DECISION_MODEL_PROVIDER_FIELDS = (
    "decision_model_provider",
    "recall_decision_model_provider",
    "reflect_decision_model_provider",
    "consolidation_decision_model_provider",
)
_LLM_MEMBER_FIELDS = (
    "llm_members",
    "retain_llm_members",
    "reflect_llm_members",
    "consolidation_llm_members",
    "mental_model_refresh_llm_members",
)
_TAILSCALE_CGNAT = ipaddress.ip_network("100.64.0.0/10")


def _is_private_url(url: str | None) -> bool:
    """True when ``url`` names this machine or a private network address."""
    if not url:
        return False
    host = urlparse(url if "://" in url else f"http://{url}").hostname or ""
    if host in ("localhost", "host.docker.internal") or host.endswith(".internal") or host.endswith(".local"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or address.is_private


def _is_gateway_url(url: str | None) -> bool:
    """True when ``url`` can be an auth gateway: this machine, a private address, or the tailnet."""
    if not url:
        return False
    if _is_private_url(url):
        return True
    host = urlparse(url if "://" in url else f"http://{url}").hostname or ""
    if host.endswith(".ts.net"):
        return True
    try:
        return ipaddress.ip_address(host) in _TAILSCALE_CGNAT
    except ValueError:
        return False


def _broker_slot_problems(label: str, model: str | None, base_url: str | None) -> list[str]:
    problems: list[str] = []
    if not is_broker_model(model):
        problems.append(
            f"{label} model {model!r} is not a broker model (use 'anthropic/<model>' or 'openai-codex/<model>')"
        )
    if not _is_gateway_url(base_url):
        problems.append(f"{label} base URL {base_url!r} is not an auth gateway (loopback, private or tailnet address)")
    return problems


def _embedding_sides(config: Any) -> list[tuple[str, str, str | None]]:
    """(label, provider, base_url) for every embeddings provider the config builds."""
    sides = [("embeddings", getattr(config, "embeddings_provider", ""), getattr(config, "embeddings_tei_url", None))]
    for label in ("query_embeddings", "passage_embeddings"):
        side = getattr(config, label, None)
        if side is not None:
            sides.append((label, getattr(side, "embeddings_provider", ""), getattr(side, "embeddings_tei_url", None)))
    return sides


def violations(config: Any) -> list[str]:
    """Every way ``config`` would call a model that is not a broker-held Anthropic or Codex login."""
    found: list[str] = []

    global_provider = getattr(config, "llm_provider", None)
    global_model = getattr(config, "llm_model", None)
    global_base_url = getattr(config, "llm_base_url", None)
    for provider_field, model_field, base_url_field in _LLM_SLOTS:
        own_provider = getattr(config, provider_field, None)
        provider = own_provider or global_provider
        if not provider:
            continue
        if provider not in ALLOWED_LLM_PROVIDERS:
            if own_provider:
                found.append(f"{provider_field}={provider!r} is not the broker (use 'broker')")
            continue
        if provider == BROKER_LLM_PROVIDER:
            # A slot that only overrides the model or URL still runs on the global broker provider.
            model = getattr(config, model_field, None) or global_model
            base_url = getattr(config, base_url_field, None) or global_base_url
            found.extend(_broker_slot_problems(provider_field, model, base_url))
    for name in _EXTRA_MODEL_FIELDS:
        model = getattr(config, name, None)
        if model and not is_broker_model(model):
            found.append(f"{name} {model!r} is not a broker model (use 'anthropic/<model>' or 'openai-codex/<model>')")
    for name in _DECISION_MODEL_PROVIDER_FIELDS:
        provider = getattr(config, name, None)
        if provider and provider != "none":
            found.append(f"{name}={provider!r} is a hosted decision model (use 'none')")
    for name in _LLM_MEMBER_FIELDS:
        for member in getattr(config, name, None) or []:
            provider = getattr(member, "provider", None)
            if provider and provider not in ALLOWED_LLM_PROVIDERS:
                found.append(f"{name} member provider {provider!r} is not the broker")
            elif provider == BROKER_LLM_PROVIDER:
                found.extend(
                    _broker_slot_problems(
                        f"{name} member",
                        getattr(member, "model", None),
                        getattr(member, "base_url", None) or global_base_url,
                    )
                )

    for label, provider, url in _embedding_sides(config):
        if not provider:
            continue
        if provider in LOCAL_EMBEDDINGS_PROVIDERS:
            continue
        if provider in SELF_HOSTED_PROVIDERS and _is_private_url(url):
            continue
        found.append(f"{label} provider {provider!r} does not run on this machine (use 'local' or 'onnx')")

    reranker = getattr(config, "reranker_provider", None)
    if reranker and reranker not in LOCAL_RERANKER_PROVIDERS:
        if not (reranker in SELF_HOSTED_PROVIDERS and _is_private_url(getattr(config, "reranker_tei_url", None))):
            found.append(f"reranker_provider={reranker!r} does not run on this machine")

    if getattr(config, "webhook_url", None):
        found.append("webhook_url is set: the server would call a third party")
    endpoint = getattr(config, "otel_exporter_otlp_endpoint", None)
    if getattr(config, "otel_traces_enabled", False) and endpoint and not _is_private_url(endpoint):
        found.append("otel_exporter_otlp_endpoint is a remote host")
    if getattr(config, "file_parser_iris_token", None):
        found.append("file_parser_iris_token is set: the iris parser is a hosted service")
    if getattr(config, "file_parser_llama_parse_api_key", None):
        found.append("file_parser_llama_parse_api_key is set: llama_parse is a hosted service")
    if getattr(config, "file_parser_markitdown_ocr_api_key", None) and not _is_private_url(
        getattr(config, "file_parser_markitdown_ocr_base_url", None)
    ):
        found.append("the markitdown OCR model is a hosted service")
    return found


def enforce(config: Any) -> None:
    """Raise ``ValueError`` listing every violation when the profile is on and the config breaks it."""
    if not getattr(config, "broker_only", False):
        return
    problems = violations(config)
    if problems:
        raise ValueError(
            "HINDSIGHT_API_BROKER_ONLY is on, but the configuration would call a model or service that is not "
            "a broker-held Anthropic or Codex login with local embeddings:\n  - " + "\n  - ".join(problems)
        )
