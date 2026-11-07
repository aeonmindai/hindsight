"""The Claude-only profile: the server talks to no third party except Anthropic.

Set ``HINDSIGHT_API_CLAUDE_ONLY=true`` and the configuration is refused at startup (and when a bank
override is validated) unless:

* every language-model provider the server can call, for any operation, is Anthropic (``anthropic``,
  the API, or ``claude-code``, the Claude Agent SDK signed in as the founder);
* embeddings and the reranker run on this machine (``local``, ``onnx``, ``flashrank``, ``jina-mlx``,
  ``rrf``, ``none``; ``tei`` only when its URL is loopback or a private address);
* nothing else dials out by configuration: no global webhook, no remote OTLP exporter, no hosted file
  parser.

``violations`` returns the reasons so a test (or an operator) can list them all at once.
"""

from __future__ import annotations

import ipaddress
from typing import Any
from urllib.parse import urlparse

CLAUDE_LLM_PROVIDERS = frozenset({"anthropic", "claude-code"})
# ``none`` turns a model-backed operation off, so it calls nobody.
ALLOWED_LLM_PROVIDERS = CLAUDE_LLM_PROVIDERS | {"none", "mock"}
LOCAL_EMBEDDINGS_PROVIDERS = frozenset({"local", "onnx"})
LOCAL_RERANKER_PROVIDERS = frozenset({"local", "onnx", "flashrank", "jina-mlx", "rrf", "none"})
# A service that runs on this machine or inside our own network, reached by URL.
SELF_HOSTED_PROVIDERS = frozenset({"tei"})

_LLM_PROVIDER_FIELDS = (
    "llm_provider",
    "retain_llm_provider",
    "vlm_provider",
    "reflect_llm_provider",
    "consolidation_llm_provider",
    "mental_model_refresh_llm_provider",
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


def _embedding_sides(config: Any) -> list[tuple[str, str, str | None]]:
    """(label, provider, base_url) for every embeddings provider the config builds."""
    sides = [("embeddings", getattr(config, "embeddings_provider", ""), getattr(config, "embeddings_tei_url", None))]
    for label in ("query_embeddings", "passage_embeddings"):
        side = getattr(config, label, None)
        if side is not None:
            sides.append((label, getattr(side, "embeddings_provider", ""), getattr(side, "embeddings_tei_url", None)))
    return sides


def violations(config: Any) -> list[str]:
    """Every way ``config`` would reach a third party other than Anthropic."""
    found: list[str] = []

    for name in _LLM_PROVIDER_FIELDS:
        provider = getattr(config, name, None)
        if provider and provider not in ALLOWED_LLM_PROVIDERS:
            found.append(f"{name}={provider!r} is not Anthropic (use 'anthropic' or 'claude-code')")
    for name in _LLM_MEMBER_FIELDS:
        for member in getattr(config, name, None) or []:
            provider = getattr(member, "provider", None)
            if provider and provider not in ALLOWED_LLM_PROVIDERS:
                found.append(f"{name} member provider {provider!r} is not Anthropic")

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
    if not getattr(config, "claude_only", False):
        return
    problems = violations(config)
    if problems:
        raise ValueError(
            "HINDSIGHT_API_CLAUDE_ONLY is on, but the configuration would reach a third party other than "
            "Anthropic:\n  - " + "\n  - ".join(problems)
        )
