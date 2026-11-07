# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/schemas/api.py
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/schemas/configuration.py
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/schemas/internal.py
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/utils/config_helpers.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""Pydantic schemas for the people package: configuration, conclusions in flight, peers, messages,
sessions and the people API.

Configuration and conclusion schemas follow Honcho's ``schemas/configuration.py`` and
``schemas/internal.py``; the API schemas follow ``schemas/api.py`` with ``workspace`` renamed to
``bank`` and the scope, tenant and webhook parts left out.
"""

import datetime
from enum import Enum
from typing import Annotated, Any, Literal, Self, cast

from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field, field_validator, model_validator

from .sanitization import NulStripped, strip_nul

# Module constants standing in for Honcho's config.toml settings.
MAX_CUSTOM_INSTRUCTIONS_TOKENS = 2000
DEFAULT_REASONING_ENABLED = True  # DERIVER.ENABLED
DEFAULT_PEER_CARD_ENABLED = True  # PEER_CARD.ENABLED
DEFAULT_SUMMARY_ENABLED = True  # SUMMARY.ENABLED
DEFAULT_MESSAGES_PER_SHORT_SUMMARY = 20  # SUMMARY.MESSAGES_PER_SHORT_SUMMARY
DEFAULT_MESSAGES_PER_LONG_SUMMARY = 60  # SUMMARY.MESSAGES_PER_LONG_SUMMARY
DEFAULT_DREAM_ENABLED = True  # DREAM.ENABLED
WORKING_REPRESENTATION_MAX_OBSERVATIONS = 100  # DERIVER.WORKING_REPRESENTATION_MAX_OBSERVATIONS
MAX_MESSAGE_SIZE = 65535

# Person handles are looser than Honcho's resource names: a handle can carry dots, an at sign or a plus
# (an email local part, a username), so the pattern allows them.
RESOURCE_NAME_PATTERN = r"^[a-zA-Z0-9_.@+-]+$"

DocumentLevel = Literal["explicit", "deductive", "inductive", "contradiction"]

_METADATA_MAX_KEYS = 100
_METADATA_MAX_DEPTH = 5


def _estimate_tokens(text: str) -> int:
    """A cheap token estimate (four characters per token), enough to enforce a budget."""
    return (len(text) + 3) // 4


def _check_metadata_limits(
    data: dict[str, Any],
    *,
    _current_depth: int = 1,
) -> None:
    """Validate metadata dict doesn't exceed key count or nesting depth limits."""
    if _current_depth > _METADATA_MAX_DEPTH:
        raise ValueError(f"Metadata nesting exceeds maximum depth of {_METADATA_MAX_DEPTH}")
    if _current_depth == 1 and len(data) > _METADATA_MAX_KEYS:
        raise ValueError(f"Metadata exceeds maximum of {_METADATA_MAX_KEYS} top-level keys")
    for v in data.values():
        if isinstance(v, dict):
            _check_metadata_limits(cast(dict[str, Any], v), _current_depth=_current_depth + 1)


def _validate_metadata(v: Any) -> Any:
    """Validate and sanitize a metadata dict: enforce limits and strip NUL bytes."""
    if not isinstance(v, dict):
        return v
    data = cast(dict[str, Any], v)
    _check_metadata_limits(data)
    return strip_nul(data)


_SanitizedMetadata = Annotated[dict[str, Any], BeforeValidator(_validate_metadata)]


# ---------------------------------------------------------------------------
# Configuration schemas (Honcho schemas/configuration.py)
# ---------------------------------------------------------------------------


class DreamType(str, Enum):
    """Types of dreams that can be triggered."""

    OMNI = "omni"
    # Lightweight card-only refresh: runs a single specialist restricted to
    # peer-card tools. Used for event-driven refreshes (scope membership
    # changes, cold starts) - never creates or deletes observations.
    CARD_REFRESH = "card_refresh"


def _validate_custom_instructions_budget(
    custom_instructions: str | None,
) -> str | None:
    if custom_instructions is None:
        return None
    if not custom_instructions.strip():
        return custom_instructions

    max_tokens = MAX_CUSTOM_INSTRUCTIONS_TOKENS
    if max_tokens <= 0:
        raise ValueError("custom_instructions are not enabled for this deployment")

    if _estimate_tokens(custom_instructions) > max_tokens:
        raise ValueError(f"custom_instructions exceeds DERIVER.MAX_CUSTOM_INSTRUCTIONS_TOKENS ({max_tokens} tokens)")

    return custom_instructions


CustomInstructions = Annotated[str | None, AfterValidator(_validate_custom_instructions_budget)]


class ReasoningConfiguration(BaseModel):
    enabled: bool | None = Field(
        default=None,
        description="Whether to enable reasoning functionality.",
    )
    custom_instructions: CustomInstructions = Field(
        default=None,
        description="Custom instructions for the deriver. Overrides the top-level custom_instructions at the same level. Set to an empty string to explicitly use none.",
    )


class PeerCardConfiguration(BaseModel):
    use: bool | None = Field(
        default=None,
        description="Whether to use peer card related to this peer during reasoning process.",
    )
    create: bool | None = Field(
        default=None,
        description="Whether to generate peer card based on content.",
    )
    custom_instructions: CustomInstructions = Field(
        default=None,
        description="Custom instructions for peer card updates. Added on top of the dream instructions and applied within the peer card's allowed entry kinds. Does not fall back to the top-level custom_instructions.",
    )


class SummaryConfiguration(BaseModel):
    enabled: bool | None = Field(
        default=None,
        description="Whether to enable summary functionality.",
    )
    messages_per_short_summary: int | None = Field(
        default=None,
        ge=10,
        description="Number of messages per short summary. Must be positive, greater than or equal to 10, and less than messages_per_long_summary.",
    )
    messages_per_long_summary: int | None = Field(
        default=None,
        ge=20,
        description="Number of messages per long summary. Must be positive, greater than or equal to 20, and greater than messages_per_short_summary.",
    )
    custom_instructions: CustomInstructions = Field(
        default=None,
        description="Custom instructions for session summaries. Overrides the top-level custom_instructions at the same level. Set to an empty string to explicitly use none.",
    )

    @model_validator(mode="after")
    def validate_summary_thresholds(self) -> Self:
        """Validate that short summary threshold <= long summary threshold."""
        short = self.messages_per_short_summary
        long = self.messages_per_long_summary

        if short is not None and long is not None and short >= long:
            raise ValueError("messages_per_short_summary must be less than messages_per_long_summary")

        return self


class DreamConfiguration(BaseModel):
    enabled: bool | None = Field(
        default=None,
        description="Whether to enable dream functionality. If reasoning is disabled, dreams will also be disabled and this setting will be ignored.",
    )
    custom_instructions: CustomInstructions = Field(
        default=None,
        description="Custom instructions for dreams. Overrides the top-level custom_instructions at the same level. Set to an empty string to explicitly use none.",
    )


class DialecticConfiguration(BaseModel):
    custom_instructions: CustomInstructions = Field(
        default=None,
        description="Custom instructions for the dialectic chat endpoints. Overrides the top-level custom_instructions at the same level. Set to an empty string to explicitly use none.",
    )


class WorkspaceConfiguration(BaseModel):
    """
    The set of options that can be in a workspace DB-level configuration dictionary.

    All fields are optional. Session-level configuration overrides workspace-level configuration, which overrides global configuration.
    """

    model_config = ConfigDict(extra="allow")  # pyright: ignore

    # Extra keys are allowed and stored as JSONB, which rejects NUL anywhere
    @model_validator(mode="before")
    @classmethod
    def _strip_nul(cls, data: Any) -> Any:
        return strip_nul(data)

    custom_instructions: CustomInstructions = Field(
        default=None,
        description="Custom instructions shared by the deriver, summarizer, dialectic, and dreamer. A module's own custom_instructions at the same level takes precedence, and session-level values beat workspace-level ones. Messages can only override reasoning.custom_instructions.",
    )
    reasoning: ReasoningConfiguration | None = Field(
        default=None,
        description="Configuration for reasoning functionality.",
    )
    peer_card: PeerCardConfiguration | None = Field(
        default=None,
        description="Configuration for peer card functionality. If reasoning is disabled, peer cards will also be disabled and these settings will be ignored.",
    )
    summary: SummaryConfiguration | None = Field(
        default=None,
        description="Configuration for summary functionality.",
    )
    dream: DreamConfiguration | None = Field(
        default=None,
        description="Configuration for dream functionality. If reasoning is disabled, dreams will also be disabled and these settings will be ignored.",
    )
    dialectic: DialecticConfiguration | None = Field(
        default=None,
        description="Configuration for the dialectic chat endpoints.",
    )


class SessionConfiguration(WorkspaceConfiguration):
    """
    The set of options that can be in a session DB-level configuration dictionary.

    All fields are optional. Session-level configuration overrides workspace-level configuration, which overrides global configuration.
    """

    pass


class MessageConfiguration(BaseModel):
    """
    The set of options that can be in a message DB-level configuration dictionary.

    All fields are optional. Message-level configuration overrides all other configurations.
    """

    @model_validator(mode="before")
    @classmethod
    def _strip_nul(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        # Both models ignore extras. Do not recursively sanitize values that
        # will be discarded: a deeply nested unknown field must stay harmless.
        fields = {strip_nul(k): v for k, v in cast(dict[str, Any], data).items() if strip_nul(k) in cls.model_fields}
        reasoning = fields.get("reasoning")
        if isinstance(reasoning, dict):
            fields["reasoning"] = {
                strip_nul(k): strip_nul(v) if isinstance(v, str) else v
                for k, v in cast(dict[str, Any], reasoning).items()
                if strip_nul(k) in ReasoningConfiguration.model_fields
            }
        return fields

    reasoning: ReasoningConfiguration | None = Field(
        default=None,
        description="Configuration for reasoning functionality.",
    )


class ResolvedReasoningConfiguration(BaseModel):
    enabled: bool
    custom_instructions: CustomInstructions = None
    custom_instructions_source: str | None = None


class ResolvedPeerCardConfiguration(BaseModel):
    use: bool
    create: bool
    custom_instructions: CustomInstructions = None
    custom_instructions_source: str | None = None


class ResolvedSummaryConfiguration(BaseModel):
    enabled: bool
    messages_per_short_summary: int
    messages_per_long_summary: int
    custom_instructions: CustomInstructions = None
    custom_instructions_source: str | None = None


class ResolvedDreamConfiguration(BaseModel):
    enabled: bool
    custom_instructions: CustomInstructions = None
    custom_instructions_source: str | None = None


class ResolvedDialecticConfiguration(BaseModel):
    custom_instructions: CustomInstructions = None
    custom_instructions_source: str | None = None


class ResolvedConfiguration(BaseModel):
    """
    The final resolved configuration for a given message.
    Hierarchy: message > session > workspace > global configuration
    """

    reasoning: ResolvedReasoningConfiguration
    peer_card: ResolvedPeerCardConfiguration
    summary: ResolvedSummaryConfiguration
    dream: ResolvedDreamConfiguration
    dialectic: ResolvedDialecticConfiguration = Field(default_factory=ResolvedDialecticConfiguration)

    @model_validator(mode="before")
    @classmethod
    def migrate_deriver_to_reasoning(cls, data: Any) -> Any:
        """Handle v3.0.0 migration: 'deriver' was renamed to 'reasoning'."""
        if not isinstance(data, dict):
            return data

        config = cast(dict[str, Any], data)

        if "deriver" in config and "reasoning" not in config:
            config["reasoning"] = config.pop("deriver")

        return config


class PeerConfig(BaseModel):
    # TODO: Update description - should say "Whether honcho forms a representation of the peer itself"
    observe_me: bool | None = Field(
        default=None,
        description="Whether Honcho will use reasoning to form a representation of this peer",
    )


class SessionPeerConfig(PeerConfig):
    # TODO: Update description - should say "Whether this peer forms representations of other peers in the session"
    observe_others: bool | None = Field(
        default=None,
        description="Whether this peer should form a session-level theory-of-mind representation of other peers in the session",
    )


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> None:
    """Recursive update of a dictionary. Skips None values in the update dictionary."""
    for key, value in update.items():
        if value is None:
            continue
        existing = base.get(key)
        if isinstance(value, dict) and isinstance(existing, dict):
            _deep_update(cast(dict[str, Any], existing), cast(dict[str, Any], value))
        else:
            base[key] = value


# Modules whose custom_instructions fall back to the top-level shared value.
# peer_card is resolved separately: its instructions are additive on top of the
# dream instructions, so falling back would inject the shared value twice.
_SHARED_INSTRUCTION_MODULES = ("reasoning", "summary", "dialectic", "dream")


def _resolve_custom_instructions(
    levels: list[tuple[str, dict[str, Any]]],
    module: str,
    *,
    shared_fallback: bool = True,
) -> tuple[str | None, str | None]:
    """Resolve a module's custom instructions across configuration levels.

    The most specific level that sets a value wins. Within one level, the module's own
    ``custom_instructions`` beats the top-level shared value. An empty string is an explicit
    "none" and stops the fallback. Returns (stripped instructions or None, source such as
    "session.shared" or "workspace.summary", or None when no level set a value).
    """
    for level_name, level in levels:
        module_cfg = level.get(module)
        value = module_cfg.get("custom_instructions") if isinstance(module_cfg, dict) else None
        source = f"{level_name}.{module}"
        if value is None and shared_fallback:
            value = level.get("custom_instructions")
            source = f"{level_name}.shared"
        if isinstance(value, str):
            return value.strip() or None, source
    return None, None


def normalize_configuration_dict(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize a bank/session/message configuration dict to match ``ResolvedConfiguration``.

    Preserves backwards compatibility with the legacy keys ``deriver.enabled`` and ``skip_deriver``:
    ``reasoning.enabled`` is derived from them when it is not set explicitly, and the legacy keys are
    removed from the returned dict.
    """
    normalized: dict[str, Any] = dict(raw)

    reasoning_present = "reasoning" in normalized
    raw_reasoning = normalized.get("reasoning")
    reasoning: dict[str, Any] = dict(raw_reasoning) if isinstance(raw_reasoning, dict) else {}
    reasoning_enabled_explicit = reasoning.get("enabled") is not None

    if not reasoning_enabled_explicit:
        raw_deriver = normalized.get("deriver")
        deriver: dict[str, Any] = dict(raw_deriver) if isinstance(raw_deriver, dict) else {}
        if deriver.get("enabled") is not None:
            reasoning["enabled"] = bool(deriver["enabled"])

    if not reasoning_enabled_explicit and normalized.get("skip_deriver") is True:
        reasoning["enabled"] = False

    if reasoning_present or reasoning:
        normalized["reasoning"] = reasoning

    normalized.pop("deriver", None)
    normalized.pop("skip_deriver", None)

    return normalized


def get_configuration(
    message_configuration: MessageConfiguration | None,
    session_configuration: dict[str, Any] | None = None,
    workspace_configuration: dict[str, Any] | None = None,
) -> ResolvedConfiguration:
    """Resolve configuration with hierarchical fallback.

    Resolution hierarchy: message configuration, session configuration, bank configuration, then the
    module defaults above. Honcho passes the session and workspace rows; here the callers pass the
    ``configuration`` dicts (``None`` for no configuration at that level).
    """
    config_dict: dict[str, Any] = {
        "reasoning": {
            "enabled": DEFAULT_REASONING_ENABLED,
            "custom_instructions": None,
        },
        "peer_card": {
            "use": DEFAULT_PEER_CARD_ENABLED,
            "create": DEFAULT_PEER_CARD_ENABLED,
        },
        "summary": {
            "enabled": DEFAULT_SUMMARY_ENABLED,
            "messages_per_short_summary": DEFAULT_MESSAGES_PER_SHORT_SUMMARY,
            "messages_per_long_summary": DEFAULT_MESSAGES_PER_LONG_SUMMARY,
        },
        "dream": {"enabled": DEFAULT_DREAM_ENABLED},
        "dialectic": {},
    }

    # Most specific first: Message -> Session -> Bank
    levels = [
        (name, normalize_configuration_dict(raw))
        for name, raw in (
            (
                "message",
                message_configuration.model_dump(exclude_none=True) if message_configuration is not None else None,
            ),
            ("session", session_configuration),
            ("workspace", workspace_configuration),
        )
        if raw is not None
    ]

    # Apply overrides least specific first; _deep_update modifies config_dict in place
    for _, level in reversed(levels):
        _deep_update(config_dict, level)

    # custom_instructions don't merge field-by-field: a more specific level's
    # shared value must beat a less specific level's module value.
    for module in (*_SHARED_INSTRUCTION_MODULES, "peer_card"):
        value, source = _resolve_custom_instructions(levels, module, shared_fallback=module != "peer_card")
        config_dict[module]["custom_instructions"] = value
        config_dict[module]["custom_instructions_source"] = source

    return ResolvedConfiguration(**config_dict)


# ---------------------------------------------------------------------------
# Conclusions in flight (Honcho schemas/internal.py: document schemas)
# ---------------------------------------------------------------------------


class DocumentBase(BaseModel):
    pass


class DocumentMetadata(BaseModel):
    message_ids: list[int] = Field(
        description="The ID range(s) of the messages that this document was derived from. Acts as a link to the primary source of the document. Note that as a document gets deduplicated, additional ranges will be added, because the same document could be derived from completely separate message ranges."
    )
    message_created_at: str = Field(
        description="The timestamp of the message that this document was derived from. Note that this is not the same as the created_at timestamp of the document. This timestamp is usually only saved with second-level precision."
    )
    source_ids: list[str] | None = Field(
        default=None,
        description="Document IDs of source documents for tree traversal -- required for deductive and inductive documents",
    )
    premises: list[str] | None = Field(
        default=None,
        description="Human-readable premise text for display -- only applicable for deductive documents",
    )
    sources: list[str] | None = Field(
        default=None,
        description="Human-readable source text for display -- only applicable for inductive documents",
    )
    pattern_type: str | None = Field(
        default=None,
        description="Type of pattern identified (preference, behavior, personality, tendency, correlation) -- only applicable for inductive documents",
    )
    confidence: str | None = Field(
        default=None,
        description="Confidence level (high, medium, low) -- only applicable for inductive documents",
    )


class DocumentCreate(DocumentBase):
    content: Annotated[str, Field(min_length=1, max_length=100000), NulStripped]
    session_name: str | None = Field(
        default=None,
        description="The session from which the document was derived (NULL for global observations)",
    )
    level: DocumentLevel = Field(
        default="explicit",
        description="The level of the document (explicit, deductive, inductive, or contradiction)",
    )
    times_derived: int = Field(
        default=1,
        ge=1,
        description="The number of times that a semantic duplicate document to this one has been derived",
    )
    metadata: DocumentMetadata = Field()
    embedding: list[float] = Field()
    # Tree linkage field
    source_ids: list[str] | None = Field(
        default=None,
        description="Document IDs of source/premise documents -- for deductive and inductive documents",
    )
    source_message_ids: list[str] | None = Field(
        default=None,
        description="Public IDs of the messages cited as evidence -- for explicit documents",
    )
    source_url: str | None = Field(
        default=None,
        description=(
            "Where the evidence was said. Filled from the first cited message (or, for derived levels, the first "
            "premise that has one) when left empty."
        ),
    )
    said_at: datetime.datetime | None = Field(
        default=None,
        description="When the evidence was said. Filled together with source_url when left empty.",
    )


class ObservationInput(BaseModel):
    """Validated observation input from LLM tool calls."""

    content: Annotated[str, Field(min_length=1), NulStripped]
    level: DocumentLevel = "explicit"
    source_ids: list[str] | None = None
    premises: list[str] | None = None
    sources: list[str] | None = None
    pattern_type: Literal["preference", "behavior", "personality", "tendency", "correlation"] | None = None
    confidence: Literal["high", "medium", "low"] | None = None

    @model_validator(mode="after")
    def validate_level_fields(self) -> Self:
        """Validate that level-specific fields are present when required."""
        if self.level == "deductive" and not self.source_ids:
            raise ValueError("deductive observations require 'source_ids' field with document IDs of premises")
        if self.level == "inductive" and not self.source_ids:
            raise ValueError("inductive observations require 'source_ids' field with document IDs of sources")
        if self.level == "contradiction" and (not self.source_ids or len(self.source_ids) < 2):
            raise ValueError(
                "contradiction observations require 'source_ids' field with at least 2 IDs of contradicting observations"
            )
        return self


# ---------------------------------------------------------------------------
# API schemas (Honcho schemas/api.py: peers, messages, sessions)
# ---------------------------------------------------------------------------


class PeerBase(BaseModel):
    pass


class PeerSpec(PeerBase):
    """Peer identity plus optional updates, for callers that already have a handle.

    ``PeerCreate`` narrows ``name`` with ``pattern=RESOURCE_NAME_PATTERN`` because it validates a
    *new, user-supplied* person handle at the API boundary. Code paths reach ``get_or_create_peers``
    with handles that already exist: re-validating those turns a lookup into a raw validation error.
    Carries **no** constraints at all, deliberately; every rule for a *new* name lives in
    ``peers._validate_new_peer_names``, which runs on the insert path only.
    """

    name: str
    metadata: _SanitizedMetadata | None = None
    configuration: Annotated[dict[str, Any], NulStripped] | None = None


class PeerCreate(PeerSpec):
    name: Annotated[
        str,
        Field(alias="id", min_length=1, max_length=512, pattern=RESOURCE_NAME_PATTERN),
    ]

    model_config = ConfigDict(populate_by_name=True)  # pyright: ignore


class PeerGet(PeerBase):
    filters: dict[str, Any] | None = None


class PeerUpdate(PeerBase):
    metadata: _SanitizedMetadata | None = None
    configuration: Annotated[dict[str, Any], NulStripped] | None = None


class Peer(PeerBase):
    name: str = Field(serialization_alias="id")
    id: str = Field(
        serialization_alias="uid",
        description=(
            "Immutable identifier assigned to this peer. `id` is the handle the client chose; `uid` is generated "
            "by the server."
        ),
    )
    bank_id: str
    created_at: datetime.datetime
    h_metadata: dict[str, Any] = Field(default_factory=dict, serialization_alias="metadata")
    internal_metadata: dict[str, Any] = Field(default_factory=dict, exclude=True)
    configuration: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)  # pyright: ignore


class PeerRepresentationGet(BaseModel):
    session_id: str | None = Field(None, description="Optional session ID within which to scope the representation")
    filters: dict[str, Any] | None = Field(
        None,
        description=(
            "Optional filters to scope the representation. This endpoint supports only the 'session_id' key: "
            'a session id, a list of session ids, or {"in": [...]}. When session_id is also set, it must be '
            "included in the allowlist."
        ),
    )
    target: str | None = Field(
        None,
        description="Optional peer ID to get the representation for, from the perspective of this peer",
    )
    search_query: str | None = Field(
        None,
        description="Optional input to curate the representation around semantic search results",
    )
    search_top_k: int | None = Field(
        None,
        ge=1,
        le=100,
        description=(
            "Only used if `search_query` is provided. Number of semantic-search-retrieved conclusions to include "
            "in the representation"
        ),
    )
    search_max_distance: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Only used if `search_query` is provided. Maximum distance to search for semantically relevant conclusions"
        ),
    )
    include_most_frequent: bool | None = Field(
        default=None,
        description=(
            "Only used if `search_query` is provided. Whether to include the most frequent conclusions in the "
            "representation"
        ),
    )
    max_conclusions: int | None = Field(
        default=25,
        ge=1,
        le=100,
        description=(
            "Only used if `search_query` is provided. Maximum number of conclusions to include in the representation"
        ),
    )


class RepresentationResponse(BaseModel):
    representation: str


class PeerCardResponse(BaseModel):
    peer_card: list[str] | None = Field(None, description="The peer card content, or None if not found")


class PeerCardSet(BaseModel):
    peer_card: list[str] = Field(..., description="The peer card content to set")

    @field_validator("peer_card", mode="before")
    @classmethod
    def sanitize_peer_card(cls, v: Any) -> Any:
        if isinstance(v, list):
            return [strip_nul(item) if isinstance(item, str) else item for item in cast(list[Any], v)]
        return v


class Message(BaseModel):
    """A message as the API lists it. ``source_url``, ``said_by``, ``said_at`` and ``channel`` say where it came from."""

    public_id: str = Field(serialization_alias="id")
    content: str
    peer_name: str = Field(serialization_alias="peer_id")
    session_name: str = Field(serialization_alias="session_id")
    h_metadata: dict[str, Any] = Field(default_factory=dict, serialization_alias="metadata")
    created_at: datetime.datetime
    bank_id: str
    token_count: int
    source_url: str | None = None
    said_by: str | None = None
    said_at: datetime.datetime | None = None
    channel: str | None = None

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)  # pyright: ignore


class SessionGet(BaseModel):
    filters: dict[str, Any] | None = None


class Session(BaseModel):
    name: str = Field(serialization_alias="id")
    id: str = Field(
        serialization_alias="uid",
        description=(
            "Immutable identifier assigned to this session. `id` is the name the client chose; `uid` is generated "
            "by the server."
        ),
    )
    is_active: bool
    bank_id: str
    h_metadata: dict[str, Any] = Field(default_factory=dict, serialization_alias="metadata")
    configuration: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime.datetime

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)  # pyright: ignore


class PeerContext(BaseModel):
    """Context for a peer, including representation and peer card."""

    peer_id: str = Field(description="The handle of the peer")
    target_id: str = Field(description="The handle of the target peer being observed")
    representation: str | None = Field(
        default=None,
        description="A curated subset of the representation of the target peer from the observer's perspective",
    )
    peer_card: list[str] | None = Field(
        default=None,
        description="The peer card for the target peer from the observer's perspective",
    )


class MessageSearchOptions(BaseModel):
    query: Annotated[str, Field(..., description="Search query"), NulStripped]
    filters: dict[str, Any] | None = Field(default=None, description="Filters to scope the search")
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results to return",
    )
