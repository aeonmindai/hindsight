# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/utils/agent_tools.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""Tools for the dream specialists: search, read, create, delete conclusions and update the peer card.

Honcho's ``utils/agent_tools.py`` serves the deriver, the dreamer and the dialectic. This file keeps the
dreamer's tools (deduction, induction and card refresh). They run on ``PeopleContext`` and asyncpg; every
write that has a dream history target records its change in the same transaction as the change.
"""

from __future__ import annotations

import asyncio
import logging
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

from pydantic import ValidationError

from .. import conclusions as conclusions_crud
from .. import peer_card as peer_card_crud
from .. import schemas
from ..context import PeopleContext
from ..dream_history import (
    CreatedConclusion,
    DreamHistoryTarget,
    RemovedConclusion,
    record_created,
    record_removed,
)
from ..representation_utils import Representation
from ..sanitization import strip_nul
from ..schemas import ResolvedConfiguration
from .tool_loop import ToolResult

logger = logging.getLogger(__name__)

# Default for Honcho's settings.LLM.MAX_TOOL_OUTPUT_CHARS and DERIVER.DEDUPLICATE.
MAX_TOOL_OUTPUT_CHARS = 10000
DEDUPLICATE = True

# Hard cap to prevent unbounded peer card growth from repeated agent updates.
MAX_PEER_CARD_FACTS = 40

# Identity-marker prefixes allowed on the peer card. Anything else is rejected
# structurally — see `_validate_peer_card_entry`.
PEER_CARD_ALLOWED_PREFIXES: tuple[str, ...] = (
    "IDENTITY:",
    "ATTRIBUTE:",
    "RELATIONSHIP:",
    "INSTRUCTION:",
)

# Per-entry character cap to block evidence-bundle dumps and runaway lines.
MAX_PEER_CARD_ENTRY_LENGTH = 200


def _validate_peer_card_entry(line: str) -> bool:
    """Structural validation for a single peer card entry.

    Returns True when the line starts with one of the allowed prefixes followed
    by a space, has a non-empty body after the prefix, and fits within the per-
    entry length cap. Subject-substance correctness (is this actually about the
    observed peer?) is left to the prompt — this is form-only.
    """
    if not line or len(line) > MAX_PEER_CARD_ENTRY_LENGTH:
        return False
    for prefix in PEER_CARD_ALLOWED_PREFIXES:
        prefix_with_space = f"{prefix} "
        if line.startswith(prefix_with_space):
            body = line[len(prefix_with_space) :].strip()
            return bool(body)
    return False


def _normalized_observation_input(
    obs: schemas.ObservationInput,
) -> schemas.ObservationInput:
    """Return an observation input with content normalized for persistence/embedding.

    NUL bytes are removed here rather than closer to the database so that the
    text that gets embedded is the same text that gets stored. `premises` and
    `sources` ride along in internal_metadata, and jsonb rejects NUL in strings
    just as text columns do.
    """
    return obs.model_copy(
        update={
            "content": strip_nul(obs.content).strip(),
            "premises": strip_nul(obs.premises),
            "sources": strip_nul(obs.sources),
        }
    )


def _deductive_observation_item_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The deductive conclusion as a self-contained statement",
            },
            "source_ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "description": "Required non-empty list of source observation IDs supporting the deduction. Copy the exact ID shown in [id:xxx] format from observation results; message search results carry no ID and cannot be cited",
            },
            "premises": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "description": "Required human-readable premise text matching the source observations",
            },
        },
        "required": ["content", "source_ids", "premises"],
        "additionalProperties": False,
    }


def _inductive_observation_item_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The inductive pattern or generalization as a self-contained statement",
            },
            "source_ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "description": "Required list of at least two source observation IDs supporting the pattern. Copy the exact ID shown in [id:xxx] format from observation results; message search results carry no ID and cannot be cited",
            },
            "sources": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "description": "Required human-readable evidence text matching the source observations",
            },
            "pattern_type": {
                "type": "string",
                "enum": [
                    "preference",
                    "behavior",
                    "personality",
                    "tendency",
                    "correlation",
                ],
                "description": "Required pattern category",
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "Required confidence level based on evidence count",
            },
        },
        "required": ["content", "source_ids", "sources", "pattern_type", "confidence"],
        "additionalProperties": False,
    }


def _safe_int(value: Any, default: int) -> int:
    """Coerce a tool input value to int, returning default on failure.

    LLMs sometimes pass non-numeric strings (e.g. 'Infinity') for integer
    parameters which would crash ``min()`` comparisons.
    """
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _bounded_int(value: Any, default: int, *, lo: int = 1, hi: int) -> int:
    """Coerce a tool int into ``[lo, hi]``, falling back to ``default`` on bad input."""
    return max(lo, min(_safe_int(value, default), hi))


# Module-level lock registry for thread-safe observation creation.
# Keyed by (workspace_name, observer, observed) to ensure all tool executors
# operating on the same data share the same lock.
#
# Uses WeakValueDictionary so entries are automatically removed when no
# ToolContext holds a reference to the lock (i.e., all executors for that
# key have finished and been garbage collected). This prevents unbounded
# growth over the lifetime of a long-running deriver process.
_observation_locks: weakref.WeakValueDictionary[tuple[str, str, str], asyncio.Lock] = weakref.WeakValueDictionary()
_registry_lock = asyncio.Lock()


# Module-level lock registry for serializing observation writes.
# Keyed by (bank_id, observer, observed) so all tool executors operating on the same data share a lock.
# A WeakValueDictionary drops entries once no ToolContext holds the lock.
_observation_locks: weakref.WeakValueDictionary[tuple[str, str, str], asyncio.Lock] = weakref.WeakValueDictionary()
_registry_lock = asyncio.Lock()


async def get_observation_lock(bank_id: str, observer: str, observed: str) -> asyncio.Lock:
    """The lock shared by all executors for this (bank, observer, observed)."""
    key = (bank_id, observer, observed)
    async with _registry_lock:
        lock = _observation_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            _observation_locks[key] = lock
        return lock


@dataclass
class ObservationFailure:
    """Records a single observation that failed during creation."""

    content_preview: str
    error: str


@dataclass
class ObservationsCreatedResult:
    """Result of a batch create_observations call."""

    created_count: int
    created_levels: list[str]
    failed: list[ObservationFailure]


def _truncation_metadata(was_truncated: bool, original_chars: int) -> dict[str, Any]:
    """``ToolResult.metadata`` fields that report truncation; empty when the output was not clamped."""
    if not was_truncated:
        return {}
    return {"was_truncated": True, "result_chars_before_truncation": original_chars}


TOOLS: dict[str, dict[str, Any]] = {
    "create_observations_deductive": {
        "name": "create_observations_deductive",
        "description": "Create new deductive observations discovered while answering the query. Every observation must include non-empty source_ids and premise text. Use this only for novel deductions grounded in existing observations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "observations": {
                    "type": "array",
                    "description": "List of new deductive observations to create",
                    "items": _deductive_observation_item_schema(),
                },
            },
            "required": ["observations"],
        },
    },
    "create_observations_inductive": {
        "name": "create_observations_inductive",
        "description": "Create new inductive observations discovered while answering the query. Every observation must include source_ids, source text, pattern_type, and confidence. Use this only for patterns supported by multiple observations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "observations": {
                    "type": "array",
                    "description": "List of new inductive observations to create",
                    "items": _inductive_observation_item_schema(),
                },
            },
            "required": ["observations"],
        },
    },
    "update_peer_card": {
        "name": "update_peer_card",
        "description": (
            "Update the peer card with stable identity markers about the observed peer. "
            "An identity marker distinguishes the peer from others of its kind and persists across interactions. "
            "The peer may be any entity with identity that changes over time (human, agent, codebase, team, organization) — do not assume the peer is human. "
            "Each entry must start with one of four prefixes: `IDENTITY:` (canonical name, kind, aliases, IDs), "
            "`ATTRIBUTE:` (stable durable property, including explicitly stated standing preferences), "
            "`RELATIONSHIP:` (durable link to another entity), or "
            "`INSTRUCTION:` (standing rule of engagement the peer has explicitly stated). "
            "Do not write `TRAIT:` or behavioral `PREFERENCE:` entries, one-off observations, transient state, "
            "inferred facts not directly supported by evidence, evidence bundles / `e.g.` clauses, or entries about co-occurring peers. "
            "Entries without an allowed prefix or that exceed the per-entry length cap are rejected."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "array",
                    "description": (
                        "Complete deduplicated peer card list (max 40 entries). "
                        "Each entry must start with one of the allowed prefixes "
                        "(`IDENTITY: `, `ATTRIBUTE: `, `RELATIONSHIP: `, `INSTRUCTION: `) "
                        "followed by one concise identity marker. Entries without an allowed prefix are rejected."
                    ),
                    "items": {"type": "string"},
                },
            },
            "required": ["content"],
        },
    },
    "search_memory": {
        "name": "search_memory",
        "description": "Search for observations in memory using semantic similarity. Use this to find relevant information about the peer when you need to recall specific details.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query text",
                },
                "top_k": {
                    "type": "integer",
                    "description": "(Optional) number of results to return (default: 20, max: 40)",
                    "default": 20,
                },
            },
            "required": ["query"],
        },
    },
    "search_messages": {
        "name": "search_messages",
        "description": "Search for messages using semantic similarity and retrieve conversation snippets. Returns matching messages with surrounding context (2 messages before and after). Nearby matches within the same session are merged into a single snippet to avoid repetition.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query text to find relevant messages",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of matching messages to return (default: 10, max: 20)",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    },
    "get_recent_observations": {
        "name": "get_recent_observations",
        "description": "Get the most recent observations about the peer. Useful for understanding what's been learned recently.",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of observations to return (default: 10)",
                    "default": 10,
                },
                "session_only": {
                    "type": "boolean",
                    "description": "If true, only return observations from the current session (default: false)",
                    "default": False,
                },
            },
        },
    },
    "delete_observations": {
        "name": "delete_observations",
        "description": "Delete observations by their IDs. Use the exact ID shown in [id:xxx] format from search results. Example: if observation shows '[id:abc123XYZ]', pass 'abc123XYZ' to delete it.",
        "input_schema": {
            "type": "object",
            "properties": {
                "observation_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of observation IDs to delete (use the exact ID from [id:xxx] in search results)",
                },
            },
            "required": ["observation_ids"],
        },
    },
}

# Tools for the deduction specialist (dreamer phase 1).
DEDUCTION_SPECIALIST_TOOLS: list[dict[str, Any]] = [
    # Discovery tools
    TOOLS["get_recent_observations"],
    TOOLS["search_memory"],
    TOOLS["search_messages"],
    # Action tools
    TOOLS["create_observations_deductive"],
    TOOLS["delete_observations"],
    TOOLS["update_peer_card"],
]

# Tools for the induction specialist (dreamer phase 2).
# Creates inductive observations from explicit and deductive observations.
# Induction does not write to the peer card: that is deduction's responsibility.
INDUCTION_SPECIALIST_TOOLS: list[dict[str, Any]] = [
    # Discovery tools
    TOOLS["get_recent_observations"],
    TOOLS["search_memory"],
    TOOLS["search_messages"],
    # Action tools
    TOOLS["create_observations_inductive"],
]

# Tools for the card-refresh specialist (card_refresh dream type).
# Card-only maintenance: discovery plus update_peer_card. Deliberately excludes every
# observation-mutating tool: a card refresh must never touch observations.
CARD_REFRESH_SPECIALIST_TOOLS: list[dict[str, Any]] = [
    # Discovery tools
    TOOLS["get_recent_observations"],
    TOOLS["search_memory"],
    # Action tool
    TOOLS["update_peer_card"],
]


# Levels whose source_ids must resolve to real documents before persistence.
_SOURCE_GROUNDED_LEVELS: tuple[str, ...] = ("deductive", "inductive", "contradiction")


async def _filter_ungrounded_source_ids(
    pctx: PeopleContext,
    observations: list[schemas.ObservationInput],
    *,
    bank_id: str,
    observer: str,
    observed: str,
) -> tuple[list[schemas.ObservationInput], list[ObservationFailure]]:
    """Drop cited source_ids that resolve to no existing document.

    Derived observations must cite the documents they are built on, but models sometimes fabricate ids
    (or paste message text) to satisfy the tool schema, and nothing downstream re-checks them: a dangling
    id becomes permanent false provenance. Resolve every cited id against real documents before
    persistence: fabricated ids are stripped, and an observation left with fewer real sources than its
    level requires (1, or 2 for contradiction) is rejected as an ``ObservationFailure``.

    Returns (grounded observations, failures for ungrounded observations).
    """
    cited_ids: set[str] = set()
    for obs in observations:
        if obs.level in _SOURCE_GROUNDED_LEVELS and obs.source_ids:
            cited_ids.update(obs.source_ids)
    if not cited_ids:
        return observations, []

    docs = await conclusions_crud.fetch_documents_by_ids(pctx, bank_id, observer, observed, list(cited_ids))
    resolved_ids = {doc.id for doc in docs}

    grounded: list[schemas.ObservationInput] = []
    failed: list[ObservationFailure] = []
    for obs in observations:
        if obs.level not in _SOURCE_GROUNDED_LEVELS or not obs.source_ids:
            grounded.append(obs)
            continue
        real_ids = [sid for sid in obs.source_ids if sid in resolved_ids]
        dropped = len(obs.source_ids) - len(real_ids)
        # Mirror the per-level minimums enforced by validate_level_fields.
        min_sources = 2 if obs.level == "contradiction" else 1
        if len(real_ids) < min_sources:
            failed.append(
                ObservationFailure(
                    content_preview=obs.content[:50],
                    error=(
                        f"{dropped} of {len(obs.source_ids)} source_ids do not "
                        f"resolve to existing observations; '{obs.level}' requires "
                        f"at least {min_sources} real source(s) "
                        f"(cite only ids shown as [id:xxx] in observation results)"
                    ),
                )
            )
            continue
        if dropped:
            logger.warning(
                "Dropped %d unresolvable source_ids from %s observation in %s/%s/%s",
                dropped,
                obs.level,
                bank_id,
                observer,
                observed,
            )
            obs = obs.model_copy(update={"source_ids": real_ids})
        grounded.append(obs)
    return grounded, failed


async def save_documents_with_history(
    pctx: PeopleContext,
    bank_id: str,
    documents: list[schemas.DocumentCreate],
    *,
    observer: str,
    observed: str,
    deduplicate: bool,
    history: DreamHistoryTarget | None,
) -> conclusions_crud.CreateDocumentsResult:
    """Save conclusions and, when a dream run is open, record the created and the dedup-replaced ones.

    The history rows are written in the same transaction as the conclusions.
    """
    async with pctx.conn() as conn, conn.transaction():
        result = await conclusions_crud.create_documents(
            pctx,
            bank_id,
            documents,
            observer=observer,
            observed=observed,
            deduplicate=deduplicate,
            conn=conn,
        )
        if history is not None:
            await record_created(
                conn,
                history,
                bank_id,
                [
                    CreatedConclusion(id=c.id, level=c.level, source_ids=list(c.source_ids or []))
                    for c in result.created_conclusions
                ],
            )
            await record_removed(
                conn,
                history,
                bank_id,
                [
                    RemovedConclusion(
                        id=old.id,
                        level=old.level,
                        content=old.content,
                        session_name=old.session_name,
                        replaced_by=replacement_id,
                    )
                    for old, replacement_id in result.replaced_conclusions
                ],
            )
    return result


async def delete_observations_with_history(
    pctx: PeopleContext,
    bank_id: str,
    observation_ids: list[str],
    *,
    observer: str,
    observed: str,
    history: DreamHistoryTarget | None,
) -> list[tuple[str, str]]:
    """Soft-delete conclusions and, when a dream run is open, record each removed one with its content.

    Returns (id, level) of the rows that were deleted.
    """
    async with pctx.conn() as conn, conn.transaction():
        # The removed rows' content is copied into the dream history, so read it before the delete.
        removed_rows = (
            await conclusions_crud.fetch_documents_by_ids(pctx, bank_id, observer, observed, observation_ids, conn=conn)
            if history is not None
            else []
        )
        deleted = await conclusions_crud.delete_documents(
            pctx, bank_id, observation_ids, observer=observer, observed=observed, conn=conn
        )
        if history is not None:
            deleted_ids_now = {doc_id for doc_id, _ in deleted}
            await record_removed(
                conn,
                history,
                bank_id,
                [
                    RemovedConclusion(id=row.id, level=row.level, content=row.content, session_name=row.session_name)
                    for row in removed_rows
                    if row.id in deleted_ids_now
                ],
            )
    return deleted


async def create_observations(
    pctx: PeopleContext,
    observations: list[schemas.ObservationInput],
    observer: str,
    observed: str,
    session_name: str | None,
    bank_id: str,
    message_ids: list[int],
    message_created_at: str,
    history: DreamHistoryTarget | None = None,
) -> ObservationsCreatedResult:
    """Create multiple observations (conclusions) in a single call.

    Embeds outside any connection, then saves them in one transaction together with the dream history
    rows for the created and the dedup-replaced conclusions.
    """
    if not observations:
        logger.warning("create_observations called with empty list")
        return ObservationsCreatedResult(created_count=0, created_levels=[], failed=[])

    # Normalize before the emptiness check: str.strip() does not remove NUL,
    # so content that normalizes away has to be dropped afterwards.
    normalized_observations = [
        normalized for normalized in (_normalized_observation_input(obs) for obs in observations) if normalized.content
    ]
    if not normalized_observations:
        logger.info("No non-empty observations to create")
        return ObservationsCreatedResult(created_count=0, created_levels=[], failed=[])

    # Ground cited source_ids against real documents before persistence.
    normalized_observations, failed = await _filter_ungrounded_source_ids(
        pctx, normalized_observations, bank_id=bank_id, observer=observer, observed=observed
    )
    if not normalized_observations:
        logger.info("No observations with resolvable source_ids to create")
        return ObservationsCreatedResult(created_count=0, created_levels=[], failed=failed)

    # Compute embeddings (no connection held)
    contents = [obs.content for obs in normalized_observations]
    embeddings_by_index: dict[int, list[float]] | None = None
    try:
        embeddings = await pctx.embed(contents)
        embeddings_by_index = dict(zip(range(len(normalized_observations)), embeddings, strict=True))
    except Exception as e:
        logger.warning(
            "Batch embedding failed for create_observations; falling back to per-observation embedding: %s",
            e,
        )

    # Build document objects with pre-computed embeddings
    documents: list[schemas.DocumentCreate] = []
    for i, obs in enumerate(normalized_observations):
        embedding: list[float]
        if embeddings_by_index is not None:
            embedding = embeddings_by_index[i]
        else:
            try:
                embedding = await pctx.embed_query(obs.content)
            except Exception as e:
                logger.warning("Error embedding observation content for level '%s': %s", obs.level, e)
                failed.append(ObservationFailure(content_preview=obs.content[:50], error=f"Embedding failed: {e}"))
                continue

        # Build metadata with level-specific fields. source_ids is intentionally omitted from
        # metadata: linkage lives in people_conclusion_sources (via DocumentCreate.source_ids below).
        metadata = schemas.DocumentMetadata(
            message_ids=message_ids,
            message_created_at=message_created_at,
            premises=obs.premises if obs.level == "deductive" else None,
            sources=obs.sources if obs.level in ("inductive", "contradiction") else None,
            pattern_type=obs.pattern_type if obs.level == "inductive" else None,
            confidence=(obs.confidence or "medium") if obs.level == "inductive" else None,
        )

        doc = schemas.DocumentCreate(
            content=obs.content,
            session_name=session_name,
            level=obs.level,
            metadata=metadata,
            embedding=embedding,
            source_ids=obs.source_ids if obs.level in ("deductive", "inductive", "contradiction") else None,
        )
        documents.append(doc)

    # Bulk create all documents in one transaction with their history rows
    accepted: list[Any] = []
    if documents:
        result = await save_documents_with_history(
            pctx, bank_id, documents, observer=observer, observed=observed, deduplicate=DEDUPLICATE, history=history
        )
        accepted = list(result.created_conclusions)
        logger.info("Created %d observations in %s/%s/%s", len(accepted), bank_id, observer, observed)

    return ObservationsCreatedResult(
        created_count=len(accepted),
        created_levels=[doc.level for doc in accepted],
        failed=failed,
    )


@dataclass
class ToolContext:
    """Context object passed to tool handlers."""

    pctx: PeopleContext
    bank_id: str
    observer: str
    observed: str
    session_name: str | None
    include_observation_ids: bool
    # Shared lock for serializing writes to the same bank/observer/observed.
    db_lock: asyncio.Lock
    # Optional resolved configuration for checking feature flags
    configuration: ResolvedConfiguration | None = None
    agent_type: str | None = None  # "deduction", "induction", "card_refresh"
    # Set only for dream specialists whose run is recorded for Representation History.
    # Write handlers record their changes to it in the same transaction as the change.
    history: DreamHistoryTarget | None = None


def _normalize_observation_id(obs_id: str) -> str:
    """Strip the display-format ``id:`` prefix from a model-supplied observation ID.

    Observations are presented to agents as ``[id:xxx]``, and despite tool-schema instructions to pass
    the bare ID, models sometimes copy the prefix verbatim. Conclusion ids are nanoids whose alphabet
    includes ``-`` and ``_``, so only the ``id:`` prefix and surrounding whitespace are stripped.
    """
    obs_id = obs_id.strip()
    if obs_id.lower().startswith("id:"):
        obs_id = obs_id[3:]
    return obs_id.strip()


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _format_datetime_utc(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_datetime_iso(raw: str) -> datetime:
    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


async def _latest_source_timestamp(
    ctx: ToolContext,
    observations: list[schemas.ObservationInput],
) -> str | None:
    """Latest ``message_created_at`` across all source observations in the batch.

    Dreamer conclusions (deductive/inductive) are derived from existing observations referenced by
    ``source_ids`` rather than from live messages. Their logical timestamp is the point when the
    conclusion became possible from its evidence, not when the dreamer happened to run, so we date
    ``internal_metadata["message_created_at"]`` to the most recent source observation. Returns None
    if no source_ids resolve to a usable timestamp (caller falls back to now).
    """
    source_ids: list[str] = []
    for obs in observations:
        if obs.source_ids:
            source_ids.extend(obs.source_ids)
    if not source_ids:
        return None

    latest: datetime | None = None
    docs = await conclusions_crud.fetch_documents_by_ids(
        ctx.pctx, ctx.bank_id, ctx.observer, ctx.observed, sorted(set(source_ids))
    )
    for doc in docs:
        raw = doc.internal_metadata.get("message_created_at")
        if not isinstance(raw, str):
            continue
        try:
            # always tz-aware, so the comparison below can't crash on mixed formats
            parsed = _parse_datetime_iso(raw)
        except ValueError:
            continue
        if latest is None or parsed > latest:
            latest = parsed

    return _format_datetime_utc(latest) if latest is not None else None


async def _handle_create_observations_impl(
    ctx: ToolContext,
    tool_input: dict[str, Any],
    *,
    forced_level: str | None = None,
) -> "str | ToolResult":
    """Handle create_observations tool."""
    raw_observations: Any = tool_input.get("observations", [])
    if not isinstance(raw_observations, list):
        return f"ERROR: observations must be a list of objects, got {type(raw_observations).__name__}"
    if not raw_observations:
        return "ERROR: observations list is empty"

    observation_dicts: list[dict[str, Any]] = []
    validation_failures: list[ObservationFailure] = []
    for obs in cast(list[Any], raw_observations):
        if isinstance(obs, dict):
            observation_dicts.append(cast(dict[str, Any], obs))
        else:
            validation_failures.append(
                ObservationFailure(
                    content_preview=str(obs)[:50],
                    error=f"Each observation must be an object with a 'content' field, got {type(obs).__name__}",
                )
            )

    # The dreamer has no messages in hand: the default level is deductive.
    for obs in observation_dicts:
        if forced_level is not None:
            obs["level"] = forced_level
        else:
            obs.setdefault("level", "deductive")
        # Models sometimes copy the display-format "id:" prefix into source_ids;
        # normalize so provenance links reference real document IDs.
        source_ids = obs.get("source_ids")
        if isinstance(source_ids, list):
            normalized_source_ids: list[str] = []
            for source_id in cast(list[Any], source_ids):
                if isinstance(source_id, str):
                    normalized_source_ids.append(_normalize_observation_id(source_id))
            obs["source_ids"] = normalized_source_ids
    # Validate observations individually so valid ones are still processed
    observations: list[schemas.ObservationInput] = []
    for obs in observation_dicts:
        try:
            validated = schemas.ObservationInput.model_validate(obs)
        except ValidationError as e:
            validation_failures.append(
                ObservationFailure(
                    content_preview=str(obs.get("content", ""))[:50],
                    error=f"Validation failed: {e}",
                )
            )
            continue
        # Session-purity invariant: explicit observations record what was directly derived from a
        # session's messages. The dreamer must not mint them: consolidation output belongs at a
        # derived level.
        if validated.level == "explicit":
            validation_failures.append(
                ObservationFailure(
                    content_preview=validated.content[:50],
                    error=(
                        "Only message ingestion can create 'explicit' observations; "
                        "use a derived level (deductive/inductive/contradiction)"
                    ),
                )
            )
            continue
        observations.append(validated)

    if not observations:
        failure_details = "; ".join(f"'{f.content_preview}': {f.error}" for f in validation_failures)
        return f"ERROR: All observations failed validation: {failure_details}"

    # Backdate the conclusion to the latest source observation, which is when the inference became possible.
    message_ids: list[int] = []
    message_created_at = (await _latest_source_timestamp(ctx, observations)) or _utc_now_iso()

    # Use lock to serialize database writes
    async with ctx.db_lock:
        result = await create_observations(
            ctx.pctx,
            observations=observations,
            observer=ctx.observer,
            observed=ctx.observed,
            session_name=ctx.session_name,
            bank_id=ctx.bank_id,
            message_ids=message_ids,
            message_created_at=message_created_at,
            history=ctx.history,
        )

    # Merge validation and embedding failures
    all_failures = validation_failures + result.failed

    # Count levels from actually-created observations
    levels = result.created_levels
    explicit_count = levels.count("explicit")
    deductive_count = levels.count("deductive")
    inductive_count = levels.count("inductive")
    contradiction_count = levels.count("contradiction")

    response = (
        f"Created {result.created_count} observations for {ctx.observed} by {ctx.observer} "
        f"({explicit_count} explicit, {deductive_count} deductive, "
        f"{inductive_count} inductive, {contradiction_count} contradiction)"
    )

    if all_failures:
        failure_details = "; ".join(f"'{f.content_preview}': {f.error}" for f in all_failures)
        response += f"\nFailed {len(all_failures)}: {failure_details}"

    # Surface created_count so the specialist can sum actual observations across the run rather than
    # just counting create_observations calls.
    return ToolResult(
        content=response,
        metadata={"created_count": result.created_count, "levels": levels},
    )


async def _handle_create_observations_deductive(ctx: ToolContext, tool_input: dict[str, Any]) -> "str | ToolResult":
    return await _handle_create_observations_impl(ctx, tool_input, forced_level="deductive")


async def _handle_create_observations_inductive(ctx: ToolContext, tool_input: dict[str, Any]) -> "str | ToolResult":
    return await _handle_create_observations_impl(ctx, tool_input, forced_level="inductive")


async def _handle_update_peer_card(ctx: ToolContext, tool_input: dict[str, Any]) -> "str | ToolResult":
    """Handle update_peer_card tool."""
    # Check if peer card creation is disabled via configuration
    if ctx.configuration is not None and not ctx.configuration.peer_card.create:
        logger.info(
            "Peer card creation disabled for %s, skipping update",
            ctx.bank_id,
        )
        return "Peer card creation is disabled for this bank/session configuration."

    raw_peer_card_content = tool_input.get("content")

    # Guard against None or empty content — keep the existing peer card.
    if raw_peer_card_content is None:
        logger.warning(
            "Peer card update called with None content for %s, keeping existing card",
            ctx.bank_id,
        )
        return "Peer card content was empty, no update performed."

    # Normalize, validate structure, and deduplicate to keep peer cards bounded
    # and on-spec.
    normalized_peer_card: list[str] = []
    seen: set[str] = set()
    rejected_count = 0
    # Keep a small sample of rejected entries to surface back to the model so it
    # can self-correct on a retry. Capped to avoid bloating the tool response.
    rejected_samples: list[str] = []
    _REJECTED_SAMPLE_CAP = 3
    _REJECTED_SAMPLE_LINE_LIMIT = 120
    items = (
        cast(list[str], raw_peer_card_content)
        if isinstance(raw_peer_card_content, list)
        else [str(raw_peer_card_content)]
    )
    for item in items:
        # Strip NUL first: jsonb rejects it, and the checks below (emptiness,
        # body, length cap, dedupe) must judge the text that gets stored.
        line = strip_nul(str(item)).strip()
        if not line:
            continue

        if not _validate_peer_card_entry(line):
            rejected_count += 1
            if len(rejected_samples) < _REJECTED_SAMPLE_CAP:
                rejected_samples.append(line[:_REJECTED_SAMPLE_LINE_LIMIT])
            logger.info(
                "Rejecting peer card entry (no allowed prefix, empty body, or over length cap): %r",
                line[:80],
            )
            continue

        # Case-insensitive dedupe with whitespace normalization.
        normalized_key = " ".join(line.lower().split())
        if normalized_key in seen:
            continue
        seen.add(normalized_key)
        normalized_peer_card.append(line)

    if rejected_count:
        logger.info(
            "Peer card update for %s/%s/%s rejected %d structurally invalid entries",
            ctx.bank_id,
            ctx.observer,
            ctx.observed,
            rejected_count,
        )

    def _format_rejection_feedback(scope: str) -> str:
        """Build a self-correction hint for the model. `scope` is grammar glue:
        either "all" (every entry rejected) or e.g. "3 of 12" (partial)."""
        samples_block = ""
        if rejected_samples:
            sample_lines = "\n".join(f"  - {s!r}" for s in rejected_samples)
            extra = (
                f" (+{rejected_count - len(rejected_samples)} more)" if rejected_count > len(rejected_samples) else ""
            )
            samples_block = f" Examples of rejected entries{extra}:\n{sample_lines}"
        return (
            f"Rejected {scope} entries for failing structural validation. "
            "Each entry must start with one of `IDENTITY: `, `ATTRIBUTE: `, "
            "`RELATIONSHIP: `, or `INSTRUCTION: ` and stay under the per-entry "
            f"length cap.{samples_block}"
        )

    # Don't clear the peer card if all content normalized to empty or every
    # entry was structurally invalid.
    if not normalized_peer_card:
        logger.warning(
            "Peer card update normalized to empty for %s (rejected=%d), keeping existing card",
            ctx.bank_id,
            rejected_count,
        )
        if rejected_count:
            return _format_rejection_feedback(f"all {rejected_count}")
        return "Peer card content was empty after normalization, no update performed."

    if len(normalized_peer_card) > MAX_PEER_CARD_FACTS:
        logger.warning(
            "Peer card update exceeded max facts (%s), truncating from %s to %s",
            MAX_PEER_CARD_FACTS,
            len(normalized_peer_card),
            MAX_PEER_CARD_FACTS,
        )
        normalized_peer_card = normalized_peer_card[:MAX_PEER_CARD_FACTS]

    async with ctx.db_lock:
        await peer_card_crud.set_peer_card(
            ctx.pctx,
            ctx.bank_id,
            normalized_peer_card,
            observer=ctx.observer,
            observed=ctx.observed,
            history=ctx.history,
        )
    logger.info(f"Updated peer card for {ctx.bank_id}/{ctx.observer}/{ctx.observed}")

    # Signal a successful peer_card update so the specialist can set its `peer_card_updated` flag.
    success_content = (
        f"Updated peer card for {ctx.observed} by {ctx.observer} with {len(normalized_peer_card)} entries."
    )
    if rejected_count:
        # Partial reject: surface the rejection so the model can re-emit the
        # dropped entries (with correct prefixes) on a retry instead of
        # silently losing them.
        accepted = len(normalized_peer_card)
        total = accepted + rejected_count
        success_content = f"{success_content} {_format_rejection_feedback(f'{rejected_count} of {total}')}"
    return ToolResult(
        content=success_content,
        metadata={
            "peer_card_updated": True,
            "facts_count": len(normalized_peer_card),
            "rejected_count": rejected_count,
        },
    )


async def _handle_search_memory(ctx: ToolContext, tool_input: dict[str, Any]) -> "str | ToolResult":
    """Handle search_memory tool."""
    top_k = _bounded_int(tool_input.get("top_k"), 20, hi=40)
    query = tool_input["query"]
    try:
        query_embedding = await ctx.pctx.embed_query(query)
    except ValueError as e:
        # Provider/config failure, not an oversized query. Keep returning a string so the tool loop
        # can continue, but don't blame the query.
        return f"ERROR: Embedding the query failed: {e}"

    # Base metadata; results_count gets filled in below.
    search_meta: dict[str, Any] = {"top_k": top_k, "used_embedding": True, "embedding_query_count": 1}

    documents = await conclusions_crud.query_documents(
        ctx.pctx,
        ctx.bank_id,
        query,
        observer=ctx.observer,
        observed=ctx.observed,
        top_k=top_k,
        embedding=query_embedding,
    )
    mem = Representation.from_documents(documents)
    total_count = mem.len()
    if total_count == 0:
        return ToolResult(
            content=f"No observations found for query '{query}'",
            metadata={**search_meta, "results_count": 0},
        )
    mem_str = mem.str_with_ids() if ctx.include_observation_ids else str(mem)
    search_meta["results_count"] = total_count
    return ToolResult(
        content=f"Found {total_count} observations for query '{query}':\n\n{mem_str}",
        metadata=search_meta,
    )


async def _handle_search_messages(ctx: ToolContext, tool_input: dict[str, Any]) -> "str | ToolResult":
    """Handle search_messages tool."""
    from ..dialectic import messages as message_search

    query = tool_input["query"]
    limit = _bounded_int(tool_input.get("limit"), 10, hi=20)
    snippets = await message_search.search_messages(
        ctx.pctx,
        ctx.bank_id,
        ctx.session_name,
        query=query,
        limit=limit,
        context_window=2,
        observer=ctx.observer or None,
    )
    search_meta: dict[str, Any] = {"top_k": limit, "results_count": len(snippets)}
    if not snippets:
        return ToolResult(content=f"No messages found for query '{query}'", metadata=search_meta)
    formatted, was_truncated, original_chars = message_search.format_message_snippets(
        snippets, f"for query '{query}'", max_chars=MAX_TOOL_OUTPUT_CHARS
    )
    return ToolResult(
        content=formatted,
        metadata={**search_meta, **_truncation_metadata(was_truncated, original_chars)},
    )


async def _handle_get_recent_observations(ctx: ToolContext, tool_input: dict[str, Any]) -> str:
    """Handle get_recent_observations tool."""
    session_only = tool_input.get("session_only", False)
    documents = await conclusions_crud.query_documents_recent(
        ctx.pctx,
        ctx.bank_id,
        observer=ctx.observer,
        observed=ctx.observed,
        limit=_bounded_int(tool_input.get("limit"), 10, hi=100),
        session_name=ctx.session_name if session_only else None,
    )
    representation = Representation.from_documents(documents)
    total_count = representation.len()
    if total_count == 0:
        return "No recent observations found"
    scope = "this session" if session_only else "all sessions"
    repr_str = representation.str_with_ids() if ctx.include_observation_ids else str(representation)
    return f"Found {total_count} recent observations from {scope}:\n\n{repr_str}"


async def _handle_delete_observations(ctx: ToolContext, tool_input: dict[str, Any]) -> "str | ToolResult":
    """Handle delete_observations tool."""
    raw_observation_ids: Any = tool_input.get("observation_ids", [])
    if not isinstance(raw_observation_ids, list) or not all(
        isinstance(obs_id, str) for obs_id in cast(list[Any], raw_observation_ids)
    ):
        return "ERROR: observation_ids must be a list of observation ID strings"
    observation_ids = [_normalize_observation_id(obs_id) for obs_id in cast(list[str], raw_observation_ids)]
    if not observation_ids:
        return "ERROR: observation_ids list is empty"

    async with ctx.db_lock:
        deleted = await delete_observations_with_history(
            ctx.pctx,
            ctx.bank_id,
            observation_ids,
            observer=ctx.observer,
            observed=ctx.observed,
            history=ctx.history,
        )

    deleted_ids = {doc_id for doc_id, _ in deleted}
    for obs_id in observation_ids:
        if obs_id not in deleted_ids:
            logger.warning(
                "Failed to delete observation %s (not found, already deleted, or wrong scope)",
                obs_id,
            )

    deleted_count = len(deleted)
    return ToolResult(
        content=f"Deleted {deleted_count} observations",
        metadata={"deleted_count": deleted_count, "levels": [level for _, level in deleted]},
    )


_TOOL_HANDLERS: dict[str, Callable[[ToolContext, dict[str, Any]], Any]] = {
    "create_observations_deductive": _handle_create_observations_deductive,
    "create_observations_inductive": _handle_create_observations_inductive,
    "update_peer_card": _handle_update_peer_card,
    "search_memory": _handle_search_memory,
    "search_messages": _handle_search_messages,
    "get_recent_observations": _handle_get_recent_observations,
    "delete_observations": _handle_delete_observations,
}


def get_tool_handler(
    tool_name: str,
) -> Callable[[ToolContext, dict[str, Any]], Any] | None:
    """Default handler lookup, for `handler_resolver`s that wrap it."""
    return _TOOL_HANDLERS.get(tool_name)


def _describe_tool_error(error: Exception) -> str:
    """Bounded description of `error`, safe to hand back to the model."""
    message = str(error).strip()
    max_chars = MAX_TOOL_OUTPUT_CHARS
    if len(message) > max_chars:
        message = message[:max_chars] + f"... [truncated {len(message) - max_chars:,} chars]"
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


async def create_tool_executor(
    pctx: PeopleContext,
    bank_id: str,
    observer: str,
    observed: str,
    session_name: str | None = None,
    include_observation_ids: bool = False,
    configuration: ResolvedConfiguration | None = None,
    agent_type: str | None = None,
    history: DreamHistoryTarget | None = None,
    handler_resolver: Callable[[str], Callable[[ToolContext, dict[str, Any]], Any] | None] | None = None,
) -> Callable[[str, dict[str, Any]], Any]:
    """Create a tool executor function with the context captured.

    ``handler_resolver`` replaces the default handler-table lookup for resolving tool names to handlers;
    returning None takes the "Unknown tool" path. ``history`` is the dream run that write handlers record
    their changes to; None records nothing.
    """
    # Get shared lock from registry to prevent race conditions when multiple tool executors operate on
    # the same bank/observer/observed concurrently
    shared_lock = await get_observation_lock(bank_id, observer, observed)

    ctx = ToolContext(
        pctx=pctx,
        bank_id=bank_id,
        observer=observer,
        observed=observed,
        session_name=session_name,
        include_observation_ids=include_observation_ids,
        db_lock=shared_lock,
        configuration=configuration,
        agent_type=agent_type,
        history=history,
    )

    async def execute_tool(tool_name: str, tool_input: dict[str, Any]) -> str:
        """Execute a tool and return the result for the LLM."""
        from .tool_loop import set_last_tool_error, set_last_tool_metadata

        # Log nondisclosive call shape only: raw `tool_input` can carry user content.
        logger.info("[tool call] %s keys=%s", tool_name, sorted(tool_input.keys()))

        result_str: str = ""
        metadata: dict[str, Any] = {}
        is_error: bool = False

        try:
            handler = (handler_resolver or _TOOL_HANDLERS.get)(tool_name)
            if handler:
                handler_result = await handler(ctx, tool_input)
                # Handlers return either a plain str or a ToolResult(content, metadata).
                if isinstance(handler_result, ToolResult):
                    result_str = handler_result.content
                    metadata = handler_result.metadata
                else:
                    result_str = handler_result
                # Handlers report recoverable failures to the model as "ERROR: ..." strings.
                is_error = result_str.startswith("ERROR:")
                logger.info(
                    "[tool result] %s len=%d metadata_keys=%s",
                    tool_name,
                    len(result_str),
                    sorted(metadata.keys()),
                )
            else:
                result_str = f"Unknown tool: {tool_name}"
                is_error = True
                logger.warning(result_str)

        except asyncio.CancelledError:
            result_str = f"Tool {tool_name} cancelled"
            is_error = True
            raise
        except ValueError as e:
            # Recoverable errors (bad input, validation failures) - return to LLM
            result_str = f"Tool {tool_name} failed with invalid input: {_describe_tool_error(e)}"
            is_error = True
            logger.warning(result_str)
        except KeyError as e:
            # Missing required parameters - return to LLM
            result_str = f"Tool {tool_name} missing required parameter: {e}"
            is_error = True
            logger.warning(result_str)
        except Exception as e:
            # Unexpected errors: log with full traceback but still return to the LLM, which should be
            # able to continue with other tools.
            result_str = f"Tool {tool_name} failed unexpectedly: {_describe_tool_error(e)}"
            is_error = True
            logger.error(result_str, exc_info=True)
        finally:
            # Publish ToolResult.metadata for the loop to stash on the tool-call record.
            set_last_tool_metadata(metadata)
            set_last_tool_error(is_error)

        return result_str

    return execute_tool
