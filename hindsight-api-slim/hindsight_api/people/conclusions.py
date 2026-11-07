# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/crud/document.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""Conclusions (Honcho's "documents"): the rows a representation is built from.

A conclusion is one statement an observer holds about an observed person, at one of four levels
(explicit, deductive, inductive, contradiction). Explicit conclusions cite the messages they were drawn
from (``source_message_ids``); derived levels cite the conclusions they follow from (``source_ids``).
Every row also carries ``source_url`` and ``said_at`` copied from its source message, so what a person
model shows can cite where it was said.

Honcho's vector-store backends, reconciler sync state and dream-history recording are not copied:
embeddings live in the ``embedding`` column and similarity is exact (``<=>``). Callers that record dream
history get the created and replaced rows back on ``CreateDocumentsResult`` and an optional ``conn`` to
write the history rows in the same transaction.
"""

import datetime
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from . import schemas
from .context import PeopleContext
from .db import connection
from .exceptions import ResourceNotFoundException, ValidationException
from .ids import generate_nanoid
from .jsonb import from_jsonb, to_jsonb

logger = logging.getLogger(__name__)

SOURCE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{21}$")

_CONCLUSION_COLUMNS = (
    "bank_id, id, observer, observed, level, content, times_derived, session_name, internal_metadata, "
    "source_url, said_at, created_at, deleted_at"
)

# Honcho's DERIVER.DEDUPLICATE default: semantic dedup of conclusions on by default.
DEDUPLICATE = True

# Shared by is_rejected_duplicate and create_documents candidate resolution.
_SEMANTIC_DUP_MAX_DISTANCE = 0.05
_SEMANTIC_DUP_TOP_K = 1

_FILTER_COLUMNS = frozenset({"level", "session_name"})


@dataclass
class Conclusion:
    """One live or soft-deleted conclusion row, with its reasoning and evidence edges loaded.

    Stands in for Honcho's ``Document`` model: ``source_ids`` are the premise conclusions it was
    derived from, ``source_message_ids`` the public ids of the messages cited as evidence.
    """

    id: str
    bank_id: str
    observer: str
    observed: str
    level: str
    content: str
    times_derived: int
    session_name: str | None
    internal_metadata: dict[str, Any]
    created_at: datetime.datetime
    source_url: str | None = None
    said_at: datetime.datetime | None = None
    deleted_at: datetime.datetime | None = None
    source_ids: list[str] | None = None
    source_message_ids: list[str] | None = None

    @property
    def message_ids(self) -> list[int]:
        """The ids of the messages this conclusion was derived from (Honcho's internal_metadata field)."""
        return list(self.internal_metadata.get("message_ids", []))


def build_source_links(source_ids: list[str] | None) -> list[str]:
    """Clean an LLM-provided source_ids list into premise ids to store.

    Dedupes (the primary key is (conclusion, kind, source)) and drops entries that are not shaped like
    conclusion ids -- the model occasionally emits timestamps or numeric refs under schema pressure.
    """
    if not source_ids:
        return []
    return [sid for sid in dict.fromkeys(source_ids) if SOURCE_ID_RE.match(sid)]


def build_source_message_links(message_ids: list[str] | None) -> list[str]:
    """Clean cited message public ids into the ids to store, deduped."""
    if not message_ids:
        return []
    return [mid for mid in dict.fromkeys(message_ids) if SOURCE_ID_RE.match(mid)]


def _merge_citations(first: list[str] | None, incoming: list[str] | None) -> list[str] | None:
    """Union two citation lists, keeping first-seen order; None when both empty."""
    return list(dict.fromkeys([*(first or []), *(incoming or [])])) or None


def _filter_clause(filters: dict[str, Any] | None, first_param: int) -> tuple[str, list[Any]]:
    """SQL for a level / session_name filter: equality, or ``{"in": [...]}`` (an empty list matches nothing)."""
    if not filters:
        return "", []
    clauses: list[str] = []
    params: list[Any] = []
    n = first_param
    for key, value in filters.items():
        if key not in _FILTER_COLUMNS:
            raise ValidationException(f"Unsupported conclusion filter {key!r}: use one of {sorted(_FILTER_COLUMNS)}")
        if isinstance(value, dict):
            if set(value) != {"in"}:
                raise ValidationException(f"Unsupported filter operator for {key!r}: {sorted(value)}")
            clauses.append(f"{key} = ANY(${n}::text[])")
            params.append(list(value["in"]))
        else:
            clauses.append(f"{key} = ${n}")
            params.append(value)
        n += 1
    return " AND " + " AND ".join(clauses), params


async def _load_links(
    ctx: PeopleContext, db: Any, bank_id: str, conclusion_ids: list[str]
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """The premise ids and the cited message ids of the given conclusions, in citation order."""
    premises: dict[str, list[str]] = {}
    messages: dict[str, list[str]] = {}
    if not conclusion_ids:
        return premises, messages
    rows = await db.fetch(
        f"SELECT conclusion_id, source_id, kind FROM {ctx.table('conclusion_sources')} "
        "WHERE bank_id = $1 AND conclusion_id = ANY($2::text[]) ORDER BY conclusion_id, kind, position",
        bank_id,
        conclusion_ids,
    )
    for row in rows:
        target = premises if row["kind"] == "conclusion" else messages
        target.setdefault(row["conclusion_id"], []).append(row["source_id"])
    return premises, messages


async def _conclusions_from_rows(ctx: PeopleContext, db: Any, rows: Sequence[Any]) -> list[Conclusion]:
    premises, messages = await _load_links(ctx, db, rows[0]["bank_id"], [r["id"] for r in rows]) if rows else ({}, {})
    return [
        Conclusion(
            id=r["id"],
            bank_id=r["bank_id"],
            observer=r["observer"],
            observed=r["observed"],
            level=r["level"],
            content=r["content"],
            times_derived=r["times_derived"],
            session_name=r["session_name"],
            internal_metadata=from_jsonb(r["internal_metadata"], {}),
            source_url=r["source_url"],
            said_at=r["said_at"],
            created_at=r["created_at"],
            deleted_at=r["deleted_at"],
            source_ids=premises.get(r["id"]) or None,
            source_message_ids=messages.get(r["id"]) or None,
        )
        for r in rows
    ]


async def get_all_documents(
    ctx: PeopleContext,
    bank_id: str,
    *,
    observer: str,
    observed: str,
    filters: dict[str, Any] | None = None,
    reverse: bool = False,
    limit: int | None = None,
    offset: int = 0,
) -> tuple[list[Conclusion], int]:
    """A page of the live conclusions of an observer/observed pair and their total count.

    Ordered by created_at, newest first by default (``reverse`` gives oldest first).
    """
    clause, params = _filter_clause(filters, 4)
    order = "ASC" if reverse else "DESC"
    n = len(params) + 4
    async with ctx.conn() as db:
        total = await db.fetchval(
            f"SELECT count(*) FROM {ctx.table('conclusions')} "
            f"WHERE bank_id = $1 AND observer = $2 AND observed = $3 AND deleted_at IS NULL{clause}",
            bank_id,
            observer,
            observed,
            *params,
        )
        rows = await db.fetch(
            f"SELECT {_CONCLUSION_COLUMNS} FROM {ctx.table('conclusions')} "
            f"WHERE bank_id = $1 AND observer = $2 AND observed = $3 AND deleted_at IS NULL{clause} "
            f"ORDER BY created_at {order}, id {order} LIMIT ${n} OFFSET ${n + 1}",
            bank_id,
            observer,
            observed,
            *params,
            limit,
            offset,
        )
        return await _conclusions_from_rows(ctx, db, rows), int(total or 0)


async def query_documents_recent(
    ctx: PeopleContext,
    bank_id: str,
    *,
    observer: str,
    observed: str,
    limit: int = 10,
    session_name: str | None = None,
) -> list[Conclusion]:
    """Query most recent conclusions, ordered by created_at descending."""
    extra, params = ("", []) if session_name is None else (" AND session_name = $4", [session_name])
    async with ctx.conn() as db:
        rows = await db.fetch(
            f"SELECT {_CONCLUSION_COLUMNS} FROM {ctx.table('conclusions')} "
            f"WHERE bank_id = $1 AND observer = $2 AND observed = $3 AND deleted_at IS NULL{extra} "
            f"ORDER BY created_at DESC, id LIMIT ${len(params) + 4}",
            bank_id,
            observer,
            observed,
            *params,
            limit,
        )
        return await _conclusions_from_rows(ctx, db, rows)


@dataclass(frozen=True)
class DocumentLevelSummary:
    """Live conclusion counts for one observer/observed pair."""

    counts_by_level: dict[str, int]
    earliest: datetime.datetime | None
    latest: datetime.datetime | None


async def summarize_documents_by_level(
    ctx: PeopleContext,
    bank_id: str,
    *,
    observer: str,
    observed: str,
) -> DocumentLevelSummary:
    """Count live conclusions per level and the span of their creation times."""
    async with ctx.conn() as db:
        rows = await db.fetch(
            f"SELECT level, count(id) AS n, min(created_at) AS first_at, max(created_at) AS last_at "
            f"FROM {ctx.table('conclusions')} "
            "WHERE bank_id = $1 AND observer = $2 AND observed = $3 AND deleted_at IS NULL GROUP BY level",
            bank_id,
            observer,
            observed,
        )
    counts: dict[str, int] = {}
    earliest: datetime.datetime | None = None
    latest: datetime.datetime | None = None
    for row in rows:
        counts[str(row["level"])] = int(row["n"])
        if row["first_at"] is not None and (earliest is None or row["first_at"] < earliest):
            earliest = row["first_at"]
        if row["last_at"] is not None and (latest is None or row["last_at"] > latest):
            latest = row["last_at"]
    return DocumentLevelSummary(counts_by_level=counts, earliest=earliest, latest=latest)


async def query_explicit_documents_after(
    ctx: PeopleContext,
    bank_id: str,
    *,
    observer: str,
    observed: str,
    after: tuple[datetime.datetime, str] | None,
    since: datetime.datetime | None,
    limit: int,
) -> tuple[list[Conclusion], int]:
    """Live explicit conclusions past a position, oldest first, plus their total.

    `after` is a `(created_at, id)` keyset position; conclusions strictly past it are returned. Without
    it, `since` (if given) bounds `created_at` instead. Ordering ends in `id`, so conclusions sharing a
    timestamp are neither skipped nor repeated when the caller pages forward from the last one returned.
    """
    conditions = "bank_id = $1 AND observer = $2 AND observed = $3 AND level = 'explicit' AND deleted_at IS NULL"
    params: list[Any] = [bank_id, observer, observed]
    if after is not None:
        conditions += " AND (created_at, id) > ($4, $5)"
        params += [after[0], after[1]]
    elif since is not None:
        conditions += " AND created_at > $4"
        params.append(since)
    async with ctx.conn() as db:
        total = int(
            await db.fetchval(f"SELECT count(id) FROM {ctx.table('conclusions')} WHERE {conditions}", *params) or 0
        )
        if total == 0:
            return [], 0
        rows = await db.fetch(
            f"SELECT {_CONCLUSION_COLUMNS} FROM {ctx.table('conclusions')} WHERE {conditions} "
            f"ORDER BY created_at, id LIMIT ${len(params) + 1}",
            *params,
            limit,
        )
        return await _conclusions_from_rows(ctx, db, rows), total


async def query_documents_most_derived(
    ctx: PeopleContext,
    bank_id: str,
    *,
    observer: str,
    observed: str,
    limit: int = 10,
) -> list[Conclusion]:
    """Query conclusions sorted by times_derived (most reinforced first), newest first among ties."""
    async with ctx.conn() as db:
        rows = await db.fetch(
            f"SELECT {_CONCLUSION_COLUMNS} FROM {ctx.table('conclusions')} "
            "WHERE bank_id = $1 AND observer = $2 AND observed = $3 AND deleted_at IS NULL "
            # created_at is the transaction timestamp, so conclusions created in the same batch share
            # it -- id keeps the order deterministic.
            "ORDER BY times_derived DESC, created_at DESC, id LIMIT $4",
            bank_id,
            observer,
            observed,
            limit,
        )
        return await _conclusions_from_rows(ctx, db, rows)


async def fetch_documents_by_ids(
    ctx: PeopleContext,
    bank_id: str,
    observer: str,
    observed: str,
    document_ids: list[str],
    filters: dict[str, Any] | None = None,
    *,
    conn: Any | None = None,
) -> list[Conclusion]:
    """Fetch live conclusions of a pair by ids, preserving input order."""
    if not document_ids:
        return []
    clause, params = _filter_clause(filters, 5)
    async with connection(ctx, conn) as db:
        rows = await db.fetch(
            f"SELECT {_CONCLUSION_COLUMNS} FROM {ctx.table('conclusions')} "
            "WHERE bank_id = $1 AND observer = $2 AND observed = $3 AND deleted_at IS NULL "
            f"AND id = ANY($4::text[]){clause}",
            bank_id,
            observer,
            observed,
            document_ids,
            *params,
        )
        found = {c.id: c for c in await _conclusions_from_rows(ctx, db, rows)}
    return [found[doc_id] for doc_id in document_ids if doc_id in found]


async def get_documents_by_ids(
    ctx: PeopleContext,
    bank_id: str,
    document_ids: list[str],
    *,
    conn: Any | None = None,
) -> list[Conclusion]:
    """Get multiple live conclusions by their ids (fewer than requested if some ids don't exist)."""
    if not document_ids:
        return []
    async with connection(ctx, conn) as db:
        rows = await db.fetch(
            f"SELECT {_CONCLUSION_COLUMNS} FROM {ctx.table('conclusions')} "
            "WHERE bank_id = $1 AND id = ANY($2::text[]) AND deleted_at IS NULL",
            bank_id,
            document_ids,
        )
        return await _conclusions_from_rows(ctx, db, rows)


async def get_child_observations(
    ctx: PeopleContext,
    bank_id: str,
    parent_id: str,
    *,
    observer: str | None = None,
    observed: str | None = None,
    reverse: bool = False,
) -> list[Conclusion]:
    """All live conclusions that have this one as a source/premise.

    Useful for traversing the reasoning tree upward (source -> derived conclusions).
    """
    params: list[Any] = [bank_id, parent_id]
    extra = ""
    if observer:
        params.append(observer)
        extra += f" AND c.observer = ${len(params)}"
    if observed:
        params.append(observed)
        extra += f" AND c.observed = ${len(params)}"
    order = "ASC" if reverse else "DESC"
    columns = ", ".join(f"c.{col.strip()}" for col in _CONCLUSION_COLUMNS.split(","))
    async with ctx.conn() as db:
        rows = await db.fetch(
            f"SELECT {columns} FROM {ctx.table('conclusions')} c "
            f"JOIN {ctx.table('conclusion_sources')} s ON s.bank_id = c.bank_id AND s.conclusion_id = c.id "
            "WHERE c.bank_id = $1 AND s.kind = 'conclusion' AND s.source_id = $2 AND c.deleted_at IS NULL"
            f"{extra} ORDER BY c.created_at {order}, c.id {order}",
            *params,
        )
        return await _conclusions_from_rows(ctx, db, rows)


async def query_documents(
    ctx: PeopleContext,
    bank_id: str,
    query: str,
    *,
    observer: str,
    observed: str,
    filters: dict[str, Any] | None = None,
    max_distance: float | None = None,
    top_k: int = 5,
    embedding: list[float] | None = None,
) -> list[Conclusion]:
    """Query conclusions using semantic similarity (exact cosine search over the embedding column).

    Args:
        query: Search query text
        observer: Handle of the observing peer
        observed: Handle of the observed peer
        filters: Optional filters (supports: level, session_name; equality or ``{"in": [...]}``)
        max_distance: Maximum cosine distance for results
        top_k: Number of results to return
        embedding: Optional pre-computed embedding for the query (avoids an extra embedding call)

    Returns:
        Matching conclusions, closest first
    """
    if top_k <= 0:
        return []

    if embedding is None:
        embedding = await ctx.embed_query(query)

    clause, params = _filter_clause(filters, 5)
    distance_clause = ""
    if max_distance is not None:
        params.append(max_distance)
        distance_clause = f" AND embedding <=> $4::vector <= ${len(params) + 4}"
    n = len(params) + 5
    async with ctx.conn() as db:
        rows = await db.fetch(
            f"SELECT {_CONCLUSION_COLUMNS} FROM {ctx.table('conclusions')} "
            "WHERE bank_id = $1 AND observer = $2 AND observed = $3 AND embedding IS NOT NULL "
            f"AND deleted_at IS NULL{distance_clause}{clause} "
            f"ORDER BY embedding <=> $4::vector LIMIT ${n}",
            bank_id,
            observer,
            observed,
            ctx.vector(embedding),
            *params,
            top_k,
        )
        return await _conclusions_from_rows(ctx, db, rows)


def _normalize_content(content: str) -> str:
    """Normalize conclusion content for exact-match deduplication.

    Content is compared after trimming surrounding whitespace and lowercasing.

    The SQL filter in ``create_documents`` must stay in sync with this:
    ``lower(regexp_replace(content, '^\\s+|\\s+$', '', 'g'))``. Postgres' ``trim()`` only strips spaces,
    so a regex is used to match Python's ``str.strip()`` across all whitespace.
    """
    return content.strip().lower()


def _dedup_key(content: str, level: str, session_name: str | None) -> tuple[str, str, str | None]:
    """Build the exact-match dedup key for a conclusion.

    Dedup never crosses levels: a same-content conclusion at a different level is a different kind of
    record (an explicit fact is not interchangeable with a deductive conclusion that happens to share
    its text).

    For **explicit** conclusions dedup additionally never crosses sessions. Explicit conclusions are
    session-pure records of what was derived from that session's messages, so a repeat of the same fact
    in a different session must produce a new conclusion in that session rather than reinforce another
    session's row. Derived levels (deductive/inductive/contradiction) are consolidations and may still
    dedup across sessions.
    """
    return (
        _normalize_content(content),
        level,
        session_name if level == "explicit" else None,
    )


def _semantic_dup_filters(doc: schemas.DocumentCreate) -> dict[str, Any] | None:
    """Merge scope for semantic dedup: never across levels, never across sessions for explicit
    conclusions. None when the conclusion has no valid merge partner (session-less explicit)."""
    filters: dict[str, Any] = {"level": doc.level}
    if doc.level == "explicit":
        if doc.session_name is None:
            return None
        filters["session_name"] = doc.session_name
    return filters


def _matches_semantic_dup_filters(conclusion: Conclusion, filters: dict[str, Any]) -> bool:
    """In-Python mirror of ``_semantic_dup_filters`` applied as SQL."""
    return all(getattr(conclusion, key) == value for key, value in filters.items())


class SemanticRejectionResult(Enum):
    NOT_DUPLICATE = 0
    REPLACED_EXISTING = 1
    REJECTED = 2


def _tokens(text: str) -> set[str]:
    """Word and punctuation tokens (Honcho counts tiktoken tokens; the fork has no tokenizer dependency)."""
    return set(re.findall(r"\w+|[^\w\s]", text))


def _score_semantic_dup(doc: schemas.DocumentCreate, existing_doc: Conclusion) -> SemanticRejectionResult:
    """Pick the survivor between an incoming conclusion and its near-identical match.

    The richer text wins: tokens the other side lacks are weighted ten times a shared token, and a tie
    keeps the incoming conclusion.
    """
    tokens_new = _tokens(doc.content)
    tokens_existing = _tokens(existing_doc.content)
    unique_new = len(tokens_new - tokens_existing)
    unique_existing = len(tokens_existing - tokens_new)
    score_new = len(tokens_new) + (unique_new * 10)
    score_existing = len(tokens_existing) + (unique_existing * 10)
    if score_new >= score_existing:
        return SemanticRejectionResult.REPLACED_EXISTING
    return SemanticRejectionResult.REJECTED


async def _semantic_dup_candidates(
    ctx: PeopleContext,
    db: Any,
    documents: list[schemas.DocumentCreate],
    bank_id: str,
    *,
    observer: str,
    observed: str,
) -> list[list[str]]:
    """Resolve each conclusion's nearest dup candidate id; positionally aligned with *documents*."""
    candidates: list[list[str]] = [[] for _ in documents]
    for index, doc in enumerate(documents):
        filters = _semantic_dup_filters(doc)
        if filters is None or not doc.embedding:
            continue
        clause, params = _filter_clause(filters, 5)
        rows = await db.fetch(
            f"SELECT id FROM {ctx.table('conclusions')} "
            "WHERE bank_id = $1 AND observer = $2 AND observed = $3 AND embedding IS NOT NULL "
            f"AND deleted_at IS NULL AND embedding <=> $4::vector <= {_SEMANTIC_DUP_MAX_DISTANCE}{clause} "
            f"ORDER BY embedding <=> $4::vector LIMIT {_SEMANTIC_DUP_TOP_K}",
            bank_id,
            observer,
            observed,
            ctx.vector(doc.embedding),
            *params,
        )
        candidates[index] = [r["id"] for r in rows]
    return candidates


def _resolve_semantic_dup(
    doc: schemas.DocumentCreate,
    candidate_ids: list[str],
    candidates_by_id: dict[str, Conclusion],
) -> tuple[SemanticRejectionResult, Conclusion | None]:
    """Classify a semantic duplicate against prefetched candidates.

    Candidates were fetched for the batch as a whole, so the per-conclusion merge scope is applied here
    instead of in SQL. Ids stay in similarity order; the closest in-scope candidate wins.
    """
    filters = _semantic_dup_filters(doc)
    if filters is None:
        return SemanticRejectionResult.NOT_DUPLICATE, None

    for candidate_id in candidate_ids:
        existing_doc = candidates_by_id.get(candidate_id)
        if existing_doc is None:
            continue
        if not _matches_semantic_dup_filters(existing_doc, filters):
            continue
        return _score_semantic_dup(doc, existing_doc), existing_doc

    return SemanticRejectionResult.NOT_DUPLICATE, None


@dataclass(frozen=True, slots=True)
class _DocumentRowOp:
    kind: str  # "reinforce" or "replace"
    document_id: str
    incoming_times_derived: int = 1
    # When a reinforce skipped insert and the locked target is gone/deleted,
    # insert this conclusion instead of dropping it.
    fallback_document: schemas.DocumentCreate | None = None


@dataclass
class CreateDocumentsResult:
    created_documents: list[schemas.DocumentCreate] = field(default_factory=list)
    exact_dup_in_batch_count: int = 0
    exact_dup_existing_count: int = 0
    semantic_dup_rejected_count: int = 0
    semantic_dup_replaced_count: int = 0
    # The rows that were inserted, with their ids and premise ids, and the rows semantic dedup replaced
    # together with the id of the row that replaced each (so dream history can be written from them).
    created_conclusions: list[Conclusion] = field(default_factory=list)
    replaced_conclusions: list[tuple[Conclusion, str | None]] = field(default_factory=list)


async def _message_provenance(
    ctx: PeopleContext, db: Any, bank_id: str, documents: list[schemas.DocumentCreate]
) -> dict[str, tuple[str | None, datetime.datetime | None]]:
    """``source_url`` and ``said_at`` of every message the batch cites (by public id or by internal id)."""
    public_ids = list(dict.fromkeys(m for d in documents for m in (d.source_message_ids or [])))
    internal_ids = list(dict.fromkeys(m for d in documents for m in d.metadata.message_ids))
    out: dict[str, tuple[str | None, datetime.datetime | None]] = {}
    if public_ids:
        rows = await db.fetch(
            f"SELECT public_id, source_url, said_at FROM {ctx.table('messages')} "
            "WHERE bank_id = $1 AND public_id = ANY($2::text[])",
            bank_id,
            public_ids,
        )
        for row in rows:
            out[row["public_id"]] = (row["source_url"], row["said_at"])
    if internal_ids:
        rows = await db.fetch(
            f"SELECT id, source_url, said_at FROM {ctx.table('messages')} WHERE bank_id = $1 AND id = ANY($2::bigint[])",
            bank_id,
            internal_ids,
        )
        for row in rows:
            out[f"#{row['id']}"] = (row["source_url"], row["said_at"])
    return out


def _doc_provenance(
    doc: schemas.DocumentCreate,
    provenance: dict[str, tuple[str | None, datetime.datetime | None]],
    premises: dict[str, tuple[str | None, datetime.datetime | None]],
) -> tuple[str | None, datetime.datetime | None]:
    """The source of a new conclusion: what the caller set, else its first cited message, else its first premise."""
    if doc.source_url is not None or doc.said_at is not None:
        return doc.source_url, doc.said_at
    keys = list(doc.source_message_ids or []) + [f"#{m}" for m in doc.metadata.message_ids]
    for key in keys:
        found = provenance.get(key)
        if found and (found[0] is not None or found[1] is not None):
            return found
    for premise_id in doc.source_ids or []:
        found = premises.get(premise_id)
        if found and (found[0] is not None or found[1] is not None):
            return found
    return None, None


async def create_documents(
    ctx: PeopleContext,
    bank_id: str,
    documents: list[schemas.DocumentCreate],
    *,
    observer: str,
    observed: str,
    deduplicate: bool = False,
    conn: Any | None = None,
) -> CreateDocumentsResult:
    """Create multiple conclusions with optional duplicate detection.

    Exact duplicates (same text after trimming and lowercasing, same level, same session for explicit
    conclusions) are always collapsed into the existing row, which is reinforced (``times_derived``
    goes up and new citations are added). The ``deduplicate`` flag additionally enables semantic
    (cosine-similarity) dedup for conclusions that survive the exact check.

    Args:
        documents: List of conclusion creation schemas
        observer: Handle of the observing peer
        observed: Handle of the observed peer
        deduplicate: Enable semantic duplicate detection
        conn: Optional open connection to join the caller's transaction

    Returns:
        The result: the DocumentCreate schemas that were actually inserted (excludes duplicates and
        failures), the dedup counts and the created and replaced rows.
    """
    result = CreateDocumentsResult()
    if not documents:
        return result

    async with connection(ctx, conn) as db:
        async with db.transaction():
            await _create_documents(ctx, db, bank_id, documents, observer, observed, deduplicate, result)
    return result


async def _create_documents(
    ctx: PeopleContext,
    db: Any,
    bank_id: str,
    documents: list[schemas.DocumentCreate],
    observer: str,
    observed: str,
    deduplicate: bool,
    result: CreateDocumentsResult,
) -> None:
    # Resolve semantic dup candidates for the whole batch up front, then load every candidate row in
    # one query. The per-conclusion decision in the loop below is pure Python.
    semantic_candidate_ids: list[list[str]] = [[] for _ in documents]
    semantic_candidates_by_id: dict[str, Conclusion] = {}
    if deduplicate:
        semantic_candidate_ids = await _semantic_dup_candidates(
            ctx, db, documents, bank_id, observer=observer, observed=observed
        )
        unique_ids = list({doc_id for ids in semantic_candidate_ids for doc_id in ids})
        semantic_candidates_by_id = {
            c.id: c for c in await fetch_documents_by_ids(ctx, bank_id, observer, observed, unique_ids, conn=db)
        }

    # exact-content dedup (independent of `deduplicate`): pre-fetch existing live conclusions whose
    # normalized content matches anything in this batch, scoped to (bank, observer, observed). The SQL
    # normalization must mirror _normalize_content. Matching is further scoped per-conclusion by level
    # (always) and session (for explicit conclusions) via _dedup_key.
    batch_normalized = list({_normalize_content(d.content) for d in documents})
    existing_by_key: dict[tuple[str, str, str | None], Conclusion] = {}
    existing_rows = await db.fetch(
        f"SELECT {_CONCLUSION_COLUMNS} FROM {ctx.table('conclusions')} "
        "WHERE bank_id = $1 AND observer = $2 AND observed = $3 AND deleted_at IS NULL "
        "AND lower(regexp_replace(content, '^\\s+|\\s+$', '', 'g')) = ANY($4::text[])",
        bank_id,
        observer,
        observed,
        batch_normalized,
    )
    for existing_doc in await _conclusions_from_rows(ctx, db, existing_rows):
        # If multiple historical rows share a dedup key, reinforcing one is sufficient; keep the first.
        existing_by_key.setdefault(
            _dedup_key(existing_doc.content, existing_doc.level, existing_doc.session_name), existing_doc
        )

    # First conclusion seen per dedup key, so exact duplicates within a single inference call collapse
    # to one conclusion. A collapsed duplicate's citations fold into the first one, and into its pending
    # row if any.
    first_in_batch: dict[tuple[str, str, str | None], schemas.DocumentCreate] = {}
    pending_by_key: dict[tuple[str, str, str | None], schemas.DocumentCreate] = {}
    # Existing row id -> the incoming conclusion replacing it, by its dedup key.
    replaced_by: dict[str, tuple[schemas.DocumentCreate, tuple[str, str, str | None]]] = {}
    row_ops: list[_DocumentRowOp] = []
    pending_times_derived: dict[str, int] = {}
    new_docs: list[schemas.DocumentCreate] = []

    for index, doc in enumerate(documents):
        # Session-purity invariant: an explicit conclusion must always carry the session it was derived
        # from. Refuse to write session-less explicit conclusions rather than silently minting global
        # explicit memory.
        if doc.level == "explicit" and doc.session_name is None:
            logger.error(
                "Refusing to create explicit conclusion without session_name in %s/%s/%s (session-purity "
                "invariant): %r",
                bank_id,
                observer,
                observed,
                doc.content[:80],
            )
            continue

        dedup_key = _dedup_key(doc.content, doc.level, doc.session_name)

        # Exact-match dedup, always on:
        # 1) collapse exact duplicates within this batch (drop silently).
        first_doc = first_in_batch.get(dedup_key)
        if first_doc is not None:
            result.exact_dup_in_batch_count += 1
            first_doc.source_message_ids = _merge_citations(first_doc.source_message_ids, doc.source_message_ids)
            continue
        first_in_batch[dedup_key] = doc

        # 2) drop exact duplicates of an existing live conclusion, recording the re-derivation as
        #    reinforcement on the existing row.
        existing_match = existing_by_key.get(dedup_key)
        if existing_match is not None:
            current_td = pending_times_derived.get(existing_match.id, existing_match.times_derived)
            pending_times_derived[existing_match.id] = max(current_td + 1, doc.times_derived)
            row_ops.append(_DocumentRowOp("reinforce", existing_match.id, doc.times_derived, fallback_document=doc))
            result.exact_dup_existing_count += 1
            continue

        if deduplicate:
            duplicate_result, existing_dup = _resolve_semantic_dup(
                doc, semantic_candidate_ids[index], semantic_candidates_by_id
            )
            if duplicate_result is SemanticRejectionResult.REPLACED_EXISTING and existing_dup is not None:
                current_td = pending_times_derived.get(existing_dup.id, existing_dup.times_derived)
                doc.times_derived = max(doc.times_derived, current_td + 1)
                pending_times_derived[existing_dup.id] = doc.times_derived
                # The replacement inherits the replaced row's citations, as it inherits its derivation count.
                doc.source_message_ids = _merge_citations(existing_dup.source_message_ids, doc.source_message_ids)
                row_ops.append(_DocumentRowOp("replace", existing_dup.id))
                replaced_by.setdefault(existing_dup.id, (doc, dedup_key))
                result.semantic_dup_replaced_count += 1
            elif duplicate_result is SemanticRejectionResult.REJECTED and existing_dup is not None:
                current_td = pending_times_derived.get(existing_dup.id, existing_dup.times_derived)
                pending_times_derived[existing_dup.id] = max(current_td + 1, doc.times_derived)
                row_ops.append(_DocumentRowOp("reinforce", existing_dup.id, doc.times_derived, fallback_document=doc))
                result.semantic_dup_rejected_count += 1
                continue

        new_docs.append(doc)
        pending_by_key[dedup_key] = doc

    # A reinforce aimed at a row this batch also replaces would land on the deleted row, or be skipped,
    # depending on op order. Its citations belong to the replacement.
    for op in row_ops:
        if op.kind != "reinforce" or op.fallback_document is None:
            continue
        replacement = replaced_by.get(op.document_id)
        if replacement is None:
            continue
        replacement_doc, _ = replacement
        replacement_doc.source_message_ids = _merge_citations(
            replacement_doc.source_message_ids, op.fallback_document.source_message_ids
        )

    replaced_rows: list[Conclusion] = []
    fallback_docs = await _apply_document_row_updates(
        ctx, db, bank_id, row_ops, observer=observer, observed=observed, replaced_out=replaced_rows
    )
    new_docs.extend(fallback_docs)

    created = await _insert_conclusions(ctx, db, bank_id, new_docs, observer, observed)
    result.created_documents.extend(new_docs)
    result.created_conclusions.extend(created)
    created_by_key = {_dedup_key(c.content, c.level, c.session_name): c.id for c in created}
    for old in replaced_rows:
        replacement_entry = replaced_by.get(old.id)
        replacement_id = created_by_key.get(replacement_entry[1]) if replacement_entry else None
        result.replaced_conclusions.append((old, replacement_id))


async def _insert_conclusions(
    ctx: PeopleContext,
    db: Any,
    bank_id: str,
    docs: list[schemas.DocumentCreate],
    observer: str,
    observed: str,
) -> list[Conclusion]:
    """Insert new conclusion rows with their premise and message edges; returns them with ids."""
    if not docs:
        return []
    provenance = await _message_provenance(ctx, db, bank_id, docs)
    premise_ids = list({sid for d in docs for sid in build_source_links(d.source_ids)})
    premise_rows = (
        await db.fetch(
            f"SELECT id, source_url, said_at FROM {ctx.table('conclusions')} "
            "WHERE bank_id = $1 AND id = ANY($2::text[])",
            bank_id,
            premise_ids,
        )
        if premise_ids
        else []
    )
    premises = {r["id"]: (r["source_url"], r["said_at"]) for r in premise_rows}

    created: list[Conclusion] = []
    for doc in docs:
        conclusion_id = generate_nanoid()
        source_url, said_at = _doc_provenance(doc, provenance, premises)
        row = await db.fetchrow(
            f"INSERT INTO {ctx.table('conclusions')} "
            "(bank_id, id, observer, observed, level, content, embedding, times_derived, session_name, "
            "internal_metadata, source_url, said_at) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7::vector, $8, $9, $10::jsonb, $11, $12) "
            f"RETURNING {_CONCLUSION_COLUMNS}",
            bank_id,
            conclusion_id,
            observer,
            observed,
            doc.level,
            doc.content,
            ctx.vector(doc.embedding) if doc.embedding else None,
            doc.times_derived,
            doc.session_name,
            to_jsonb(doc.metadata.model_dump(exclude_none=True)),
            source_url,
            said_at,
        )
        premise_links = build_source_links(doc.source_ids)
        message_links = build_source_message_links(doc.source_message_ids)
        for kind, links in (("conclusion", premise_links), ("message", message_links)):
            for position, source_id in enumerate(links):
                await db.execute(
                    f"INSERT INTO {ctx.table('conclusion_sources')} (bank_id, conclusion_id, source_id, kind, position) "
                    "VALUES ($1, $2, $3, $4, $5) ON CONFLICT DO NOTHING",
                    bank_id,
                    conclusion_id,
                    source_id,
                    kind,
                    position,
                )
        [conclusion] = await _conclusions_from_rows(ctx, db, [row])
        created.append(conclusion)
    return created


async def _apply_document_row_updates(
    ctx: PeopleContext,
    db: Any,
    bank_id: str,
    ops: list[_DocumentRowOp],
    *,
    observer: str,
    observed: str,
    replaced_out: list[Conclusion] | None = None,
) -> list[schemas.DocumentCreate]:
    """Lock target rows by id, apply ops, return fallbacks for vanished targets.

    Rows a `replace` op soft-deletes are appended to ``replaced_out``.
    """
    if not ops:
        return []
    # Deadlock fix: lock in id order.
    ids = sorted({op.document_id for op in ops})
    locked_rows = await db.fetch(
        f"SELECT {_CONCLUSION_COLUMNS} FROM {ctx.table('conclusions')} "
        "WHERE bank_id = $1 AND id = ANY($2::text[]) AND observer = $3 AND observed = $4 ORDER BY id FOR UPDATE",
        bank_id,
        ids,
        observer,
        observed,
    )
    locked = {c.id: c for c in await _conclusions_from_rows(ctx, db, locked_rows)}
    # Current citations of the reinforced rows, read under the row lock.
    existing_citations: dict[str, list[str]] = {
        op.document_id: list(locked[op.document_id].source_message_ids or [])
        for op in ops
        if op.kind == "reinforce" and op.document_id in locked
    }
    fallbacks: list[schemas.DocumentCreate] = []
    stale_at_lock = {
        op.document_id for op in ops if (row := locked.get(op.document_id)) is None or row.deleted_at is not None
    }
    deleted_now: set[str] = set()
    for op in ops:
        row = locked.get(op.document_id)
        if op.kind == "replace":
            if row is not None and row.deleted_at is None and row.id not in deleted_now:
                await db.execute(
                    f"UPDATE {ctx.table('conclusions')} SET deleted_at = now() WHERE bank_id = $1 AND id = $2",
                    bank_id,
                    row.id,
                )
                deleted_now.add(row.id)
                if replaced_out is not None:
                    replaced_out.append(row)
            continue
        # reinforce
        if op.document_id in stale_at_lock:
            if op.fallback_document is not None:
                fallbacks.append(op.fallback_document)
            continue
        if row is None or row.deleted_at is not None or row.id in deleted_now:
            # An earlier op in this batch replaced this row.
            continue
        row.times_derived = max(row.times_derived + 1, op.incoming_times_derived)
        await db.execute(
            f"UPDATE {ctx.table('conclusions')} SET times_derived = $3 WHERE bank_id = $1 AND id = $2",
            bank_id,
            row.id,
            row.times_derived,
        )
        if op.fallback_document is not None:
            await _append_citations(
                ctx, db, bank_id, row.id, existing_citations[row.id], op.fallback_document.source_message_ids
            )
    return fallbacks


async def _append_citations(
    ctx: PeopleContext,
    db: Any,
    bank_id: str,
    conclusion_id: str,
    existing: list[str],
    incoming: list[str] | None,
) -> None:
    """Add the incoming citations a reinforced row lacks, after its own.

    ``existing`` is updated in place, so a second reinforce of the same row in one batch sees the first
    one's additions.
    """
    for message_id in build_source_message_links(incoming):
        if message_id in existing:
            continue
        await db.execute(
            f"INSERT INTO {ctx.table('conclusion_sources')} (bank_id, conclusion_id, source_id, kind, position) "
            "VALUES ($1, $2, $3, 'message', $4) ON CONFLICT DO NOTHING",
            bank_id,
            conclusion_id,
            message_id,
            len(existing),
        )
        existing.append(message_id)


async def delete_document(
    ctx: PeopleContext,
    bank_id: str,
    document_id: str,
    *,
    observer: str,
    observed: str,
    session_name: str | None = None,
    conn: Any | None = None,
) -> None:
    """Soft-delete a conclusion by id.

    Sets the deleted_at timestamp to mark the conclusion as deleted.

    Raises:
        ResourceNotFoundException: If the conclusion is not found or doesn't match criteria
    """
    deleted = await delete_documents(
        ctx, bank_id, [document_id], observer=observer, observed=observed, session_name=session_name, conn=conn
    )
    if not deleted:
        raise ResourceNotFoundException(
            f"Document {document_id} not found or does not belong to the specified collection/session"
        )


async def delete_documents(
    ctx: PeopleContext,
    bank_id: str,
    document_ids: Sequence[str],
    *,
    observer: str,
    observed: str,
    session_name: str | None = None,
    conn: Any | None = None,
) -> list[tuple[str, str]]:
    """Soft-delete multiple conclusions in a single UPDATE ... RETURNING statement.

    Returns (id, level) tuples for rows that actually got deleted -- rows that matched the
    bank/observer/observed filter and were not already soft-deleted. Ids that didn't match are silently
    skipped; callers can diff the returned ids against the input to detect misses.
    """
    if not document_ids:
        return []
    params: list[Any] = [bank_id, list(document_ids), observer, observed]
    extra = ""
    if session_name is not None:
        params.append(session_name)
        extra = " AND session_name = $5"
    async with connection(ctx, conn) as db:
        rows = await db.fetch(
            f"UPDATE {ctx.table('conclusions')} SET deleted_at = now() "
            "WHERE bank_id = $1 AND id = ANY($2::text[]) AND observer = $3 AND observed = $4 "
            f"AND deleted_at IS NULL{extra} RETURNING id, level",
            *params,
        )
    return [(row["id"], row["level"]) for row in rows]


async def delete_document_by_id(ctx: PeopleContext, bank_id: str, document_id: str, *, conn: Any | None = None) -> None:
    """Soft-delete a conclusion by id and bank.

    Raises:
        ResourceNotFoundException: If no live conclusion with this id exists in the bank
    """
    async with connection(ctx, conn) as db:
        rows = await db.fetch(
            f"UPDATE {ctx.table('conclusions')} SET deleted_at = now() "
            "WHERE bank_id = $1 AND id = $2 AND deleted_at IS NULL RETURNING id",
            bank_id,
            document_id,
        )
    if not rows:
        raise ResourceNotFoundException(f"Document {document_id} not found or does not belong to bank {bank_id}")
