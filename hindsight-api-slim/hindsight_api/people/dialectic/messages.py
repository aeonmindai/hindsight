# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/crud/message.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""
Read paths over ``people_messages`` for the dialectic's message tools.

Ported from the search half of Honcho's ``crud/message.py``: scoping a query to the sessions an
observer belongs to (and to a caller's session allowlist), conversation snippets with surrounding
context, exact-text search, date ranges and the messages around a conclusion's sources.

Differences from Honcho:

* Honcho's ``search_messages`` ranks messages by vector similarity against ``message_embeddings``.
  The people tables hold no message embeddings, so ``search_messages`` and
  ``search_messages_temporal`` rank by full-text relevance (``ts_rank_cd`` over the GIN index on
  ``to_tsvector('english', content)``). Any word of the query may match; messages matching more of
  the query's words rank higher.
* A message carries ``said_at`` (when it was said, from its source) next to ``created_at`` (when it
  was stored). Dates, ordering and the timestamps shown to the model use ``COALESCE(said_at,
  created_at)``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..context import PeopleContext
from .formatting import escape_ilike_pattern, format_new_turn_with_timestamp

# Output budget of one message tool call, in characters (Honcho's settings.LLM.MAX_TOOL_OUTPUT_CHARS).
MAX_TOOL_OUTPUT_CHARS = 10_000
# Longest message body one tool shows, in characters (Honcho's agent_tools truncation).
MAX_MESSAGE_CONTENT_CHARS = 2_000
# Most words of a query used for ranked text search.
MAX_QUERY_TERMS = 32

_MESSAGE_COLUMNS = (
    "m.id, m.public_id, m.session_name, m.peer_name, m.content, m.seq_in_session, "
    "m.created_at, m.said_at, m.source_url, m.channel, m.token_count"
)
_WHEN = "COALESCE(m.said_at, m.created_at)"
_TS_VECTOR = "to_tsvector('english', m.content)"
_WORD = re.compile(r"[^\W_]+")


@dataclass(frozen=True)
class MessageRow:
    """One row of ``people_messages``."""

    id: int
    public_id: str
    session_name: str
    peer_name: str
    content: str
    seq_in_session: int
    created_at: datetime
    said_at: datetime | None
    source_url: str | None
    channel: str | None
    token_count: int

    @property
    def when(self) -> datetime:
        """When the message was said, falling back to when it was stored."""
        return self.said_at or self.created_at


Snippet = tuple[list[MessageRow], list[MessageRow]]


def _row(record: Any) -> MessageRow:
    return MessageRow(
        id=record["id"],
        public_id=record["public_id"],
        session_name=record["session_name"],
        peer_name=record["peer_name"],
        content=record["content"],
        seq_in_session=record["seq_in_session"],
        created_at=record["created_at"],
        said_at=record["said_at"],
        source_url=record["source_url"],
        channel=record["channel"],
        token_count=record["token_count"],
    )


def _scope_clauses(
    ctx: PeopleContext,
    params: list[Any],
    session_name: str | None,
    session_allowlist: list[str] | None,
    observer: str | None,
    session_column: str = "m.session_name",
) -> tuple[list[str], bool]:
    """SQL-side session scope (Honcho's ``resolve_session_scope_clauses``).

    Returns ``(clauses, deny)``. ``clauses`` are ANDed onto the caller's statement and append their
    values to ``params``; ``deny`` means the caller must return an empty result without querying.
    An empty allowlist fails closed. ``params[0]`` must be the bank id.
    """
    if session_name:
        # Fail closed when the allowlist forbids the pinned session.
        if session_allowlist is not None and session_name not in session_allowlist:
            return [], True
        return [], False

    clauses: list[str] = []
    if observer is not None:
        params.append(observer)
        clauses.append(
            f"EXISTS (SELECT 1 FROM {ctx.table('session_peers')} sp "
            f"WHERE sp.bank_id = $1 AND sp.peer_name = ${len(params)} AND sp.session_name = {session_column})"
        )
    if session_allowlist is not None:
        if not session_allowlist:
            return [], True
        params.append(list(session_allowlist))
        clauses.append(f"{session_column} = ANY(${len(params)}::text[])")
    return clauses, False


def _tsquery(query: str) -> str | None:
    """An OR query over the words of ``query`` for ``to_tsquery``, or None when it has no words."""
    words: list[str] = []
    for word in _WORD.findall(query.lower()):
        if len(word) > 1 and word not in words:
            words.append(word)
    return " | ".join(words[:MAX_QUERY_TERMS]) or None


async def build_merged_snippets(
    ctx: PeopleContext,
    bank_id: str,
    matched_messages: list[MessageRow],
    context_window: int,
) -> list[Snippet]:
    """
    Group matched messages by session, merge overlapping context ranges, and fetch context.

    Takes a list of matched messages and builds conversation snippets by:
    1. Grouping matches by session name
    2. Sorting matches within each session by sequence number
    3. Merging overlapping context windows to avoid duplicate context
    4. Fetching the full context for each merged range from the database

    Returns:
        List of tuples: (matched_messages_in_range, context_messages)
        Each tuple represents a snippet where context_messages includes all messages
        in the merged range (including the matched messages), ordered chronologically.
    """
    if not matched_messages:
        return []

    session_matches: dict[str, list[MessageRow]] = {}
    for msg in matched_messages:
        session_matches.setdefault(msg.session_name, []).append(msg)

    session_ranges: dict[str, list[tuple[int, int, list[MessageRow]]]] = {}
    for session_key, matches in session_matches.items():
        matches.sort(key=lambda m: m.seq_in_session)

        merged_ranges: list[tuple[int, int, list[MessageRow]]] = []
        for match in matches:
            start = match.seq_in_session - context_window
            end = match.seq_in_session + context_window

            if merged_ranges and start <= merged_ranges[-1][1] + 1:
                prev_start, prev_end, prev_matches = merged_ranges[-1]
                merged_ranges[-1] = (prev_start, max(prev_end, end), [*prev_matches, match])
            else:
                merged_ranges.append((start, end, [match]))

        session_ranges[session_key] = merged_ranges

    # One statement fetches every (session, range) pair; the merged ranges of a session never overlap.
    range_sessions: list[str] = []
    range_starts: list[int] = []
    range_ends: list[int] = []
    for sess_name, merged_ranges in session_ranges.items():
        for start_seq, end_seq, _ in merged_ranges:
            range_sessions.append(sess_name)
            range_starts.append(start_seq)
            range_ends.append(end_seq)

    async with ctx.conn() as conn:
        records = await conn.fetch(
            f"""
            SELECT {_MESSAGE_COLUMNS}
            FROM {ctx.table("messages")} m
            JOIN unnest($2::text[], $3::bigint[], $4::bigint[]) AS r(session_name, lo, hi)
              ON m.session_name = r.session_name AND m.seq_in_session BETWEEN r.lo AND r.hi
            WHERE m.bank_id = $1 AND m.deleted_at IS NULL
            ORDER BY m.session_name ASC, m.seq_in_session ASC
            """,
            bank_id,
            range_sessions,
            range_starts,
            range_ends,
        )

    by_session: dict[str, list[MessageRow]] = {}
    for record in records:
        message = _row(record)
        by_session.setdefault(message.session_name, []).append(message)

    snippets: list[Snippet] = []
    for session_key, merged_ranges in session_ranges.items():
        all_context_messages = by_session.get(session_key, [])
        for start_seq, end_seq, range_matches in merged_ranges:
            context_messages = [msg for msg in all_context_messages if start_seq <= msg.seq_in_session <= end_seq]
            snippets.append((range_matches, context_messages))

    return snippets


async def get_messages_id_range(
    ctx: PeopleContext,
    bank_id: str,
    session_name: str,
    start_id: int = 0,
    end_id: int | None = None,
    token_limit: int | None = None,
) -> list[MessageRow]:
    """
    Get messages from a session by primary key ID range.
    If end_id is not provided, all messages after and including start_id will be returned.
    If start_id is not provided, start will be beginning of session.

    Note: list is *inclusive* of the start_id message and exclusive of end_id.

    Args:
        ctx: People context
        bank_id: Bank identifier
        session_name: Name of the session
        start_id: Primary key ID of the first message to return
        end_id: Primary key ID of the last message (exclusive)
        token_limit: If set, return only the most recent messages whose token
            counts add up to at most this limit (the walk stops at the first message that does not fit)

    Returns:
        List of messages in chronological order
    """
    if start_id < 0 or (end_id is not None and (start_id >= end_id or end_id <= 0)):
        return []

    params: list[Any] = [bank_id, session_name, start_id]
    id_clause = "m.id >= $3"
    if end_id:
        params.append(end_id)
        id_clause += " AND m.id < $4"

    async with ctx.conn() as conn:
        records = await conn.fetch(
            f"""
            SELECT {_MESSAGE_COLUMNS}
            FROM {ctx.table("messages")} m
            WHERE m.bank_id = $1 AND m.session_name = $2 AND m.deleted_at IS NULL AND {id_clause}
            ORDER BY m.id DESC
            """,
            *params,
        )

    newest_first = [_row(record) for record in records]
    if token_limit:
        kept: list[MessageRow] = []
        running_token_sum = 0
        for message in newest_first:
            running_token_sum += message.token_count
            if running_token_sum > token_limit:
                break
            kept.append(message)
        newest_first = kept
    newest_first.reverse()
    return newest_first


async def get_observation_context(
    ctx: PeopleContext,
    bank_id: str,
    session_name: str | None,
    message_ids: list[str],
    observer: str | None = None,
    session_allowlist: list[str] | None = None,
) -> list[MessageRow]:
    """
    Retrieve messages for given message IDs along with surrounding context.

    Takes message IDs (from a conclusion's message ids) and retrieves those messages plus the
    messages immediately before and after each one, in the target's own session, to provide
    conversation context.

    Args:
        ctx: People context
        bank_id: Bank identifier
        session_name: Session identifier (optional); pins the query to one session
        message_ids: Message public ids to retrieve
        observer: When provided and session_name is None, scope results to sessions this peer belongs to
        session_allowlist: Optional session allowlist. None is unrestricted; an empty list fails closed
            (empty result); a populated list is intersected with the observer's session scope

    Returns:
        Messages in session and sequence order, including the requested messages and surrounding context
    """
    if not message_ids:
        return []

    params: list[Any] = [bank_id]
    scope_clauses, deny = _scope_clauses(ctx, params, session_name, session_allowlist, observer)
    if deny:
        return []

    params.append(list(message_ids))
    ids_param = len(params)
    where_extra = "".join(f" AND {clause}" for clause in scope_clauses)
    if session_name:
        params.append(session_name)
        where_extra += f" AND m.session_name = ${len(params)}"

    async with ctx.conn() as conn:
        records = await conn.fetch(
            f"""
            WITH target_seqs AS (
                SELECT m.session_name, m.seq_in_session
                FROM {ctx.table("messages")} m
                WHERE m.bank_id = $1 AND m.public_id = ANY(${ids_param}::text[]) AND m.deleted_at IS NULL{where_extra}
            )
            SELECT {_MESSAGE_COLUMNS}
            FROM {ctx.table("messages")} m
            WHERE m.bank_id = $1 AND m.deleted_at IS NULL{where_extra}
              AND EXISTS (
                SELECT 1 FROM target_seqs t
                WHERE t.session_name = m.session_name
                  AND m.seq_in_session BETWEEN t.seq_in_session - 1 AND t.seq_in_session + 1
              )
            ORDER BY m.session_name ASC, m.seq_in_session ASC
            """,
            *params,
        )
    return [_row(record) for record in records]


async def _ranked_matches(
    ctx: PeopleContext,
    bank_id: str,
    session_name: str | None,
    query: str,
    *,
    limit: int,
    after_date: datetime | None,
    before_date: datetime | None,
    observer: str | None,
    session_allowlist: list[str] | None,
) -> list[MessageRow]:
    tsquery = _tsquery(query)
    if tsquery is None:
        return []
    params: list[Any] = [bank_id]
    scope_clauses, deny = _scope_clauses(ctx, params, session_name, session_allowlist, observer)
    if deny:
        return []
    params.append(tsquery)
    query_param = len(params)
    where = [
        "m.bank_id = $1",
        "m.deleted_at IS NULL",
        f"{_TS_VECTOR} @@ to_tsquery('english', ${query_param})",
        *scope_clauses,
    ]
    if session_name:
        params.append(session_name)
        where.append(f"m.session_name = ${len(params)}")
    if after_date:
        params.append(after_date)
        where.append(f"{_WHEN} >= ${len(params)}")
    if before_date:
        params.append(before_date)
        where.append(f"{_WHEN} <= ${len(params)}")
    params.append(limit)
    async with ctx.conn() as conn:
        records = await conn.fetch(
            f"""
            SELECT {_MESSAGE_COLUMNS}
            FROM {ctx.table("messages")} m
            WHERE {" AND ".join(where)}
            ORDER BY ts_rank_cd({_TS_VECTOR}, to_tsquery('english', ${query_param})) DESC, {_WHEN} DESC, m.id DESC
            LIMIT ${len(params)}
            """,
            *params,
        )
    return [_row(record) for record in records]


async def search_messages(
    ctx: PeopleContext,
    bank_id: str,
    session_name: str | None,
    query: str,
    limit: int = 10,
    context_window: int = 2,
    observer: str | None = None,
    session_allowlist: list[str] | None = None,
) -> list[Snippet]:
    """
    Search for messages by ranked full-text relevance and return conversation snippets.

    Each result includes matched messages plus surrounding context. Overlapping
    snippets within the same session are merged to avoid repetition.

    Args:
        ctx: People context
        bank_id: Bank identifier
        session_name: Name of the session (optional)
        query: Search query text
        limit: Maximum number of matching messages to return
        context_window: Number of messages before/after each match to include
        observer: When provided and session_name is None, scope results to sessions this peer belongs to
        session_allowlist: Optional session allowlist. None is unrestricted; an empty list fails closed

    Returns:
        List of tuples: (matched_messages, context_messages)
        Each snippet may contain multiple matches if they were close together.
    """
    matched = await _ranked_matches(
        ctx,
        bank_id,
        session_name,
        query,
        limit=limit,
        after_date=None,
        before_date=None,
        observer=observer,
        session_allowlist=session_allowlist,
    )
    return await build_merged_snippets(ctx, bank_id, matched, context_window)


async def search_messages_temporal(
    ctx: PeopleContext,
    bank_id: str,
    session_name: str | None,
    query: str,
    after_date: datetime | None = None,
    before_date: datetime | None = None,
    limit: int = 10,
    context_window: int = 2,
    observer: str | None = None,
    session_allowlist: list[str] | None = None,
) -> list[Snippet]:
    """
    Search for messages by ranked full-text relevance with optional date filtering.

    Combines text relevance with time constraints. Use after_date to find recent mentions,
    or before_date to find what was said before a certain point.
    """
    matched = await _ranked_matches(
        ctx,
        bank_id,
        session_name,
        query,
        limit=limit,
        after_date=after_date,
        before_date=before_date,
        observer=observer,
        session_allowlist=session_allowlist,
    )
    return await build_merged_snippets(ctx, bank_id, matched, context_window)


async def grep_messages(
    ctx: PeopleContext,
    bank_id: str,
    session_name: str | None,
    text: str,
    limit: int = 10,
    context_window: int = 2,
    observer: str | None = None,
    session_allowlist: list[str] | None = None,
) -> list[Snippet]:
    """
    Search for messages containing specific text (case-insensitive substring match).
    Unlike ranked search, this finds EXACT text matches. Useful for finding
    specific names, dates, phrases, or keywords.

    Returns:
        List of tuples: (matched_messages, context_messages)
        Each snippet may contain multiple matches if they were close together.
    """
    params: list[Any] = [bank_id]
    scope_clauses, deny = _scope_clauses(ctx, params, session_name, session_allowlist, observer)
    if deny:
        return []
    params.append(f"%{escape_ilike_pattern(text)}%")
    where = ["m.bank_id = $1", "m.deleted_at IS NULL", f"m.content ILIKE ${len(params)} ESCAPE '\\'", *scope_clauses]
    if session_name:
        params.append(session_name)
        where.append(f"m.session_name = ${len(params)}")
    params.append(limit)
    async with ctx.conn() as conn:
        records = await conn.fetch(
            f"""
            SELECT {_MESSAGE_COLUMNS}
            FROM {ctx.table("messages")} m
            WHERE {" AND ".join(where)}
            ORDER BY {_WHEN} DESC, m.id DESC
            LIMIT ${len(params)}
            """,
            *params,
        )
    matched = [_row(record) for record in records]
    return await build_merged_snippets(ctx, bank_id, matched, context_window)


async def get_messages_by_date_range(
    ctx: PeopleContext,
    bank_id: str,
    session_name: str | None,
    after_date: datetime | None = None,
    before_date: datetime | None = None,
    limit: int = 20,
    order: str = "desc",
    observer: str | None = None,
    session_allowlist: list[str] | None = None,
) -> list[MessageRow]:
    """
    Get messages within a date range.

    Args:
        ctx: People context
        bank_id: Bank identifier
        session_name: Name of the session (optional - searches all sessions if None)
        after_date: Return messages after this datetime
        before_date: Return messages before this datetime
        limit: Maximum messages to return
        order: Sort order - 'asc' for oldest first, 'desc' for newest first
        observer: When provided and session_name is None, scope results to sessions this peer belongs to
        session_allowlist: Optional session allowlist. None is unrestricted; an empty list fails closed

    Returns:
        List of messages within the date range
    """
    params: list[Any] = [bank_id]
    scope_clauses, deny = _scope_clauses(ctx, params, session_name, session_allowlist, observer)
    if deny:
        return []
    where = ["m.bank_id = $1", "m.deleted_at IS NULL", *scope_clauses]
    if session_name:
        params.append(session_name)
        where.append(f"m.session_name = ${len(params)}")
    if after_date:
        params.append(after_date)
        where.append(f"{_WHEN} >= ${len(params)}")
    if before_date:
        params.append(before_date)
        where.append(f"{_WHEN} <= ${len(params)}")
    direction = "ASC" if order == "asc" else "DESC"
    params.append(limit)
    async with ctx.conn() as conn:
        records = await conn.fetch(
            f"""
            SELECT {_MESSAGE_COLUMNS}
            FROM {ctx.table("messages")} m
            WHERE {" AND ".join(where)}
            ORDER BY {_WHEN} {direction}, m.id {direction}
            LIMIT ${len(params)}
            """,
            *params,
        )
    return [_row(record) for record in records]


# ---------------------------------------------------------------------------
# Formatting for the model (Honcho's agent_tools helpers)
# ---------------------------------------------------------------------------


def truncate_output(output: str, max_chars: int = MAX_TOOL_OUTPUT_CHARS) -> tuple[str, bool, int]:
    """Cap a tool output; returns ``(text, was_truncated, original_chars)``."""
    original_chars = len(output)
    if original_chars <= max_chars:
        return output, False, original_chars
    notice = f"\n\n[Output truncated: showing the first {max_chars} of {original_chars} characters]"
    return output[:max_chars] + notice, True, original_chars


def truncate_message_content(content: str, max_chars: int = MAX_MESSAGE_CONTENT_CHARS) -> str:
    """Cap one message body."""
    if len(content) <= max_chars:
        return content
    return content[:max_chars] + "... [truncated]"


def extract_pattern_snippet(content: str, pattern: str, max_chars: int = MAX_MESSAGE_CONTENT_CHARS) -> str:
    """The part of a long message around the first match of ``pattern`` (the whole message when it is short)."""
    if len(content) <= max_chars:
        return content
    index = content.lower().find(pattern.lower())
    if index < 0:
        return truncate_message_content(content, max_chars)
    half = max(0, (max_chars - len(pattern)) // 2)
    start = max(0, index - half)
    end = min(len(content), index + len(pattern) + half)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(content) else ""
    return f"{prefix}{content[start:end]}{suffix}"


def format_message(message: MessageRow, content: str | None = None) -> str:
    """One message as a line the model can cite: its turn, then where it was said."""
    turn = format_new_turn_with_timestamp(
        content if content is not None else truncate_message_content(message.content), message.when, message.peer_name
    )
    return f"{turn}\n{message_source_line(message)}"


def message_source_line(message: MessageRow) -> str:
    """The citation line printed under a message."""
    parts = [f"id: {message.public_id}"]
    if message.source_url:
        parts.append(f"source: {message.source_url}")
    parts.append(f"said_at: {message.when.isoformat()}")
    if message.channel:
        parts.append(f"channel: {message.channel}")
    return "  [" + ", ".join(parts) + "]"


def _omission_notice(shown: int, total: int) -> str:
    """Notice appended when whole-snippet truncation drops some snippets."""
    omitted = total - shown
    if omitted <= 0:
        return ""
    return f"\n\n[{shown} of {total} snippets shown - {omitted} omitted to fit output budget]"


def format_message_snippets(
    snippets: Sequence[Snippet], desc: str, *, pattern: str | None = None, max_chars: int = MAX_TOOL_OUTPUT_CHARS
) -> tuple[str, bool, int]:
    """Format message snippets for output.

    Truncates at whole-snippet boundaries rather than mid-snippet: when the assembled output
    exceeds ``max_chars``, trailing snippets are dropped and a ``[N of M snippets shown]`` notice
    is appended, so the model sees that content was omitted and can narrow its query.

    ``pattern`` (grep) cuts long messages to the part around the match.

    Returns ``(text, was_truncated, original_chars_before_truncation)``.
    """
    snippet_texts: list[str] = []
    total_matches = sum(len(matches) for matches, _ in snippets)
    for i, (matches, context) in enumerate(snippets, 1):
        lines: list[str] = []
        for msg in context:
            body = extract_pattern_snippet(msg.content, pattern) if pattern else truncate_message_content(msg.content)
            lines.append(format_message(msg, body))
        sess = context[0].session_name if context else "unknown"
        snippet_texts.append(f"--- Snippet {i} (session: {sess}, {len(matches)} match(es)) ---\n" + "\n".join(lines))

    header = (
        f"Found {total_matches} matching messages in {len(snippets)} conversation snippets {desc}.\n"
        "These are raw messages. Cite one by its source URL and said_at time:\n\n"
    )
    full_output = header + "\n\n".join(snippet_texts)
    original_chars = len(full_output)
    if original_chars <= max_chars:
        return full_output, False, original_chars

    shown_texts: list[str] = []
    for text in snippet_texts:
        candidate = header + "\n\n".join([*shown_texts, text])
        remaining = len(snippet_texts) - len(shown_texts) - 1
        notice = _omission_notice(len(shown_texts) + 1, len(snippet_texts))
        if len(candidate) + (len(notice) if remaining else 0) <= max_chars:
            shown_texts.append(text)
        else:
            break

    if not shown_texts:
        # Even the first snippet exceeds the budget on its own: head-truncate the whole output.
        return truncate_output(full_output, max_chars)[0], True, original_chars

    content = header + "\n\n".join(shown_texts)
    if len(snippet_texts) - len(shown_texts):
        content += _omission_notice(len(shown_texts), len(snippet_texts))
    return content, True, original_chars
