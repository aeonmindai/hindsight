# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/utils/evidence.py
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/schemas/api.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""Collation of what the dialectic agent read while answering.

Evidence is gathered passively: read paths hand the rows they already loaded to
an accumulator, and nothing is re-queried when the response is built. The model
is never asked to cite its sources, which keeps collection deterministic and
free of model load, at the cost of over-reporting -- the agent may read a
conclusion it does not end up leaning on.

Because nothing is re-queried, evidence inherits the scoping of the reads that
produced it: whatever the bank, observer/observed pair and session allowlist
permitted the agent to see is exactly what can appear here.

Evidence is an audit surface and the source list of an answer. Conclusions carry their text, which is
short, model-written, and the thing being audited, while messages carry identity and provenance
(``source_url``, ``said_at``, ``channel``) alone: message content is unbounded, and including it
would inflate every response.

The fork adds provenance to every record: ``source_url`` and ``said_at`` on conclusions and
messages, and engine-recalled facts as a third kind of record. :meth:`Evidence.sources` flattens
the three into the list of sources an answer can cite.
"""

from __future__ import annotations

import datetime
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..conclusions import Conclusion
    from .messages import MessageRow


class EvidenceObservation(BaseModel):
    """A conclusion the dialectic agent read while answering."""

    id: str = Field(description="Conclusion ID")
    level: str = Field(description="Conclusion level: explicit, deductive, inductive, or contradiction")
    content: str = Field(description="The conclusion text")
    created_at: datetime.datetime = Field(description="When the conclusion was stored")
    session_id: str | None = Field(default=None, description="Session the conclusion is scoped to, if any")
    observer_id: str = Field(description="The person who holds the conclusion")
    observed_id: str = Field(description="The person the conclusion is about")
    source_ids: list[str] = Field(
        default_factory=list,
        description=(
            "IDs of the conclusions this one was derived from. Empty for explicit conclusions, "
            "which derive from messages rather than from other conclusions."
        ),
    )
    source_url: str | None = Field(default=None, description="Where the conclusion's source message was said")
    said_at: datetime.datetime | None = Field(default=None, description="When the source message was said")


class EvidenceMessageRef(BaseModel):
    """A message the dialectic agent read while answering.

    Identity and provenance only, no content. Fetch the message by `id` when the text is needed.
    """

    id: str = Field(description="Message ID")
    session_id: str = Field(description="Session the message belongs to")
    peer_id: str = Field(description="Person who said the message")
    created_at: datetime.datetime = Field(description="When the message was stored")
    said_at: datetime.datetime | None = Field(default=None, description="When the message was said")
    source_url: str | None = Field(default=None, description="Where the message was said")
    channel: str | None = Field(default=None, description="Channel the message was said in")


class EvidenceFact(BaseModel):
    """A retained fact (from the bank's memory engine) the dialectic agent read while answering."""

    id: str = Field(description="Memory unit ID")
    text: str = Field(description="The fact text")
    fact_type: str = Field(description="world, experience or observation")
    document_id: str | None = Field(default=None, description="Document the fact was retained from")
    source_url: str | None = Field(default=None, description="Where the fact was said, when its metadata names it")
    said_at: str | None = Field(default=None, description="When the fact was said, when known")


class EvidenceToolCall(BaseModel):
    """A tool the dialectic agent invoked while answering."""

    tool_name: str = Field(description="Name of the tool")
    tool_input: dict[str, Any] = Field(default_factory=dict, description="Arguments the agent passed to the tool")


class Source(BaseModel):
    """One thing an answer can cite: a conclusion, a message or a retained fact, with where it was said."""

    kind: str = Field(description="conclusion, message or fact")
    id: str = Field(description="ID of the conclusion, message or fact")
    source_url: str | None = Field(default=None, description="Where it was said")
    said_at: str | None = Field(default=None, description="When it was said (ISO 8601)")
    said_by: str | None = Field(default=None, description="Who said it, when known")
    session: str | None = Field(default=None, description="Session it belongs to, when any")
    text: str | None = Field(default=None, description="The conclusion or fact text (messages carry none)")


class Evidence(BaseModel):
    """What the dialectic agent read and did while answering.

    Collated from the agent's own reads rather than reported by the model, so it is deterministic but
    over-reports: it lists what the agent accessed, which is not necessarily what the answer relied on.
    """

    conclusions: list[EvidenceObservation] = Field(default_factory=list)
    messages: list[EvidenceMessageRef] = Field(default_factory=list)
    facts: list[EvidenceFact] = Field(default_factory=list)
    tool_calls: list[EvidenceToolCall] = Field(default_factory=list)

    def sources(self) -> list[Source]:
        """Every conclusion, message and fact read, as a citable source, oldest first."""
        sources: list[Source] = []
        for conclusion in self.conclusions:
            when = conclusion.said_at or conclusion.created_at
            sources.append(
                Source(
                    kind="conclusion",
                    id=conclusion.id,
                    source_url=conclusion.source_url,
                    said_at=when.isoformat(),
                    said_by=conclusion.observed_id,
                    session=conclusion.session_id,
                    text=conclusion.content,
                )
            )
        for message in self.messages:
            when = message.said_at or message.created_at
            sources.append(
                Source(
                    kind="message",
                    id=message.id,
                    source_url=message.source_url,
                    said_at=when.isoformat(),
                    said_by=message.peer_id,
                    session=message.session_id,
                )
            )
        for fact in self.facts:
            sources.append(
                Source(
                    kind="fact",
                    id=fact.id,
                    source_url=fact.source_url,
                    said_at=fact.said_at,
                    text=fact.text,
                )
            )
        return sorted(sources, key=lambda source: (source.said_at or "", source.kind, source.id))


def _tool_call_from_log_entry(entry: dict[str, Any]) -> EvidenceToolCall:
    """Read one tool-loop log entry (``tool_name`` and ``tool_input``, or ``name`` and ``input``)."""
    name = entry.get("tool_name", entry.get("name", ""))
    raw_input = entry.get("tool_input", entry.get("input"))
    return EvidenceToolCall(
        tool_name=str(name),
        tool_input=cast("dict[str, Any]", raw_input) if isinstance(raw_input, dict) else {},
    )


@dataclass
class EvidenceAccumulator:
    """Collects the rows a dialectic agent reads, for one chat call.

    Threaded down through the agent into the tool context. Tool handlers reach it through the context,
    which they copy with ``dataclasses.replace`` -- a shallow copy, so every copy appends to this same
    instance.

    Conclusions, messages and facts are keyed by ID so that a row two tools both returned is recorded
    once. Keying on ID rather than content matters: distinct conclusions can read the same.
    """

    conclusions: dict[str, EvidenceObservation] = field(default_factory=dict)
    messages: dict[str, EvidenceMessageRef] = field(default_factory=dict)
    facts: dict[str, EvidenceFact] = field(default_factory=dict)
    tool_calls: list[EvidenceToolCall] = field(default_factory=list)

    def add_documents(self, documents: Iterable[Conclusion]) -> None:
        """Record conclusions a read path returned."""
        for document in documents:
            if document.id in self.conclusions:
                continue
            self.conclusions[document.id] = EvidenceObservation(
                id=document.id,
                level=document.level or "explicit",
                content=document.content,
                created_at=document.created_at,
                session_id=document.session_name,
                observer_id=document.observer,
                observed_id=document.observed,
                source_ids=list(document.source_ids or []),
                source_url=document.source_url,
                said_at=document.said_at,
            )

    def add_messages(self, messages: Iterable[MessageRow]) -> None:
        """Record messages a read path returned."""
        for message in messages:
            if message.public_id in self.messages:
                continue
            self.messages[message.public_id] = EvidenceMessageRef(
                id=message.public_id,
                session_id=message.session_name,
                peer_id=message.peer_name,
                created_at=message.created_at,
                said_at=message.said_at,
                source_url=message.source_url,
                channel=message.channel,
            )

    def add_facts(self, facts: Iterable[EvidenceFact]) -> None:
        """Record retained facts a read path returned."""
        for fact in facts:
            self.facts.setdefault(fact.id, fact)

    def record_tool_calls(self, tool_calls_made: Sequence[dict[str, Any]]) -> None:
        """Replace the tool call log with a completed loop's history.

        The tool loop rewrites its log wholesale on every exit path, so this overwrites rather than
        appends. Failed calls never reach the log, so the result records successful invocations only.
        """
        self.tool_calls = [_tool_call_from_log_entry(entry) for entry in tool_calls_made]

    def build(self) -> Evidence:
        """Assemble what was collected into the API shape."""
        return Evidence(
            conclusions=sorted(
                self.conclusions.values(), key=lambda observation: (observation.created_at, observation.id)
            ),
            messages=sorted(self.messages.values(), key=lambda ref: (ref.created_at, ref.id)),
            facts=sorted(self.facts.values(), key=lambda fact: fact.id),
            tool_calls=list(self.tool_calls),
        )
