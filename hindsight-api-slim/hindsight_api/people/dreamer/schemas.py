# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/schemas/api.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""Dream schemas: the Representation History API models (the dream types live in ``people.schemas``)."""

from __future__ import annotations

import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..schemas import DocumentLevel, DreamType

__all__ = ["DocumentLevel", "DreamType"]


class ScheduleDreamRequest(BaseModel):
    observer: str = Field(..., description="Observer peer name")
    observed: str | None = Field(None, description="Observed peer name (defaults to observer if not specified)")
    dream_type: DreamType = Field(..., description="Type of dream to schedule")
    session_id: str | None = Field(None, description="Session ID to scope the dream to if specified")
    rebuild: bool = Field(
        False,
        description=(
            "card_refresh dreams only: rebuild the peer card solely from "
            "observations currently in the collection, without injecting the "
            "existing card (use after removals)"
        ),
    )


class DreamRunList(BaseModel):
    """Filters for listing dream runs."""

    observed_id: str | None = Field(default=None, description="Only runs about this peer")
    observer_id: str | None = Field(
        default=None,
        description="Only runs from this observer's perspective. Combine with `observed_id` for one representation.",
    )


class DreamRun(BaseModel):
    """One dream over a representation: the observer's model of the observed peer."""

    id: str = Field(description="The dream's run ID")
    observer: str = Field(
        description="The peer whose representation the dream changed",
        serialization_alias="observer_id",
    )
    observed: str = Field(
        description="The peer the representation is about",
        serialization_alias="observed_id",
    )
    dream_type: str | None = Field(
        default=None,
        description="The kind of dream, e.g. `omni` (deduction and induction) or a card refresh",
    )
    trigger_reason: str | None = Field(
        default=None,
        description="What scheduled the dream, e.g. `document_threshold` or `manual`",
    )
    status: Literal["running", "completed", "failed", "interrupted"] = Field(
        description=("`failed` and `interrupted` runs can still have changes: each change is committed as it is made.")
    )
    created_count: int = Field(description="Conclusions the dream created")
    removed_count: int = Field(description="Conclusions the dream removed")
    peer_card_updated: bool = Field(description="Whether the dream changed the peer card")
    started_at: datetime.datetime
    finished_at: datetime.datetime | None = None
    pruned_at: datetime.datetime | None = Field(
        default=None,
        description=(
            "When retention first deleted some of this run's changes. Past the"
            " retention window a run keeps only the created conclusions that still"
            " exist; `removed` and `peer_card` are gone. The counts stay as the"
            " dream recorded them."
        ),
    )

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class DreamConclusionRef(BaseModel):
    """A conclusion referenced by a dream, resolved when the run is read."""

    id: str
    content: str | None = Field(default=None, description="Null when the conclusion is `unavailable`")
    level: DocumentLevel | None = None
    status: Literal["current", "removed", "unavailable"] = Field(
        description=(
            "`current`: still in the representation. `removed`: a later dream"
            " removed it; `removed_in_run` names that dream. `unavailable`: it was"
            " deleted (through the API, with its session, or by the deriver), or"
            " the dream that removed it is past the retention window."
        )
    )
    removed_in_run: str | None = None


class DreamCreatedConclusion(BaseModel):
    """A conclusion a dream created, with the conclusions it was derived from."""

    conclusion: DreamConclusionRef
    sources: list[DreamConclusionRef] = Field(
        description="Premises for deductive conclusions, supporting evidence for inductive ones"
    )
    specialist: str | None = Field(default=None, description="The dream specialist that made the change")


class DreamRemovedConclusion(BaseModel):
    """A conclusion a dream removed, as it read when it was removed."""

    id: str
    content: str
    level: DocumentLevel | None = None
    replaced_by: str | None = Field(
        default=None,
        description="Set when a new conclusion from the same dream superseded this one as a near-duplicate",
    )
    specialist: str | None = None


class DreamPeerCardChange(BaseModel):
    """The peer card before the dream's first update and after its last."""

    before: list[str] | None = Field(description="Null when there was no card")
    after: list[str]


class DreamRunDetail(DreamRun):
    """A dream run with every change it committed."""

    created: list[DreamCreatedConclusion]
    removed: list[DreamRemovedConclusion]
    peer_card: DreamPeerCardChange | None = None
