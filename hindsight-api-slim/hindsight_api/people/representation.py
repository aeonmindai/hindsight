# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/crud/representation.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
import datetime
import logging
from typing import Any

from . import schemas
from .conclusions import (
    _CONCLUSION_COLUMNS,
    DEDUPLICATE,
    Conclusion,
    CreateDocumentsResult,
    _conclusions_from_rows,
    create_documents,
    query_documents,
)
from .context import PeopleContext
from .db import connection
from .formatting import format_datetime_utc
from .representation_utils import (
    ALLOWLIST_SAFE_LEVELS,
    DeductiveObservation,
    ExplicitObservation,
    Representation,
    allowlist_safe_levels,
)
from .sanitization import strip_nul
from .schemas import WORKING_REPRESENTATION_MAX_OBSERVATIONS, ResolvedConfiguration

logger = logging.getLogger(__name__)


def _observation_text(obs: ExplicitObservation | DeductiveObservation) -> str:
    """Return the canonical text payload for an explicit or deductive observation."""
    return obs.conclusion if isinstance(obs, DeductiveObservation) else obs.content


def _normalized_observation(
    obs: ExplicitObservation | DeductiveObservation,
) -> ExplicitObservation | DeductiveObservation:
    """Return an observation with its persisted/embed text normalized.

    NUL bytes are removed here rather than closer to the database so that the
    text that gets embedded is the same text that gets stored.
    """
    text = strip_nul(_observation_text(obs)).strip()
    if isinstance(obs, DeductiveObservation):
        return obs.model_copy(
            update={
                "conclusion": text,
                # Premises ride along in internal_metadata, and jsonb rejects
                # NUL in strings just as text columns do.
                "premises": strip_nul(obs.premises),
            }
        )
    return obs.model_copy(update={"content": text})


class RepresentationManager:
    """Unified manager for representation and conclusion queries."""

    def __init__(
        self,
        ctx: PeopleContext,
        bank_id: str,
        *,
        observer: str,
        observed: str,
    ) -> None:
        self.ctx: PeopleContext = ctx
        self.bank_id: str = bank_id
        self.observer: str = observer
        self.observed: str = observed

    async def save_representation(
        self,
        representation: Representation,
        message_ids: list[int],
        session_name: str,
        message_created_at: datetime.datetime,
        message_level_configuration: ResolvedConfiguration | None = None,
        *,
        conn: Any | None = None,
    ) -> CreateDocumentsResult:
        """Save Representation objects as a set of conclusions.

        Args:
            representation: Representation object
            message_ids: Message ID range to link with observations
            session_name: Session name to link with existing summary context
            message_created_at: Timestamp when the message was created
            message_level_configuration: Resolved configuration of the message (dreams are only
                scheduled when it enables them; default: enabled)
            conn: Optional open connection to join the caller's transaction

        Returns:
            The result of conclusion creation, including saved conclusions and deduplication counts.
        """

        empty_result = CreateDocumentsResult()

        if not representation.deductive and not representation.explicit:
            logger.debug("No observations to save")
            return empty_result

        # Normalize before the emptiness check: str.strip() does not remove
        # NUL, so content that normalizes away has to be dropped afterwards.
        all_observations = [
            normalized
            for normalized in (
                _normalized_observation(obs) for obs in representation.deductive + representation.explicit
            )
            if _observation_text(normalized)
        ]
        if not all_observations:
            logger.debug("No non-empty observations to save")
            return empty_result

        # Batch embed all observations
        observation_texts = [_observation_text(obs) for obs in all_observations]
        embeddings = await self.ctx.embed(observation_texts)

        # Batch create conclusion objects
        new_documents_result = await self._save_representation_internal(
            all_observations,
            embeddings,
            message_ids,
            session_name,
            message_created_at,
            conn=conn,
        )

        dream_enabled = message_level_configuration.dream.enabled if message_level_configuration else True
        if dream_enabled and conn is None:
            try:
                # Imported here: the dreamer imports this module.
                from .dreamer.dream_scheduler import check_and_schedule_dream

                await check_and_schedule_dream(self.ctx, self.bank_id, self.observer, self.observed)
            except Exception as e:
                logger.warning(f"Failed to check dream scheduling: {e}")

        return new_documents_result

    async def _save_representation_internal(
        self,
        all_observations: list[ExplicitObservation | DeductiveObservation],
        embeddings: list[list[float]],
        message_ids: list[int],
        session_name: str,
        message_created_at: datetime.datetime,
        *,
        conn: Any | None = None,
    ) -> CreateDocumentsResult:
        # Prepare all conclusions for bulk creation
        documents_to_create: list[schemas.DocumentCreate] = []
        for obs, embedding in zip(all_observations, embeddings, strict=True):
            # NOTE: will add additional levels of reasoning in the future
            source_message_ids: list[str] | None = None
            if isinstance(obs, DeductiveObservation):
                obs_level = "deductive"
                obs_content = obs.conclusion
                obs_premises = obs.premises
            else:
                obs_level = "explicit"
                obs_content = obs.content
                obs_premises = None
                source_message_ids = obs.source_message_ids or None

            metadata: schemas.DocumentMetadata = schemas.DocumentMetadata(
                message_ids=message_ids,
                premises=obs_premises,
                message_created_at=format_datetime_utc(message_created_at),
            )

            documents_to_create.append(
                schemas.DocumentCreate(
                    content=obs_content,
                    session_name=session_name,
                    level=obs_level,
                    metadata=metadata,
                    embedding=embedding,
                    source_message_ids=source_message_ids,
                    # A deductive observation from the deriver carries no premise ids; its source is
                    # the messages the batch was drawn from, resolved by create_documents.
                )
            )

        # Use bulk creation with optional duplicate detection
        return await create_documents(
            self.ctx,
            self.bank_id,
            documents_to_create,
            observer=self.observer,
            observed=self.observed,
            deduplicate=DEDUPLICATE,
            conn=conn,
        )

    async def get_working_representation(
        self,
        *,
        session_allowlist: list[str] | None = None,
        include_semantic_query: str | None = None,
        embedding: list[float] | None = None,
        semantic_search_top_k: int | None = None,
        semantic_search_max_distance: float | None = None,
        include_most_derived: bool = False,
        max_observations: int = WORKING_REPRESENTATION_MAX_OBSERVATIONS,
        conn: Any | None = None,
    ) -> Representation:
        """Get working representation with flexible query options.

        Args:
            session_allowlist: Optional session allowlist to filter by. Applied
                uniformly to every query path (semantic, most-derived, and
                recent). None means no session restriction; an empty list
                fail-closes to an empty representation.
            include_semantic_query: Query for semantic search
            embedding: Pre-computed embedding for the semantic query.
            semantic_search_top_k: Number of semantic results
            semantic_search_max_distance: Maximum distance for semantic search
            include_most_derived: Include most derived (most reinforced) observations
            max_observations: Maximum number of observations to return
            conn: Optional open connection to reuse

        Returns:
            Representation combining various query strategies
        """
        if include_semantic_query and embedding is None:
            # Best-effort precompute when caller didn't supply one (or their precompute was suppressed).
            try:
                embedding = await self.ctx.embed_query(include_semantic_query)
            except Exception:
                logger.warning("Query embedding failed, degrading to non-semantic retrieval", exc_info=True)

        return await self._get_working_representation_internal(
            session_allowlist=session_allowlist,
            include_semantic_query=include_semantic_query,
            embedding=embedding,
            semantic_search_top_k=semantic_search_top_k,
            semantic_search_max_distance=semantic_search_max_distance,
            include_most_derived=include_most_derived,
            max_observations=max_observations,
            conn=conn,
        )

    # Private helper methods

    async def _get_working_representation_internal(
        self,
        *,
        session_allowlist: list[str] | None = None,
        include_semantic_query: str | None = None,
        embedding: list[float] | None = None,
        semantic_search_top_k: int | None = None,
        semantic_search_max_distance: float | None = None,
        include_most_derived: bool = False,
        max_observations: int = WORKING_REPRESENTATION_MAX_OBSERVATIONS,
        conn: Any | None = None,
    ) -> Representation:
        """Internal implementation of get_working_representation."""
        # Fail closed on an empty allowlist. This must short-circuit before
        # any query: an `IN ()` clause with an empty list could silently widen
        # the scope instead.
        if session_allowlist is not None and not session_allowlist:
            return Representation()

        total = max_observations

        # Calculate how many observations to get from each source.
        # Floor of 1 when a semantic query was explicitly requested.
        semantic_observations = (
            min(
                max(
                    1,
                    semantic_search_top_k if semantic_search_top_k is not None else total // 3,
                ),
                total,
            )
            if include_semantic_query and embedding is not None
            else 0
        )

        if include_semantic_query and include_most_derived:
            # three-way blend: both semantic and derived requested
            top_observations = min(max(0, total // 3), total - semantic_observations)
        elif include_most_derived:
            # two-way blend: only derived requested
            top_observations = min(max(0, total // 2), total - semantic_observations)
        else:
            # no derived observations requested
            top_observations = 0

        representation = Representation()
        selected_document_ids: set[str] = set()

        # Get semantic observations if requested
        if include_semantic_query and embedding is not None:
            semantic_docs = await self.query_documents(
                query=include_semantic_query,
                top_k=semantic_observations,
                max_distance=semantic_search_max_distance,
                embedding=embedding,
                session_allowlist=session_allowlist,
                conn=conn,
            )
            representation.merge_representation(Representation.from_documents(semantic_docs))
            selected_document_ids.update(document.id for document in semantic_docs)

        # Get most derived observations if requested. The semantic query may
        # return fewer documents than requested, so cap this query by the
        # actual remaining representation capacity rather than the configured
        # budget alone.
        if include_most_derived:
            remaining_observations = max(0, total - representation.len())
            derived_docs = await self._query_documents_most_derived(
                top_k=min(top_observations, remaining_observations),
                session_allowlist=session_allowlist,
                conn=conn,
            )
            representation.merge_representation(Representation.from_documents(derived_docs))
            selected_document_ids.update(document.id for document in derived_docs)

        # Reclaim any capacity left by queries that returned fewer unique
        # documents than requested. This keeps the final representation from
        # shrinking when semantic or most-derived search underfills its slice.
        recent_observations = max(0, total - representation.len())
        recent_docs = await self._query_documents_recent(
            top_k=recent_observations,
            session_allowlist=session_allowlist,
            excluded_document_ids=selected_document_ids,
            conn=conn,
        )

        representation.merge_representation(Representation.from_documents(recent_docs))

        return representation

    async def query_documents(
        self,
        query: str,
        top_k: int,
        max_distance: float | None = None,
        level: str | None = None,
        embedding: list[float] | None = None,
        session_allowlist: list[str] | None = None,
        *,
        conn: Any | None = None,
    ) -> list[Conclusion]:
        """Query conclusions by semantic similarity (Honcho's ``_query_documents_semantic``).

        Failures degrade to no semantic observations so the representation can still be built from the
        other sources.
        """
        try:
            if level:
                return await self._query_documents_for_level(
                    query,
                    level,
                    top_k,
                    max_distance,
                    embedding=embedding,
                    session_allowlist=session_allowlist,
                )
            return await query_documents(
                self.ctx,
                self.bank_id,
                query,
                observer=self.observer,
                observed=self.observed,
                max_distance=max_distance,
                top_k=top_k,
                embedding=embedding,
                filters=self._build_filter_conditions(session_allowlist=session_allowlist) or None,
            )
        except Exception:
            logger.exception("Error getting relevant observations")
            return []

    async def _query_documents_recent(
        self,
        top_k: int,
        session_allowlist: list[str] | None = None,
        excluded_document_ids: set[str] | None = None,
        *,
        conn: Any | None = None,
    ) -> list[Conclusion]:
        """Query most recent conclusions."""
        if top_k <= 0:
            return []
        params: list[Any] = [self.bank_id, self.observer, self.observed]
        extra = ""
        if excluded_document_ids:
            params.append(sorted(excluded_document_ids))
            extra += f" AND id <> ALL(${len(params)}::text[])"
        if session_allowlist is not None:
            params.append(session_allowlist)
            extra += f" AND session_name = ANY(${len(params)}::text[])"
            # Only levels with a trustworthy session stamp are scopeable -- see ALLOWLIST_SAFE_LEVELS.
            params.append(list(ALLOWLIST_SAFE_LEVELS))
            extra += f" AND level = ANY(${len(params)}::text[])"
        params.append(top_k)
        async with connection(self.ctx, conn) as db:
            rows = await db.fetch(
                f"SELECT {_CONCLUSION_COLUMNS} FROM {self.ctx.table('conclusions')} "
                f"WHERE bank_id = $1 AND observer = $2 AND observed = $3 AND deleted_at IS NULL{extra} "
                f"ORDER BY created_at DESC, id LIMIT ${len(params)}",
                *params,
            )
            return await _conclusions_from_rows(self.ctx, db, rows)

    async def _query_documents_most_derived(
        self, top_k: int, session_allowlist: list[str] | None = None, *, conn: Any | None = None
    ) -> list[Conclusion]:
        """Query most derived conclusions."""
        if top_k <= 0:
            return []
        params: list[Any] = [self.bank_id, self.observer, self.observed]
        extra = ""
        if session_allowlist is not None:
            params.append(session_allowlist)
            extra += f" AND session_name = ANY(${len(params)}::text[])"
            # Only levels with a trustworthy session stamp are scopeable -- see ALLOWLIST_SAFE_LEVELS.
            params.append(list(ALLOWLIST_SAFE_LEVELS))
            extra += f" AND level = ANY(${len(params)}::text[])"
        params.append(top_k)
        async with connection(self.ctx, conn) as db:
            rows = await db.fetch(
                f"SELECT {_CONCLUSION_COLUMNS} FROM {self.ctx.table('conclusions')} "
                f"WHERE bank_id = $1 AND observer = $2 AND observed = $3 AND deleted_at IS NULL{extra} "
                # created_at is the transaction timestamp, so conclusions created in the same batch share
                # it -- id keeps the order deterministic.
                f"ORDER BY times_derived DESC, created_at DESC, id LIMIT ${len(params)}",
                *params,
            )
            return await _conclusions_from_rows(self.ctx, db, rows)

    async def _get_observations_internal(
        self,
        query: str,
        top_k: int,
        max_distance: float,
        level: str | None,
    ) -> list[Conclusion]:
        """Internal method that does the actual observation retrieval."""
        return await self.query_documents(query, top_k, max_distance, level)

    async def _query_documents_for_level(
        self,
        query: str,
        level: str,
        count: int,
        max_distance: float | None = None,
        embedding: list[float] | None = None,
        session_allowlist: list[str] | None = None,
    ) -> list[Conclusion]:
        """Query conclusions for a specific level."""
        documents = await query_documents(
            self.ctx,
            self.bank_id,
            query,
            observer=self.observer,
            observed=self.observed,
            max_distance=max_distance,
            top_k=count,
            filters=self._build_filter_conditions(level, session_allowlist=session_allowlist),
            embedding=embedding,
        )

        # Sort by creation time
        docs_sorted: list[Conclusion] = sorted(list(documents), key=lambda x: x.created_at, reverse=True)
        return docs_sorted

    def _build_filter_conditions(
        self,
        level: str | None = None,
        session_allowlist: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build filter conditions for conclusion queries.

        Returns a flat dict of key-value pairs. Callers must not pass an empty session_allowlist list --
        empty allowlists fail closed before any query is issued (see _get_working_representation_internal).
        """
        filters: dict[str, Any] = {}

        if level:
            filters["level"] = level

        # `is not None` (not truthiness): an explicit empty allowlist must emit
        # an empty `in` so the query fails closed. Truthiness here would
        # silently drop the filter and widen scope.
        if session_allowlist is not None:
            filters["session_name"] = {"in": session_allowlist}
            # Only levels with a trustworthy session stamp are scopeable. This
            # overrides any narrower `level` above; an empty intersection emits
            # `{"in": []}`, which matches nothing rather than everything.
            filters["level"] = {"in": allowlist_safe_levels([level] if level else None)}

        return filters


# Module-level functions for backward compatibility and convenience


async def get_working_representation(
    ctx: PeopleContext,
    bank_id: str,
    *,
    observer: str,
    observed: str,
    session_allowlist: list[str] | None = None,
    include_semantic_query: str | None = None,
    embedding: list[float] | None = None,
    semantic_search_top_k: int | None = None,
    semantic_search_max_distance: float | None = None,
    include_most_derived: bool = False,
    max_observations: int = WORKING_REPRESENTATION_MAX_OBSERVATIONS,
    conn: Any | None = None,
) -> Representation:
    """Get raw working representation data from the relevant conclusions.

    This is a convenience function that creates a RepresentationManager and calls
    get_working_representation on it.
    """
    manager = RepresentationManager(ctx, bank_id, observer=observer, observed=observed)
    return await manager.get_working_representation(
        session_allowlist=session_allowlist,
        include_semantic_query=include_semantic_query,
        embedding=embedding,
        semantic_search_top_k=semantic_search_top_k,
        semantic_search_max_distance=semantic_search_max_distance,
        include_most_derived=include_most_derived,
        max_observations=max_observations,
        conn=conn,
    )
