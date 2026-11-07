# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/dreamer/orchestrator.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""
Dream orchestrator for the specialist-based architecture.

This module coordinates the full dream cycle:
0. [Optional] Surprisal sampling: Pre-filter observations by geometric surprisal
1. Run deduction specialist (self-directed exploration, creates deductive observations)
2. Run induction specialist (self-directed exploration, creates inductive observations)

Specialists are self-directed agents that explore the observation space and create
higher-level observations. When surprisal sampling finds interesting observations,
they're passed as hints, but specialists are free to follow the evidence wherever it leads.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ..context import PeopleContext
from ..dream_history import close_run, open_run
from ..ids import generate_nanoid
from ..jsonb import from_jsonb, to_jsonb
from ..schemas import DreamType, get_configuration
from . import collection_state
from .dream_scheduler import (
    WORK_LIST_BACKLOG_KEY,
    schedule_backlog_dream,
)
from .payload import DreamPayload
from .settings import DREAM
from .specialists import (
    SPECIALISTS,
    CardRefreshSpecialist,
    SpecialistResult,
)
from .surprisal import SurprisalScore
from .working_memory import (
    NOTES_KEY,
    DreamState,
    advance_cursor,
    clear_position,
    load_dream_state,
    plan_cursor_move,
)

logger = logging.getLogger(__name__)


@dataclass
class DreamResult:
    """Result of a dream cycle for telemetry reporting."""

    # Run identification
    run_id: str
    specialists_run: list[str]

    # Specialist outcomes
    deduction_success: bool
    induction_success: bool

    # Surprisal sampling
    surprisal_enabled: bool
    surprisal_conclusion_count: int

    # Aggregate metrics
    total_iterations: int
    total_duration_ms: float
    input_tokens: int
    output_tokens: int

    # Working state shown to the specialists, when the brief or notes are on
    state: DreamState | None = None


def working_memory_enabled() -> bool:
    """Whether dreams keep the cursor: needed by the work list and by notes."""
    return DREAM.BRIEF_ENABLED or DREAM.NOTES_MODE != "off"


async def run_dream(
    ctx: PeopleContext,
    bank_id: str,
    observer: str,
    observed: str,
    session_name: str | None = None,
    *,
    dream_type: str | None = None,
    trigger_reason: str | None = None,
    delay_reason: str | None = None,
    documents_since_last_dream_at_schedule: int | None = None,
    document_threshold: int | None = None,
) -> DreamResult | None:
    """
    Run a full dream cycle with optional surprisal-based sampling.

    The dream cycle runs specialists sequentially:
    0. [Optional] Surprisal sampling: Pre-filter observations by geometric surprisal
    1. Deduction specialist: Creates deductive observations from explicit facts
    2. Induction specialist: Creates inductive observations from patterns

    Uses short-lived DB sessions to avoid holding connections during LLM calls.

    Args:
        ctx: People context
        bank_id: Bank identifier
        observer: Observer peer name
        observed: Observed peer name
        session_name: Session identifier if specified
    """
    if not DREAM.ENABLED:
        return None

    run_id = generate_nanoid()
    start_time = time.perf_counter()

    logger.info(f"[{run_id}] Starting dream cycle for {bank_id}/{observer}/{observed}")

    # Session configuration (the bank has no dream configuration of its own).
    configuration = get_configuration(
        None,
        await collection_state.get_session_configuration(ctx, bank_id, session_name)
        if session_name is not None
        else None,
        None,
    )
    if not configuration.dream.enabled:
        logger.info(f"[{run_id}] Dreams disabled for {bank_id}/{session_name}, skipping dream")
        return None

    # Track specialist outcomes
    deduction_success = False
    induction_success = False
    surprisal_observation_count = 0
    deduction_result: SpecialistResult | None = None
    induction_result: SpecialistResult | None = None

    # Surprisal-based sampling (if enabled)
    # Specialists are self-directed by default - hints are optional suggestions
    exploration_hints: list[str] | None = None

    if DREAM.SURPRISAL.ENABLED:
        logger.info(f"[{run_id}] Computing surprisal scores")
        try:
            from .surprisal import sample_observations_with_surprisal

            high_surprisal_obs = await sample_observations_with_surprisal(
                ctx,
                bank_id=bank_id,
                observer=observer,
                observed=observed,
            )

            logger.info(f"[{run_id}] Surprisal: Found {len(high_surprisal_obs)} high-surprisal observations")
            surprisal_observation_count = len(high_surprisal_obs)

            if len(high_surprisal_obs) > 0:
                # Use high-surprisal observations as hints for exploration
                exploration_hints = _create_queries_from_surprisal(high_surprisal_obs)
                logger.info(
                    f"[{run_id}] ✨ SURPRISAL HINTS: Suggesting {len(exploration_hints)} "
                    + "high-surprisal topics for specialists to investigate"
                )
                logger.info(
                    f"[{run_id}] Targeting observations with surprisal range: "
                    + f"{high_surprisal_obs[-1].surprisal:.3f} to {high_surprisal_obs[0].surprisal:.3f}"
                )
            else:
                logger.info(f"[{run_id}] No high-surprisal observations - specialists will explore freely")

        except Exception as e:
            logger.error(f"[{run_id}] Surprisal sampling failed: {e}", exc_info=True)
            # Specialists will explore freely without hints

    # Loaded once for both specialists, so deduction's writes don't change the
    # work list induction is shown and the cursor advances once per dream.
    state: DreamState | None = None
    if working_memory_enabled():
        try:
            state = await load_dream_state(
                ctx,
                bank_id,
                observer=observer,
                observed=observed,
                include_work_list=DREAM.BRIEF_ENABLED,
            )
        except Exception as e:
            logger.error(f"[{run_id}] Loading dream working state failed: {e}", exc_info=True)

    # Specialist phase wrapped in try/finally so DreamRunEvent ALWAYS emits —
    # both for graceful failures (specialist raises Exception) AND unexpected
    # exceptions including CancelledError. The orphaned-child-event problem
    # before this fix: specialists.py:350 catches BaseException and re-raises,
    # but the orchestrator's old `except SpecialistExecutionError` clauses
    # didn't match anything actually raised in src/, so any real failure
    # propagated past the emit at the bottom of this function. Now: the
    # specialist try blocks catch Exception (CancelledError still propagates
    # correctly), and the outer try/finally guarantees the parent event fires.
    #
    # Pre-init aggregate locals so the function-level return below and the
    # finally's emit both see defined values even on an early exception.
    duration_ms = 0.0
    total_iterations = 0
    total_input_tokens = 0
    total_output_tokens = 0
    history = await open_run(
        ctx,
        run_id=run_id,
        bank_id=bank_id,
        observer=observer,
        observed=observed,
        dream_type=dream_type,
        trigger_reason=trigger_reason,
    )
    try:
        # Run deduction specialist (manages its own DB sessions)
        logger.info(f"[{run_id}] Running deduction specialist")
        deduction_specialist = SPECIALISTS["deduction"]
        try:
            deduction_result = await deduction_specialist.run(
                ctx,
                bank_id=bank_id,
                observer=observer,
                observed=observed,
                session_name=session_name,
                hints=exploration_hints,
                configuration=configuration,
                parent_run_id=run_id,
                state=state,
                history=history,
            )
            logger.info(f"[{run_id}] Deduction completed: {deduction_result.content[:200]}...")
            deduction_success = deduction_result.success
        except Exception as e:
            # `Exception` (not `BaseException`) — CancelledError must still
            # propagate so the worker can shut down. SpecialistExecutionError
            # is no longer raised by `src/`, but the catch is broad enough to
            # cover provider/DB/tool/validation errors.
            logger.error(f"[{run_id}] Deduction specialist failed: {e}", exc_info=True)

        # Run induction specialist (after deduction so it can see new deductive obs)
        logger.info(f"[{run_id}] Running induction specialist")
        induction_specialist = SPECIALISTS["induction"]
        try:
            induction_result = await induction_specialist.run(
                ctx,
                bank_id=bank_id,
                observer=observer,
                observed=observed,
                session_name=session_name,
                hints=exploration_hints,
                configuration=configuration,
                parent_run_id=run_id,
                state=state,
                history=history,
            )
            logger.info(f"[{run_id}] Induction completed: {induction_result.content[:200]}...")
            induction_success = induction_result.success
        except Exception as e:
            logger.error(f"[{run_id}] Induction specialist failed: {e}", exc_info=True)

        # Log final metrics
        duration_ms = (time.perf_counter() - start_time) * 1000

        logger.info(f"[{run_id}] Dream cycle completed in {duration_ms:.0f}ms")
    finally:
        # Emit DreamRunEvent unconditionally so analytics see a parent for
        # every DreamSpecialistEvent. Aggregation guards None specialist
        # results. Emit-side errors are swallowed by the global telemetry
        # path; a defensive try around event construction protects against
        # schema-validation surprises during partial state.
        if duration_ms == 0.0:
            duration_ms = (time.perf_counter() - start_time) * 1000
        if history is not None:
            await close_run(ctx, bank_id, history.run_id, succeeded=deduction_success and induction_success)
        total_iterations = (deduction_result.iterations if deduction_result else 0) + (
            induction_result.iterations if induction_result else 0
        )
        total_input_tokens = (deduction_result.input_tokens if deduction_result else 0) + (
            induction_result.input_tokens if induction_result else 0
        )
        total_output_tokens = (deduction_result.output_tokens if deduction_result else 0) + (
            induction_result.output_tokens if induction_result else 0
        )

    return DreamResult(
        run_id=run_id,
        specialists_run=["deduction", "induction"],
        deduction_success=deduction_success,
        induction_success=induction_success,
        surprisal_enabled=DREAM.SURPRISAL.ENABLED,
        surprisal_conclusion_count=surprisal_observation_count,
        total_iterations=total_iterations,
        total_duration_ms=duration_ms,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        state=state,
    )


async def run_card_refresh_dream(
    ctx: PeopleContext,
    bank_id: str,
    observer: str,
    observed: str,
    session_name: str | None = None,
    *,
    rebuild: bool = False,
    dream_type: str | None = None,
    trigger_reason: str | None = None,
    delay_reason: str | None = None,
) -> DreamResult | None:
    """
    Run a lightweight card-only refresh dream.

    Runs a single CardRefreshSpecialist restricted to peer-card tools
    (get_recent_observations, search_memory, update_peer_card) with a low
    tool-iteration cap. It never creates or deletes observations.

    Args:
        ctx: People context
        bank_id: Bank identifier
        observer: Observer peer name
        observed: Observed peer name
        session_name: Session identifier if specified
        rebuild: When True the existing card is NOT injected into the prompt
            and the specialist rebuilds it solely from observations present in
            the collection (used after removals).
    """
    if not DREAM.ENABLED:
        return None

    run_id = generate_nanoid()
    start_time = time.perf_counter()

    logger.info(f"[{run_id}] Starting card-refresh dream for {bank_id}/{observer}/{observed} (rebuild={rebuild})")

    # Session configuration (the bank has no dream configuration of its own).
    configuration = get_configuration(
        None,
        await collection_state.get_session_configuration(ctx, bank_id, session_name)
        if session_name is not None
        else None,
        None,
    )
    if not configuration.dream.enabled:
        logger.info(f"[{run_id}] Dreams disabled for {bank_id}/{session_name}, skipping card refresh")
        return None
    if not configuration.peer_card.create:
        logger.info(f"[{run_id}] Peer card creation disabled for {bank_id}, skipping card refresh")
        return None

    specialist_success = False
    specialist_result: SpecialistResult | None = None
    duration_ms = 0.0
    history = await open_run(
        ctx,
        run_id=run_id,
        bank_id=bank_id,
        observer=observer,
        observed=observed,
        dream_type=dream_type,
        trigger_reason=trigger_reason,
    )
    try:
        specialist = CardRefreshSpecialist(rebuild=rebuild)
        try:
            specialist_result = await specialist.run(
                ctx,
                bank_id=bank_id,
                observer=observer,
                observed=observed,
                session_name=session_name,
                configuration=configuration,
                parent_run_id=run_id,
                history=history,
            )
            logger.info(f"[{run_id}] Card refresh completed: {specialist_result.content[:200]}...")
            specialist_success = specialist_result.success
        except Exception as e:
            # Exception (not BaseException) — CancelledError must propagate so
            # the worker can shut down; the finally still emits the run event.
            logger.error(f"[{run_id}] Card refresh specialist failed: {e}", exc_info=True)

        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"[{run_id}] Card-refresh dream completed in {duration_ms:.0f}ms")
    finally:
        # Emit DreamRunEvent unconditionally so analytics see a parent for the
        # specialist event, mirroring run_dream. Card refresh is a
        # deduction-family run, so its outcome rides on deduction_success.
        if duration_ms == 0.0:
            duration_ms = (time.perf_counter() - start_time) * 1000
        if history is not None:
            await close_run(ctx, bank_id, history.run_id, succeeded=specialist_success)

    return DreamResult(
        run_id=run_id,
        specialists_run=["card_refresh"],
        deduction_success=specialist_success,
        induction_success=False,
        surprisal_enabled=False,
        surprisal_conclusion_count=0,
        total_iterations=specialist_result.iterations if specialist_result else 0,
        total_duration_ms=duration_ms,
        input_tokens=specialist_result.input_tokens if specialist_result else 0,
        output_tokens=specialist_result.output_tokens if specialist_result else 0,
    )


def _create_queries_from_surprisal(
    high_surprisal_obs: list[SurprisalScore],
) -> list[str]:
    """
    Create search queries from high-surprisal observations.

    Strategy: Use observation content as semantic search queries.
    Truncate if too long (>200 chars).

    Args:
        high_surprisal_obs: List of SurprisalScore objects

    Returns:
        List of query strings (max 10)
    """
    queries: list[Any] = []
    for score in high_surprisal_obs:
        content = score.observation.content
        if len(content) > 200:
            content = content[:200] + "..."
        queries.append(content)
    return queries[:10]  # Limit to 10 queries


async def process_dream(
    ctx: PeopleContext,
    payload: DreamPayload,
    bank_id: str,
    *,
    queue_item_id: int | None = None,
) -> None:
    """
    Process a dream task by performing collection maintenance operations.

    Args:
        payload: The dream task payload containing workspace, peer, and dream type information
    """
    logger.info(
        f"""
(っ- ‸ - ς)ᶻ z 𐰁 ᶻ z 𐰁 ᶻ z 𐰁\n
DREAM: {payload.dream_type} documents for {bank_id}/{payload.observer}/{payload.observed}\n
𐰁 z ᶻ 𐰁 z ᶻ 𐰁 z ᶻ(っ- ‸ - ς)"""
    )

    try:
        match payload.dream_type:
            case DreamType.OMNI:
                result = await run_dream(
                    ctx,
                    bank_id=bank_id,
                    observer=payload.observer,
                    observed=payload.observed,
                    session_name=payload.session_name,
                    # scheduling context — propagated to DreamRunEvent.
                    dream_type=payload.dream_type.value,
                    trigger_reason=payload.trigger_reason,
                    delay_reason=payload.delay_reason,
                    documents_since_last_dream_at_schedule=payload.documents_since_last_dream_at_schedule,
                    document_threshold=payload.document_threshold,
                )

                # Log completion (telemetry event already emitted in run_dream)
                if result is not None:
                    logger.info(
                        f"Dream completed: run_id={result.run_id}, "
                        + f"iterations={result.total_iterations}, "
                        + f"duration={result.total_duration_ms:.0f}ms"
                    )

                    any_succeeded = result.deduction_success or result.induction_success
                    all_succeeded = result.deduction_success and result.induction_success
                    # Work-list conclusions this dream left for the next one.
                    # They are kept out of the recorded count, so the scheduler
                    # still sees them as new, and a follow-up dream is
                    # scheduled for them below.
                    backlog = 0
                    if working_memory_enabled() and result.state is not None:
                        _, backlog = plan_cursor_move(
                            result.state,
                            any_specialist_succeeded=any_succeeded,
                            all_specialists_succeeded=all_succeeded,
                        )
                    # When nothing ran, the next dream waits for new
                    # conclusions rather than retrying on a timer.
                    follow_up = backlog if any_succeeded else 0

                    # Both guard fields advance together only on successful consolidation.
                    now_iso = datetime.now(UTC).isoformat()
                    async with ctx.conn() as conn, conn.transaction():
                        # The pair's state row is locked for the read-modify-write below.
                        await conn.execute(
                            f"INSERT INTO {ctx.table('dream_state')} (bank_id, observer, observed) "
                            "VALUES ($1, $2, $3) ON CONFLICT DO NOTHING",
                            bank_id,
                            payload.observer,
                            payload.observed,
                        )
                        internal_metadata = from_jsonb(
                            await conn.fetchval(
                                f"SELECT internal_metadata FROM {ctx.table('dream_state')} "
                                "WHERE bank_id = $1 AND observer = $2 AND observed = $3 FOR UPDATE",
                                bank_id,
                                payload.observer,
                                payload.observed,
                            ),
                            {},
                        )
                        current_explicit_count, _ = await collection_state.explicit_stats(
                            ctx, bank_id, payload.observer, payload.observed
                        )
                        dream_meta = dict(internal_metadata.get("dream", {}))
                        dream_meta["last_dream_at"] = now_iso
                        dream_meta["last_dream_document_count"] = max(0, current_explicit_count - backlog)
                        dream_meta[WORK_LIST_BACKLOG_KEY] = follow_up
                        update_data: dict[str, Any] = {"dream": dream_meta}
                        if not working_memory_enabled():
                            # This dream won't move the cursor, so a stored
                            # position would be stale when it is turned back on.
                            cleared = clear_position(internal_metadata.get(NOTES_KEY))
                            if cleared is not None:
                                update_data[NOTES_KEY] = cleared
                        await conn.execute(
                            f"UPDATE {ctx.table('dream_state')} SET internal_metadata = internal_metadata || $4::jsonb "
                            "WHERE bank_id = $1 AND observer = $2 AND observed = $3",
                            bank_id,
                            payload.observer,
                            payload.observed,
                            to_jsonb(update_data),
                        )

                    # Skipped only when nothing ran and nothing was shown.
                    if working_memory_enabled() and (any_succeeded or result.state is not None):
                        await advance_cursor(
                            ctx,
                            bank_id,
                            observer=payload.observer,
                            observed=payload.observed,
                            state=result.state,
                            run_id=result.run_id,
                            any_specialist_succeeded=any_succeeded,
                            all_specialists_succeeded=all_succeeded,
                        )
                    if follow_up:
                        await schedule_backlog_dream(
                            ctx,
                            bank_id,
                            observer=payload.observer,
                            observed=payload.observed,
                        )

            case DreamType.CARD_REFRESH:
                # Card-only refresh: never touches observations and never
                # advances the omni dream guard pair (last_dream_at /
                # last_dream_document_count) — a card refresh must not delay
                # or satisfy consolidation scheduling.
                result = await run_card_refresh_dream(
                    ctx,
                    bank_id=bank_id,
                    observer=payload.observer,
                    observed=payload.observed,
                    session_name=payload.session_name,
                    rebuild=payload.rebuild,
                    dream_type=payload.dream_type.value,
                    trigger_reason=payload.trigger_reason,
                    delay_reason=payload.delay_reason,
                )
                if result is not None:
                    logger.info(
                        f"Card-refresh dream completed: run_id={result.run_id}, "
                        + f"iterations={result.total_iterations}, "
                        + f"duration={result.total_duration_ms:.0f}ms"
                    )

    except Exception as e:
        logger.error(
            f"Error processing dream task {payload.dream_type} for {payload.observer}/{payload.observed}: {str(e)}",
            exc_info=True,
        )
        # Don't re-raise - we want to mark the dream task as processed even if it fails
