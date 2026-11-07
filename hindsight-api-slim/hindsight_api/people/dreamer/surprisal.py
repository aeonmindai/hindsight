# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/dreamer/surprisal.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""
Surprisal-based observation sampling for dream processing.

Computes geometric surprisal scores for observations using tree-based
data structures, enabling targeted deductive reasoning on anomalous
or novel observations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..context import PeopleContext
from .settings import DREAM
from .trees import SurprisalTree, create_tree

logger = logging.getLogger(__name__)


@dataclass
class ObservationData:
    """Plain data extracted from a conclusion row."""

    id: str
    content: str
    level: str | None
    embedding: np.ndarray | list[float] | None


@dataclass
class SurprisalScore:
    """Container for observation with surprisal score."""

    observation: ObservationData
    surprisal: float
    embedding: np.ndarray


async def sample_observations_with_surprisal(
    ctx: PeopleContext,
    bank_id: str,
    observer: str,
    observed: str,
) -> list[SurprisalScore]:
    """
    Sample observations and compute surprisal scores.

    Workflow:
    1. Fetch observations based on SAMPLING_STRATEGY
    2. Extract embeddings (already stored on objects, no DB needed)
    3. Build tree structure using trees.create_tree()
    4. Compute surprisal for each observation
    5. Rank by surprisal (highest first)
    6. Filter by threshold and take top N

    Args:
        bank_id: Bank identifier
        observer: Observer peer name
        observed: Observed peer name

    Returns:
        List of SurprisalScore objects, ranked by surprisal (highest first)
    """
    try:
        # 1. Fetch observations (short DB scope, closed before compute)
        raw_observations = await _fetch_observations(
            ctx,
            bank_id=bank_id,
            observer=observer,
            observed=observed,
        )
        # Skip any with null embeddings (can't compute surprisal).
        observations = [obs for obs in raw_observations if obs.embedding is not None]
        skipped = len(raw_observations) - len(observations)
        if skipped:
            logger.warning(f"Skipped {skipped} observations with null embeddings")

        # Edge case: No observations
        if not observations:
            logger.warning(f"No observations found for {bank_id}/{observer}/{observed}")
            return []

        # Edge case: Too few observations for tree
        min_observations = DREAM.SURPRISAL.TREE_K * 2
        if len(observations) < min_observations:
            logger.warning(
                f"Too few observations ({len(observations)} < {min_observations}), skipping surprisal computation"
            )
            return []

        # 2. Extract embeddings
        embeddings = _extract_embeddings(observations)
        if embeddings.size == 0:
            logger.error("Failed to extract embeddings")
            return []

        # 3. Build tree
        tree = _build_tree(embeddings)

        # 4. Compute surprisal
        scores = _compute_surprisal_scores(observations, embeddings, tree)

        # Edge case: Invalid surprisal values
        valid_scores = [s for s in scores if not np.isinf(s.surprisal) and not np.isnan(s.surprisal)]
        if len(valid_scores) < len(scores):
            logger.warning(f"Filtered {len(scores) - len(valid_scores)} invalid surprisal scores")

        # 5. Normalize surprisal scores to [0, 1] range
        normalized_scores = _normalize_scores(valid_scores)

        # 6. Rank by normalized surprisal
        normalized_scores.sort(key=lambda x: x.surprisal, reverse=True)

        # Log top 5 scores BEFORE filtering
        top_n = min(5, len(normalized_scores))
        percent = DREAM.SURPRISAL.TOP_PERCENT_SURPRISAL * 100
        logger.info(f"🎯 Surprisal computation complete. Taking top {percent:.0f}%")
        logger.info(f"Top {top_n} observations by normalized surprisal score:")
        for i, score in enumerate(normalized_scores[:top_n], 1):
            content = score.observation.content
            if len(content) > 80:
                content = content[:77] + "..."
            logger.info(f"  #{i} [surprisal={score.surprisal:.3f}] [level={score.observation.level}] {content}")

        filtered = _filter_by_percent(normalized_scores)

        logger.info(f"Selected: {len(filtered)}/{len(observations)} observations (top {percent:.0f}%)")

        # Log summary statistics for filtered results
        if filtered:
            logger.info(
                "📊 Filtered statistics: "
                + f"min={filtered[-1].surprisal:.3f}, "
                + f"max={filtered[0].surprisal:.3f}, "
                + f"mean={sum(s.surprisal for s in filtered) / len(filtered):.3f}"
            )
        else:
            logger.info("No observations exceeded the surprisal threshold")

        return filtered

    except Exception as e:
        logger.error(f"Surprisal sampling failed: {e}", exc_info=True)
        # Return empty to allow dream to continue
        return []


async def _fetch_observations(
    ctx: PeopleContext,
    bank_id: str,
    observer: str,
    observed: str,
) -> list[ObservationData]:
    """
    Fetch observations based on configured sampling strategy.

    Args:
        ctx: People context
        bank_id: Bank identifier
        observer: Observer peer name
        observed: Observed peer name

    Returns:
        List of ObservationData (with embeddings)
    """
    strategy = DREAM.SURPRISAL.SAMPLING_STRATEGY
    sample_size = DREAM.SURPRISAL.SAMPLE_SIZE
    levels = DREAM.SURPRISAL.INCLUDE_LEVELS

    if strategy not in ("recent", "random", "all"):
        logger.warning(f"Unknown sampling strategy: {strategy}, using 'recent'")
        strategy = "recent"
    # "recent" and "all" both order by created_at DESC up to the limit; "random" samples at random.
    order = "random()" if strategy == "random" else "created_at DESC"
    return await _fetch_ordered_observations(
        ctx,
        bank_id=bank_id,
        observer=observer,
        observed=observed,
        limit=sample_size,
        levels=levels,
        order=order,
    )


async def _fetch_ordered_observations(
    ctx: PeopleContext,
    bank_id: str,
    observer: str,
    observed: str,
    limit: int,
    levels: list[str],
    order: str,
) -> list[ObservationData]:
    """
    Fetch up to ``limit`` live observations with their embeddings, filtered by level.

    The limit is applied after the level filter. ``order`` is one of two fixed SQL fragments chosen by
    the caller (never user input).
    """
    args: list[object] = [bank_id, observer, observed]
    level_clause = ""
    if levels:
        args.append(levels)
        level_clause = f" AND level = ANY(${len(args)}::text[])"
    args.append(limit)
    async with ctx.conn() as conn:
        rows = await conn.fetch(
            f"SELECT id, content, level, embedding::real[] AS embedding FROM {ctx.table('conclusions')} "
            f"WHERE bank_id = $1 AND observer = $2 AND observed = $3 AND deleted_at IS NULL{level_clause} "
            f"ORDER BY {order} LIMIT ${len(args)}",
            *args,
        )
    return [
        ObservationData(id=row["id"], content=row["content"], level=row["level"], embedding=row["embedding"])
        for row in rows
    ]


def _extract_embeddings(observations: list[ObservationData]) -> np.ndarray:
    """
    Extract embeddings from observations as numpy array.

    Args:
        observations: List of Document objects with embeddings

    Returns:
        np.ndarray of shape (N, 1536) containing embeddings
    """
    if not observations:
        return np.array([])

    embeddings_list = [obs.embedding for obs in observations]
    embeddings_array = np.array(embeddings_list, dtype=np.float32)

    return embeddings_array


def _build_tree(embeddings: np.ndarray) -> SurprisalTree:
    """
    Build tree structure from embeddings.

    Args:
        embeddings: np.ndarray of shape (N, embedding_dim)

    Returns:
        SurprisalTree configured per settings
    """
    if embeddings.size == 0:
        # Return empty tree (will handle gracefully in caller)
        return create_tree(DREAM.SURPRISAL.TREE_TYPE)

    tree = create_tree(
        tree_type=DREAM.SURPRISAL.TREE_TYPE,
        k=DREAM.SURPRISAL.TREE_K,
    )

    tree.batch_insert(embeddings)

    return tree


def _compute_surprisal_scores(
    observations: list[ObservationData],
    embeddings: np.ndarray,
    tree: SurprisalTree,
) -> list[SurprisalScore]:
    """
    Compute surprisal score for each observation.

    Args:
        observations: List of ObservationData objects
        embeddings: np.ndarray of embeddings matching observations
        tree: Built SurprisalTree

    Returns:
        List of SurprisalScore objects (unfiltered, unsorted)
    """
    scores: list[SurprisalScore] = []

    for obs, embedding in zip(observations, embeddings, strict=False):
        surprisal = tree.surprisal(embedding)

        scores.append(
            SurprisalScore(
                observation=obs,
                surprisal=surprisal,
                embedding=embedding,
            )
        )

    return scores


def _normalize_scores(scores: list[SurprisalScore]) -> list[SurprisalScore]:
    """
    Normalize surprisal scores to [0, 1] range using min-max normalization.

    Args:
        scores: List of SurprisalScore objects with raw surprisal values

    Returns:
        List of SurprisalScore objects with normalized surprisal values
    """
    if not scores:
        return []

    # Handle edge case: all scores are identical
    surprisal_values = [s.surprisal for s in scores]
    min_surprisal = min(surprisal_values)
    max_surprisal = max(surprisal_values)

    if max_surprisal == min_surprisal:
        # All scores identical - set all to 0.5 (middle of range)
        return [SurprisalScore(observation=s.observation, surprisal=0.5, embedding=s.embedding) for s in scores]

    # Min-max normalization: (x - min) / (max - min)
    normalized: list[SurprisalScore] = []
    for score in scores:
        normalized_value = (score.surprisal - min_surprisal) / (max_surprisal - min_surprisal)
        normalized.append(
            SurprisalScore(
                observation=score.observation,
                surprisal=normalized_value,
                embedding=score.embedding,
            )
        )

    return normalized


def _filter_by_percent(scores: list[SurprisalScore]) -> list[SurprisalScore]:
    """
    Filter observations by top percentage.

    Assumes scores are already sorted by surprisal (highest first).

    Args:
        scores: List of SurprisalScore objects, sorted by surprisal DESC

    Returns:
        Filtered list of SurprisalScore objects (top N% by surprisal)
    """
    if not scores:
        return []

    # Take top percentage
    top_percent = DREAM.SURPRISAL.TOP_PERCENT_SURPRISAL
    count = max(1, int(len(scores) * top_percent))  # At least 1 observation

    return scores[:count]
