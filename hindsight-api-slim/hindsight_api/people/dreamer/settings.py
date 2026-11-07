# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/config.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""Dream settings as module constants.

Honcho reads these from ``settings.DREAM`` (config.toml and environment). The fork has no such
settings object: the values below are the defaults, kept as one mutable dataclass instance so a
test or a caller can change one in place (``DREAM.NOTES_MODE = "on"``). The per-specialist model
configs (``DEDUCTION_MODEL_CONFIG``, ``INDUCTION_MODEL_CONFIG``) are not copied: every dream call
goes through the engine's Claude-configured background provider (``ctx.llm_background``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SurprisalSettings:
    """Settings for tree-based surprisal sampling during dreams."""

    ENABLED: bool = False

    # Tree configuration
    TREE_TYPE: Literal["kdtree", "balltree", "rptree", "covertree", "lsh", "graph", "prototype"] = "kdtree"
    TREE_K: int = 5  # k for kNN-based trees

    # Sampling strategy
    SAMPLING_STRATEGY: Literal["recent", "random", "all"] = "recent"
    SAMPLE_SIZE: int = 200

    # Surprisal filtering (normalized scores: 0.0 = lowest, 1.0 = highest)
    TOP_PERCENT_SURPRISAL: float = 0.10  # Top 10% of observations
    # Hybrid mode: min high-surprisal observations to replace standard questions
    MIN_HIGH_SURPRISAL_FOR_REPLACE: int = 10

    # Observation level filtering
    INCLUDE_LEVELS: list[str] = field(default_factory=lambda: ["explicit", "deductive"])


@dataclass
class DreamSettings:
    ENABLED: bool = True
    DOCUMENT_THRESHOLD: int = 50
    IDLE_TIMEOUT_MINUTES: int = 60
    MIN_HOURS_BETWEEN_DREAMS: int = 8
    DUE_POLL_INTERVAL_SECONDS: int = 300
    MAX_ENQUEUED_PER_POLL: int = 100
    ENABLED_TYPES: list[str] = field(default_factory=lambda: ["omni"])

    # Card-refresh dreams only; capped further by the specialist's own ceiling.
    # Deduction and induction use the per-specialist limits below.
    MAX_TOOL_ITERATIONS: int = 20
    DEDUCTION_MAX_TOOL_ITERATIONS: int = 12
    INDUCTION_MAX_TOOL_ITERATIONS: int = 10
    # Deduction and induction start with a description of how memory is
    # organized and conclusion counts by level.
    BRIEF_ENABLED: bool = False
    # Cap on the since-last-dream work list in the brief. A larger backlog is
    # worked through over consecutive dreams.
    BRIEF_MAX_NEW_CONCLUSIONS: int = 50
    # Token cap on the same list: conclusions created through the API can be
    # long. At least one is always shown, so one huge conclusion can't stall it.
    BRIEF_MAX_NEW_CONCLUSION_TOKENS: int = 8000
    # Deduction and induction see a tool-call budget line, the last iterations
    # are reserved for writes, and the final call at the limit can still write.
    BUDGET_ENABLED: bool = False
    # Working notes carried between dreams: "off", "shadow" (written but never
    # loaded, so a sample can be judged) or "on".
    NOTES_MODE: Literal["off", "shadow", "on"] = "off"
    # Split evenly between deduction's and induction's own entries.
    NOTES_MAX_TOKENS: int = 3000
    # Iterations reserved for writes at the end of a pass. Unset means 15% of
    # the specialist's limit, rounded, with a minimum of one, and at most one
    # less than the limit.
    WRITE_WINDOW_ITERATIONS: int | None = None
    # Representation History: each dream's created and removed conclusions and
    # peer-card changes, readable through the dreams API. Past the retention
    # window the prune keeps a run's created conclusions that still exist, and
    # deletes the run once none do.
    HISTORY_ENABLED: bool = True
    HISTORY_RETENTION_DAYS: int = 30

    # Surprisal-based sampling subsystem
    SURPRISAL: SurprisalSettings = field(default_factory=SurprisalSettings)


DREAM = DreamSettings()
