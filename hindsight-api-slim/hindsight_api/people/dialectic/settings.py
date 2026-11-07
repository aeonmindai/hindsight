# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/config.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""Module constants for the dialectic.

Honcho reads these from ``settings.DIALECTIC`` (config.toml and environment). The fork has no such
settings: the values are constants here, with Honcho's defaults. The model is not configured per
level: every level runs on the engine's interactive (reflect) provider, ``ctx.llm_interactive``.
"""

from dataclasses import dataclass
from typing import Literal, get_args

ReasoningLevel = Literal["minimal", "low", "medium", "high", "max"]
REASONING_LEVELS: list[ReasoningLevel] = list(get_args(ReasoningLevel))


@dataclass(frozen=True)
class DialecticLevelSettings:
    """Settings for a specific reasoning level in the dialectic."""

    MAX_TOOL_ITERATIONS: int
    # None means use the global MAX_OUTPUT_TOKENS.
    MAX_OUTPUT_TOKENS: int | None = None
    # None/"auto" lets the model answer without a tool call, "any"/"required" forces a tool call first.
    TOOL_CHOICE: str | None = None


LEVELS: dict[ReasoningLevel, DialecticLevelSettings] = {
    "minimal": DialecticLevelSettings(MAX_TOOL_ITERATIONS=1, MAX_OUTPUT_TOKENS=250, TOOL_CHOICE="auto"),
    "low": DialecticLevelSettings(MAX_TOOL_ITERATIONS=5, TOOL_CHOICE="auto"),
    "medium": DialecticLevelSettings(MAX_TOOL_ITERATIONS=2),
    "high": DialecticLevelSettings(MAX_TOOL_ITERATIONS=4),
    "max": DialecticLevelSettings(MAX_TOOL_ITERATIONS=10),
}

MAX_OUTPUT_TOKENS = 8192
# Estimated input tokens after which the tool loop stops researching and answers.
MAX_INPUT_TOKENS = 100_000
# Longest chat query accepted, in characters.
MAX_QUERY_LENGTH = 10_000
# Extra tool rounds bank chat gets on top of its level's limit (not applied at "minimal").
WORKSPACE_EXTRA_TOOL_ITERATIONS = 3
# Show a tool-call budget line after each round, at levels allowing three or more tool iterations.
BUDGET_ENABLED = False
# Add the memory-layout section to the system prompt and, for unrestricted pair queries, conclusion counts
# to the first message.
BRIEF_ENABLED = False
# Recent messages of the session to put in the system prompt, in tokens. 0 turns it off.
SESSION_HISTORY_MAX_TOKENS = 4_096
