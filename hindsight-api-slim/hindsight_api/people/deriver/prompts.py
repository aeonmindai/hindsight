# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/deriver/prompts.py
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/utils/formatting.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""
Minimal prompts for the deriver module optimized for speed.

This module contains simplified prompt templates focused only on observation extraction.
NO peer card instructions, NO working representation - just extract observations.
"""

import re
from datetime import datetime
from functools import cache
from inspect import cleandoc as c

from ...engine.token_encoding import count_tokens

_MESSAGE_TAG = re.compile(r"<(?=/?message\b)", re.IGNORECASE)


def estimate_tokens(text: str | list[str] | None) -> int:
    """Token count of a text (or lines joined by newlines) under the engine's configured encoding."""
    if not text:
        return 0
    if isinstance(text, list):
        text = "\n".join(text)
    return count_tokens(text)


def custom_instructions_section(
    custom_instructions: str | None,
    *,
    heading: str = "CUSTOM INSTRUCTIONS:",
    note: str | None = None,
) -> str:
    """
    Render operator-configured custom instructions as a prompt section.

    Args:
        custom_instructions: Resolved instructions; blank or None renders nothing
        heading: Section heading line
        note: Optional line between the heading and the instructions

    Returns:
        The rendered section, or an empty string when there are no instructions
    """
    normalized = (custom_instructions or "").strip()
    if not normalized:
        return ""
    lines = [heading, note, normalized] if note else [heading, normalized]
    return "\n".join(lines)


def format_deriver_message(idx: int, peer: str, target: str, created_at: datetime, content: str) -> str:
    """Wrap one batch message in a tag carrying its index, author, and target flag.

    Honcho restricts peer ids to ``[a-zA-Z0-9_-]``; person handles here are free text, so the author
    attribute has ``<``, ``>`` and ``"`` escaped. Any ``<message``/``</message`` inside the content is
    neutralized so a message cannot forge its own tag boundary.
    """
    is_target = "true" if peer == target else "false"
    time_str = created_at.strftime("%Y-%m-%d %H:%M:%S")
    safe_content = _MESSAGE_TAG.sub("&lt;", content)
    safe_peer = peer.replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    return f'<message idx="{idx}" peer="{safe_peer}" target="{is_target}" time="{time_str}">{safe_content}</message>'


def _normalized_custom_instructions(custom_instructions: str | None) -> str | None:
    """Return stripped custom instructions, if any."""
    if custom_instructions is None:
        return None

    normalized = custom_instructions.strip()
    return normalized or None


def minimal_deriver_prompt(
    peer_id: str,
    messages: str,
    custom_instructions: str | None = None,
) -> str:
    """
    Generate minimal prompt for fast observation extraction.

    Args:
        peer_id: The ID of the user being analyzed.
        messages: Batch messages, each wrapped by ``format_deriver_message``.

    Returns:
        Formatted prompt string for observation extraction.
    """
    instructions_section = custom_instructions_section(
        custom_instructions,
        note="These instructions apply to the target peer identified below.",
    )
    return c(
        f"""
Analyze messages to extract **explicit atomic facts** about the target peer.

[EXPLICIT] DEFINITION: Facts about the target peer that can be derived directly from their messages.
   - Transform statements into one or multiple conclusions
   - Each conclusion must be self-contained with enough context
   - Use absolute dates/times when possible, resolving relative references against the message `time` attribute

RULES:
- The target peer is the peer identified below under `Target peer:`.
- A peer can be a human user, AI agent, bot, service, or other actor.
- Each message is wrapped as `<message idx="N" peer="..." target="true|false" time="...">`. `target="true"` marks messages authored by the target peer; `target="false"` marks everyone else.
- Extract ALL observations from `target="true"` messages. Use `target="false"` messages only as context to interpret them; never derive a fact about the target peer from what another peer said, did, or reported.
- When the target peer answers a question or accepts a proposal from another peer ("yes", "the first one"), the details in that question or proposal count as stated by the target peer. A bare acknowledgement ("ok", "thanks") does not.
- A batch may contain few or no `target="true"` messages, even when it holds many long messages from other peers (agent turns, tool output, system notices). In that case produce few or no conclusions.
- For each observation, set `source_indices` to the `idx` values of the messages it rests on: the `target="true"` messages that state it, plus any `target="false"` message needed to interpret them (e.g. the question answered by "the first one"). Cite only messages that directly support the observation.
- Do not record statistics or rates about how the peer writes (emoji, message length, punctuation, how often they use anything). Record how they want to be addressed or treated only when they say so.
- Do not record conversational mechanics (greeting, thanking, acknowledging, asking or answering a question) or anything true of nearly every participant.
- Each conclusion states exactly one independent fact. Keep a fact's reason, purpose, tool, place, and qualifiers attached to it; they are part of the same fact. Split when a conclusion joins independent facts, or when a clause adds a new fact about a person or thing.
- Do not emit two conclusions that state the same fact in different words.
- Properly attribute observations to the correct subject. If the target peer is referencing someone or something else, make that clear.
- Write each observation for a reader who knows only the peer id: identify other people and things by their relation to this peer, and leave no pronoun, "it", "there", or relative time unresolved.
- Include only what is needed to identify who or what the observation is about and to state the fact. Do not add explanation or background.

{instructions_section}

Target peer:
{peer_id}

Write `{peer_id}` as the subject of every observation about this peer. Never write "the target peer" or "the user" in its place.

Messages to analyze:
<messages>
{messages}
</messages>
"""
    )


@cache
def estimate_minimal_deriver_prompt_tokens() -> int:
    """Estimate the static minimal deriver prompt without custom instructions."""
    prompt = minimal_deriver_prompt(
        peer_id="",
        messages="",
        custom_instructions=None,
    )
    return estimate_tokens(prompt)


def estimate_deriver_prompt_tokens(custom_instructions: str | None) -> int:
    """Estimate minimal deriver prompt tokens, including custom instructions if present."""
    normalized_custom_instructions = _normalized_custom_instructions(custom_instructions)
    if normalized_custom_instructions is None:
        return estimate_minimal_deriver_prompt_tokens()

    prompt = minimal_deriver_prompt(
        peer_id="",
        messages="",
        custom_instructions=normalized_custom_instructions,
    )
    return estimate_tokens(prompt)
