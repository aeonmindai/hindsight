# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/dreamer/specialists.py
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/utils/formatting.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""
Agentic specialists for the dream cycle.

Each specialist is a fully autonomous agent that:
1. Receives probing questions as entry points
2. Uses tools to search for relevant observations
3. Creates new observations (deductive or inductive)
4. Can delete duplicates (deduction only)
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

from .. import conclusions as conclusions_crud
from .. import peer_card as peer_card_crud
from ..context import PeopleContext
from ..dream_history import DreamHistoryTarget
from ..ids import generate_nanoid
from ..memory_brief import format_memory_counts, memory_layout_section
from ..peers import get_peer
from ..schemas import ResolvedConfiguration
from .agent_tools import (
    CARD_REFRESH_SPECIALIST_TOOLS,
    DEDUCTION_SPECIALIST_TOOLS,
    INDUCTION_SPECIALIST_TOOLS,
    ToolContext,
    create_tool_executor,
    get_tool_handler,
)
from .settings import DREAM
from .tool_loop import ToolLoopResult, ToolResult, execute_tool_loop, get_current_iteration
from .working_memory import (
    UPDATE_NOTES_TOOL,
    DreamState,
    LoadedNotes,
    format_notes,
    format_work_list,
    load_notes_for_pass,
    make_update_notes_handler,
    notes_share_tokens,
    notes_system_section,
)

logger = logging.getLogger(__name__)


@dataclass
class SpecialistResult:
    """Result of a specialist run for telemetry and aggregation."""

    run_id: str
    specialist_type: str
    iterations: int
    tool_calls_count: int
    input_tokens: int
    output_tokens: int
    duration_ms: float
    success: bool
    content: str
    # Rollups from the pass (Honcho reports these on DreamSpecialistEvent).
    iteration_limit: int = 0
    hit_iteration_limit: bool = False
    budget_line_enabled: bool = False
    write_window_used: bool = False
    refused_search_count: int = 0
    notes_added: int = 0
    notes_replaced: int = 0
    notes_removed: int = 0
    notes_entries_loaded: int = 0
    notes_tokens_loaded: int = 0
    pending_loaded: int = 0
    pending_resolved: int = 0
    pending_expired: int = 0
    new_conclusions_shown: int = 0
    new_conclusions_remaining: int = 0
    created_observation_count: int = 0
    deleted_observation_count: int = 0
    peer_card_updated: bool = False
    search_tool_calls_count: int = 0
    created_counts_by_level: dict[str, int] = field(default_factory=dict)
    deleted_counts_by_level: dict[str, int] = field(default_factory=dict)


# Tool names to exclude when peer card creation is disabled
PEER_CARD_TOOL_NAMES = {"update_peer_card"}

# Opening instruction when the since-last-dream list is in the prompt; it
# replaces "start with get_recent_observations", which would re-fetch it.
WORK_LIST_START = (
    "Start from the conclusions under NEW SINCE THE LAST DREAM, and search "
    + "older memory only where they bear on it."
)

# Tools that only read memory. They are refused during the write window and
# left out of the final call at the limit.
SEARCH_TOOL_NAMES = frozenset({"get_recent_observations", "search_memory", "search_messages"})


def write_window_size(limit: int) -> int:
    """Iterations reserved for writes at the end of a pass of `limit` rounds."""
    override = DREAM.WRITE_WINDOW_ITERATIONS
    # 15% rounded half up: 2 for both deduction (12) and induction (10).
    size = override if override is not None else int(limit * 0.15 + 0.5)
    # Leave at least one round for searching, unless the limit is one round.
    return min(max(1, limit - 1), max(1, size))


def _reserved_calls_phrase(first: int, limit: int) -> str:
    if first == limit:
        return f"Call {limit} is"
    if first + 1 == limit:
        return f"Calls {first} and {limit} are"
    return f"Calls {first} to {limit} are"


def dream_budget_line(window: int) -> Callable[[int, int], str | None]:
    """Budget line for a dreamer pass whose last `window` rounds are writes only."""

    def line(used: int, limit: int) -> str | None:
        if used >= limit:
            # The loop is about to make its final call, which says so itself.
            return None
        first_write_round = limit - window + 1
        if used + 1 >= first_write_round:
            return (
                f"Tool calls used: {used} of {limit}. Searches are now refused: "
                + "write what you have decided on, or finish with DONE."
            )
        reserved = _reserved_calls_phrase(first_write_round, limit)
        return f"Tool calls used: {used} of {limit}. {reserved} for writing; " + "searches will be refused."

    return line


def budget_system_section(limit: int, window: int) -> str:
    """System-prompt section explaining the budget and the write window."""
    reserved = "is" if window == 1 else "are"
    last = "last call" if window == 1 else f"last {window} calls"
    return f"""## BUDGET

You have {limit} tool calls. The {last} {reserved} reserved for writing: searches made then are refused. After each round of results, a line tells you how many calls you have used. If you reach the limit, you get one final call that can only write.

Ending without writing anything is a valid outcome when nothing new is worth writing. Finish with DONE."""


def _handler_resolver(
    first_write_round: int | None,
    extra_handlers: dict[str, Callable[[ToolContext, dict[str, Any]], Any]],
) -> Callable[[str], Callable[[ToolContext, dict[str, Any]], Any] | None]:
    """Handler lookup for a dreamer pass.

    Refuses search tools from `first_write_round` on (None means never), and
    serves `extra_handlers` for tools outside the shared handler table.
    """

    async def refuse(_ctx: ToolContext, _tool_input: dict[str, Any]) -> ToolResult:
        return ToolResult(
            content=(
                "Searches are refused for the rest of this pass. Write what you "
                + "have decided on now, or finish with DONE."
            ),
            metadata={"refused": True},
        )

    def resolve(tool_name: str) -> Callable[[ToolContext, dict[str, Any]], Any] | None:
        if (
            first_write_round is not None
            and tool_name in SEARCH_TOOL_NAMES
            and get_current_iteration() >= first_write_round
        ):
            return refuse
        if tool_name in extra_handlers:
            return extra_handlers[tool_name]
        return get_tool_handler(tool_name)

    return resolve


# Shared PEER CARD system-prompt section (identity-store taxonomy + rules).
# Used verbatim by DeductionSpecialist and CardRefreshSpecialist.
PEER_CARD_SYSTEM_SECTION = """

## PEER CARD (REQUIRED)

The peer card is the target observee's identity store: stable identity markers that distinguish this entity from others and persist across interactions. Behavior, tendencies, transient state, and episodic facts belong in observations, not on the peer card.

A peer can be anything with identity that changes over time — a human, an agent, a codebase, a team, an organization. Do not assume the target observee is human. Do not require any field; empty is the correct output when evidence is absent.

### Allowed entry kinds

Each entry must start with one of these four prefixes (exact case, followed by a space):

- `IDENTITY: ...` — canonical name, kind, aliases, IDs
  - `IDENTITY: Name: Alice`
  - `IDENTITY: Kind: Python monorepo`
  - `IDENTITY: Version: 4.2`
  - `IDENTITY: Aliases: alice@example.com`
- `ATTRIBUTE: ...` — stable durable property of the entity (including explicitly stated standing preferences)
  - `ATTRIBUTE: Location: NYC`
  - `ATTRIBUTE: Language: Python`
  - `ATTRIBUTE: Prefers tea`
  - `ATTRIBUTE: Charter: ship Honcho infrastructure`
- `RELATIONSHIP: ...` — durable link to another entity
  - `RELATIONSHIP: Spouse: Bob`
  - `RELATIONSHIP: Maintainer: vineeth`
  - `RELATIONSHIP: Members: vineeth, rajat`
- `INSTRUCTION: ...` — standing rule of engagement that the target observee has explicitly stated (do/don't for the observer). Only when explicit; never inferred from behavior.
  - `INSTRUCTION: Call me Vee`
  - `INSTRUCTION: Never push to main without review`

### Rules

1. **Stable.** If the value plausibly changes within six months absent a deliberate announcement, it does not belong on the card. Prefer leaving the card empty over filling it with volatile content.
2. **Subject is the target observee.** Every entry must be a fact about the target observee, not about another participant in the session. Never write facts about co-occurring peers into the card, no matter how frequently they appear in the messages.
3. **Evidence-grounded.** Only write what the target observee has explicitly stated, or what another participant has explicitly stated about the target observee with the target observee's assent. No "general knowledge" inferences (`"co-founder"` does not imply an age; mentioning a colleague does not imply a family relationship).
4. **Type-agnostic.** The target observee may not be human. Do not require name/age/location/family/occupation fields.
5. **No behavioral content.** TRAITs, behavioral tendencies, patterns, and inferred preferences belong in observations, not on the peer card. Do not write `TRAIT:` entries or behavioral `PREFERENCE:` entries — they will be rejected.
6. **No evidence bundles.** Each entry is one concise fact. No `e.g.` clauses, no parenthetical example lists, no semicolon-separated value dumps.

### Migrating an existing peer card

The CURRENT PEER CARD shown in the user message may contain entries from an older format that do not start with an allowed prefix (e.g. `Name: Alice`, `Lives in NYC`, `TRAIT: Analytical`, `PREFERENCE: Detailed explanations`). When you call `update_peer_card`, you are responsible for re-emitting the entries you want to keep — entries you omit are dropped, and entries without an allowed prefix are silently rejected.

For each legacy entry:

- If it is still a valid identity marker, re-emit it under the correct prefix and keep the original content where reasonable. Examples:
  - `Name: Alice` → `IDENTITY: Name: Alice`
  - `Lives in NYC` → `ATTRIBUTE: Location: NYC`
  - `Works at Google` → `ATTRIBUTE: Employer: Google`
  - `INSTRUCTION: Call me Vee` → keep as is (already correctly prefixed)
- Drop entries that violate the rules above: behavioral `TRAIT:` lines, inferred behavioral `PREFERENCE:` lines, one-off events, transient state, evidence bundles. Do not re-prefix them — they are not identity markers.

When in doubt about a specific legacy entry, prefer migrating it (so valid info isn't lost) over dropping it. Splitting one dense legacy entry into multiple correctly-prefixed entries is fine and encouraged (e.g. a semicolon-separated `Tech Stack:` dump can become several `ATTRIBUTE:` lines, one per durable tool/platform).

Call `update_peer_card` with the complete deduplicated list when there is a durable identity update to record, or when the existing card needs migration. Entries that do not start with one of the four allowed prefixes will be rejected. Keep concise (max 40 entries)."""


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


def custom_instructions_prompt(configuration: ResolvedConfiguration | None, *, peer_card_enabled: bool) -> str:
    """Render the dream and peer card custom instructions for a specialist.

    Peer card instructions are additive on top of the dream instructions and
    only apply when the specialist can write the card.
    """
    if configuration is None:
        return ""
    sections = [custom_instructions_section(configuration.dream.custom_instructions)]
    if peer_card_enabled:
        sections.append(
            custom_instructions_section(
                configuration.peer_card.custom_instructions,
                heading="PEER CARD CUSTOM INSTRUCTIONS:",
                note="Apply these within the allowed entry kinds and rules of the PEER CARD section.",
            )
        )
    return "".join(f"\n\n{section}" for section in sections if section)


class BaseSpecialist(ABC):
    """Base class for agentic specialists."""

    name: str = "base"
    # Whether this specialist is allowed to write to the peer card. Defaults to True;
    # specialists that should never touch the card (e.g., induction) override to False.
    can_update_peer_card: bool = True
    # Whether the current peer card is fetched and injected into the user prompt.
    # Card-refresh runs in rebuild mode set this to False so the card is
    # reconstructed solely from observations present in the collection.
    inject_peer_card: bool = True
    # Subclasses can override to customize the peer card update instruction
    peer_card_update_instruction: str = "Only update this with durable identity markers via `update_peer_card`."
    # Whether the working-memory settings apply: DREAM.BRIEF_ENABLED (memory
    # layout, counts, work list), DREAM.BUDGET_ENABLED (budget line, write
    # window) and DREAM.NOTES_MODE (working notes).
    uses_working_memory: bool = True

    @abstractmethod
    def get_tools(self, *, peer_card_enabled: bool = True) -> list[dict[str, Any]]:
        """Get the tools available to this specialist."""
        ...

    def get_max_tokens(self) -> int:
        """Get max output tokens for this specialist."""
        return 16384

    def get_max_iterations(self) -> int:
        """Get max tool iterations."""
        return 15

    @abstractmethod
    def build_system_prompt(self, observed: str, *, peer_card_enabled: bool = True) -> str:
        """Build the system prompt for this specialist."""
        ...

    @abstractmethod
    def build_user_prompt(
        self,
        observed: str,
        hints: list[str] | None,
        peer_card: list[str] | None = None,
        *,
        work_list_shown: bool = False,
    ) -> str:
        """Build the user prompt with optional exploration hints and current peer card.

        `work_list_shown` means the since-last-dream list precedes this prompt,
        so the agent starts from it instead of re-fetching recent conclusions.
        """
        ...

    def _build_target_observee_context(self, observed: str) -> str:
        return f"""Target observee:
{observed}

The target observee is the peer identified above. When created observations need to name this subject, use the exact observee id above, not the phrase "the target observee".

"""

    def _build_peer_card_context(self, peer_card: list[str] | None) -> str:
        """Build the peer card context section for user prompts."""
        if not peer_card:
            return ""
        facts = "\n".join(f"- {fact}" for fact in peer_card)
        return f"""
## CURRENT PEER CARD

{facts}

{self.peer_card_update_instruction}
If you update it, send the full deduplicated list and remove stale entries.

"""

    async def run(
        self,
        ctx: PeopleContext,
        bank_id: str,
        observer: str,
        observed: str,
        session_name: str | None,
        hints: list[str] | None = None,
        configuration: ResolvedConfiguration | None = None,
        parent_run_id: str | None = None,
        *,
        state: DreamState | None = None,
        history: DreamHistoryTarget | None = None,
    ) -> SpecialistResult:
        """
        Run the specialist agent.

        Takes connections only for short reads and writes, never across a model call.

        Args:
            ctx: People context (engine and Claude-configured providers)
            bank_id: Bank identifier
            observer: The observing peer
            observed: The peer being observed
            session_name: Session identifier
            hints: Optional hints to guide exploration (specialists explore freely if None)
            configuration: Resolved configuration for checking feature flags (optional)
            parent_run_id: Optional run_id from orchestrator for correlation
            state: Working state loaded by the orchestrator (cursor, notes and
                the since-last-dream work list), when any of it is enabled
            history: The orchestrator's open Representation History run. Its
                writes are recorded there under this specialist's name.

        Returns:
            SpecialistResult with metrics and content
        """
        run_id = parent_run_id or generate_nanoid()
        start_time = time.perf_counter()

        # Rollups counted from the tool results' metadata, never from tool names: one create call can
        # make N (or zero) observations.
        created_observation_count = 0
        deleted_observation_count = 0
        peer_card_updated = False
        search_tool_calls_count = 0
        refused_search_count = 0
        notes_added = 0
        notes_replaced = 0
        notes_removed = 0
        pending_resolved = 0
        loaded_notes: LoadedNotes | None = None
        created_counts_by_level: Counter[str] = Counter()
        deleted_counts_by_level: Counter[str] = Counter()

        # Determine if peer card tools should be included. Specialists that
        # cannot write to the peer card (e.g., induction) skip the fetch and
        # the prompt section entirely.
        peer_card_enabled = self.can_update_peer_card and (configuration is None or configuration.peer_card.create)
        injected_instructions = custom_instructions_prompt(configuration, peer_card_enabled=peer_card_enabled)
        brief_enabled = self.uses_working_memory and DREAM.BRIEF_ENABLED
        budget_enabled = self.uses_working_memory and DREAM.BUDGET_ENABLED
        max_iterations = self.get_max_iterations()
        window = write_window_size(max_iterations)
        # Notes need the dream number from the loaded state.
        notes_enabled = self.uses_working_memory and DREAM.NOTES_MODE != "off" and state is not None
        work_list = state.work_list if state is not None else None

        await get_peer(ctx, bank_id, observer)
        if observer != observed:
            await get_peer(ctx, bank_id, observed)

        # Fetch current peer card to inject into prompt (saves a tool call).
        # Skipped when inject_peer_card is False (card-refresh rebuild
        # mode): the card must be reconstructed from observations only.
        current_peer_card: list[str] | None = None
        if peer_card_enabled and self.inject_peer_card:
            current_peer_card = await peer_card_crud.get_peer_card(ctx, bank_id, observer, observed)

        memory_counts: str | None = None
        if brief_enabled:
            memory_counts = format_memory_counts(
                await conclusions_crud.summarize_documents_by_level(ctx, bank_id, observer=observer, observed=observed)
            )

        # In shadow mode notes are written but never loaded, so a sample can
        # be judged before they shape any dream.
        if notes_enabled and state is not None and DREAM.NOTES_MODE == "on":
            # Like the dream state, the notes are optional: a failure here
            # runs the pass without them rather than failing it.
            try:
                loaded_notes = await load_notes_for_pass(
                    ctx,
                    bank_id,
                    observer=observer,
                    observed=observed,
                    dream_number=state.dream_number,
                    epoch=state.notes_epoch,
                )
            except Exception:
                logger.warning("Loading dream notes failed; running without them", exc_info=True)

        tools = self.get_tools(peer_card_enabled=peer_card_enabled)
        if notes_enabled:
            tools = [*tools, UPDATE_NOTES_TOOL]
        system_prompt = self.build_system_prompt(observed, peer_card_enabled=peer_card_enabled)
        if brief_enabled:
            system_prompt += "\n\n" + memory_layout_section(
                available_tools=(tool["name"] for tool in tools), cite_ids=True
            )
        if budget_enabled:
            system_prompt += "\n\n" + budget_system_section(max_iterations, window)
        if notes_enabled:
            system_prompt += "\n\n" + notes_system_section(notes_share_tokens())
        work_list_shown = brief_enabled and work_list is not None
        user_prompt = self.build_user_prompt(
            observed=observed,
            hints=hints,
            peer_card=current_peer_card,
            work_list_shown=work_list_shown,
        )
        if work_list_shown and work_list is not None:
            user_prompt = f"{format_work_list(work_list)}\n\n{user_prompt}"
        if loaded_notes is not None:
            user_prompt = f"{format_notes(loaded_notes.entries, specialist=self.name)}\n\n{user_prompt}"
        if memory_counts is not None:
            user_prompt = f"{memory_counts}\n\n{user_prompt}"

        # Build messages
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt + injected_instructions},
            {"role": "user", "content": user_prompt},
        ]

        # Create tool executor
        tool_executor: Callable[[str, dict[str, Any]], Any] = await create_tool_executor(
            ctx,
            bank_id=bank_id,
            observer=observer,
            observed=observed,
            session_name=session_name,
            include_observation_ids=True,
            configuration=configuration,
            agent_type=self.name,
            history=(DreamHistoryTarget(run_id=history.run_id, specialist=self.name) if history is not None else None),
            handler_resolver=(
                _handler_resolver(
                    max_iterations - window + 1 if budget_enabled else None,
                    {
                        "update_notes": make_update_notes_handler(
                            specialist=self.name,
                            run_id=run_id,
                            dream_number=state.dream_number,
                            epoch=state.notes_epoch,
                        )
                    }
                    if notes_enabled and state is not None
                    else {},
                )
                if budget_enabled or notes_enabled
                else None
            ),
        )

        # Run the agent loop on the engine's background provider.
        response: ToolLoopResult = await execute_tool_loop(
            ctx.llm_background,
            messages=messages,
            tools=tools,
            tool_executor=tool_executor,
            max_tool_iterations=max_iterations,
            max_tokens=self.get_max_tokens(),
            scope=f"people_dreamer_{self.name}",
            budget_line=dream_budget_line(window) if budget_enabled else None,
            final_call_tools=([t for t in tools if t["name"] not in SEARCH_TOOL_NAMES] if budget_enabled else None),
        )

        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"{self.name}: Completed in {duration_ms:.0f}ms, "
            + f"{len(response.tool_calls_made)} tool calls, "
            + f"{response.input_tokens} in / {response.output_tokens} out"
        )

        # Count actual observations created/deleted from the ToolResult.metadata stashed on each
        # tool-call record. Counting tool-name occurrences would mis-attribute: a single
        # create_observations call can produce N (or zero) observations.
        _search_tools = {"search_memory", "search_messages", "search_messages_temporal"}
        for tc in response.tool_calls_made:
            tool_name_any: Any = tc.get("tool_name") or tc.get("name")
            meta_any: Any = tc.get("tool_result_metadata") or {}
            if isinstance(meta_any, dict) and cast(dict[str, Any], meta_any).get("refused"):
                refused_search_count += 1
                continue
            if tool_name_any in _search_tools:
                search_tool_calls_count += 1
            if isinstance(meta_any, dict):
                meta_dict = cast(dict[str, Any], meta_any)
                created_val: Any = meta_dict.get("created_count") or 0
                deleted_val: Any = meta_dict.get("deleted_count") or 0
                created_observation_count += int(created_val)
                deleted_observation_count += int(deleted_val)
                if meta_dict.get("peer_card_updated"):
                    peer_card_updated = True
                notes_added += int(meta_dict.get("notes_added") or 0)
                notes_replaced += int(meta_dict.get("notes_replaced") or 0)
                notes_removed += int(meta_dict.get("notes_removed") or 0)
                pending_resolved += int(meta_dict.get("pending_resolved") or 0)
                # Accumulate per-level counts from create/delete observations: both handlers stash
                # {"levels": ["deductive", ...]} in metadata.
                levels_any: Any = meta_dict.get("levels")
                if isinstance(levels_any, list):
                    level_strs = [str(level) for level in cast(list[Any], levels_any) if level is not None]
                    # create_observations metadata has `created_count`, delete_observations `deleted_count`.
                    if "created_count" in meta_dict:
                        created_counts_by_level.update(level_strs)
                    elif "deleted_count" in meta_dict:
                        deleted_counts_by_level.update(level_strs)

        return SpecialistResult(
            run_id=run_id,
            specialist_type=self.name,
            iterations=response.iterations,
            tool_calls_count=len(response.tool_calls_made),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            duration_ms=duration_ms,
            success=True,
            # A final call that only writes has no text; callers slice this.
            content=response.content or "",
            iteration_limit=max_iterations,
            hit_iteration_limit=response.hit_iteration_limit,
            budget_line_enabled=budget_enabled,
            write_window_used=budget_enabled and response.iterations >= max_iterations - window + 1,
            refused_search_count=refused_search_count,
            notes_added=notes_added,
            notes_replaced=notes_replaced,
            notes_removed=notes_removed,
            notes_entries_loaded=len(loaded_notes.entries) if loaded_notes else 0,
            notes_tokens_loaded=loaded_notes.tokens if loaded_notes else 0,
            pending_loaded=loaded_notes.pending if loaded_notes else 0,
            pending_resolved=pending_resolved,
            pending_expired=loaded_notes.pending_expired if loaded_notes else 0,
            new_conclusions_shown=len(work_list.items) if brief_enabled and work_list is not None else 0,
            new_conclusions_remaining=work_list.remaining if brief_enabled and work_list is not None else 0,
            created_observation_count=created_observation_count,
            deleted_observation_count=deleted_observation_count,
            peer_card_updated=peer_card_updated,
            search_tool_calls_count=search_tool_calls_count,
            created_counts_by_level=dict(created_counts_by_level),
            deleted_counts_by_level=dict(deleted_counts_by_level),
        )


class DeductionSpecialist(BaseSpecialist):
    """
    Creates deductive observations from explicit observations.

    This specialist:
    1. Explores recent observations and messages to understand what's there
    2. Identifies logical implications, knowledge updates, and contradictions
    3. Creates new deductive observations with premise linkage
    4. Deletes outdated observations
    5. Updates peer card with biographical facts
    """

    name: str = "deduction"
    peer_card_update_instruction: str = "Update this with `update_peer_card` only for stable identity markers. See the PEER CARD section in the system prompt for the allowed entry kinds and rules."

    def get_tools(self, *, peer_card_enabled: bool = True) -> list[dict[str, Any]]:
        if peer_card_enabled:
            return DEDUCTION_SPECIALIST_TOOLS
        return [t for t in DEDUCTION_SPECIALIST_TOOLS if t["name"] not in PEER_CARD_TOOL_NAMES]

    def get_max_tokens(self) -> int:
        return 8192

    def get_max_iterations(self) -> int:
        return DREAM.DEDUCTION_MAX_TOOL_ITERATIONS

    def build_system_prompt(self, observed: str, *, peer_card_enabled: bool = True) -> str:
        _ = observed
        peer_card_section = ""
        if peer_card_enabled:
            peer_card_section = PEER_CARD_SYSTEM_SECTION

        return f"""You are a deductive reasoning agent analyzing observations about the target observee.

## YOUR JOB

Create deductive observations by finding logical implications in what's already known. Think like a detective connecting evidence.

## PHASE 1: DISCOVERY

Explore what's actually in memory. Use these tools freely:
- `get_recent_observations` - See what's been learned recently
- `search_memory` - Search for specific topics
- `search_messages` - See actual conversation content

Spend a few tool calls understanding the landscape before creating anything.

## PHASE 2: ACTION

Once you understand what's there, create observations and clean up:

### Knowledge Updates (HIGH PRIORITY)
When the same fact has different values at different times:
- "meeting Tuesday" [old] → "meeting moved to Thursday" [new]
- Create a deductive update observation
- DELETE the outdated observation immediately

### Logical Implications
Extract implicit information:
- "works as SWE at Google" → "has software engineering skills", "employed in tech"
- "has kids ages 5 and 8" → "is a parent", "has school-age children"

### Contradictions
When statements can't both be true (not just updates), flag them:
- "I love coffee" vs "I hate coffee" → contradiction observation
{peer_card_section}

## CREATING OBSERVATIONS

Use `create_observations_deductive`.

```json
{{
  "observations": [{{
    "content": "The logical conclusion",
    "source_ids": ["id1", "id2"],
    "premises": ["premise 1 text", "premise 2 text"]
  }}]
}}
```

## RULES

1. Don't explain your reasoning - just call tools
2. Create observations based on what you ACTUALLY FIND, not what you expect
3. Always include source_ids linking to the observations you're synthesizing
4. Copy source_ids exactly from the [id:xxx] shown in observation results - `search_messages` results have no ID and cannot be cited; invented IDs are discarded
5. Empty or missing source_ids will be rejected
6. Delete outdated observations - don't leave duplicates
7. Quality over quantity - fewer good deductions beat many weak ones
8. When you are finished, do not output a summary of what you did - output only the token DONE"""

    def build_user_prompt(
        self,
        observed: str,
        hints: list[str] | None,
        peer_card: list[str] | None = None,
        *,
        work_list_shown: bool = False,
    ) -> str:
        target_observee_context = self._build_target_observee_context(observed)
        peer_card_context = self._build_peer_card_context(peer_card)

        if hints:
            hints_str = "\n".join(f"- {q}" for q in hints[:5])
            begin = WORK_LIST_START if work_list_shown else "Begin with `get_recent_observations` to see what's there."
            return f"""{target_observee_context}{peer_card_context}Start by exploring recent observations and messages. These topics may be worth investigating:

{hints_str}

But follow the evidence - if you find something more interesting, pursue that instead.

{begin}"""

        start = (
            WORK_LIST_START
            if work_list_shown
            else "Start with `get_recent_observations` to see what's been learned recently, then investigate whatever seems most promising."
        )
        return f"""{target_observee_context}{peer_card_context}Explore the observation space and create deductive observations.

{start}

Look for:
1. Knowledge updates (same fact, different values over time)
2. Logical implications that haven't been made explicit
3. Contradictions that need flagging

Go."""


class InductionSpecialist(BaseSpecialist):
    """
    Creates inductive observations from explicit and deductive observations.

    This specialist:
    1. Explores observations to understand what's there
    2. Identifies patterns and generalizations across multiple observations
    3. Creates new inductive observations with source linkage

    Does not write to the peer card — the peer card stores stable identity markers,
    which is deduction's responsibility. Inductive patterns and tendencies stay as
    observations.
    """

    name: str = "induction"
    # Induction never writes to the peer card; behavioral patterns are observations.
    can_update_peer_card: bool = False

    def get_tools(self, *, peer_card_enabled: bool = True) -> list[dict[str, Any]]:
        _ = peer_card_enabled
        return INDUCTION_SPECIALIST_TOOLS

    def get_max_tokens(self) -> int:
        return 8192

    def get_max_iterations(self) -> int:
        return DREAM.INDUCTION_MAX_TOOL_ITERATIONS

    def build_system_prompt(self, observed: str, *, peer_card_enabled: bool = True) -> str:
        _ = observed
        _ = peer_card_enabled
        return """You are an inductive reasoning agent identifying patterns about the target observee.

## YOUR JOB

Create inductive observations by finding patterns across multiple observations. Think like a psychologist identifying behavioral tendencies.

## PHASE 1: DISCOVERY

Explore broadly to find patterns. Use these tools:
- `get_recent_observations` - Recent learnings
- `search_memory` - Topic-specific search
- `search_messages` - Actual conversation content

Look at BOTH explicit observations AND deductive ones. Patterns often emerge from synthesizing across both levels.

## PHASE 2: ACTION

Create inductive observations when you see patterns:

### Behavioral Patterns
- "Tends to reschedule meetings when stressed"
- "Makes decisions after consulting with partner"
- "Projects follow: enthusiasm → doubt → completion"

### Preferences
- "Prefers morning meetings"
- "Likes detailed technical explanations"
- Never a count or rate of how they write (emoji, message length, punctuation): that is not a preference.

### Personality Traits
- "Generally optimistic about outcomes"
- "Detail-oriented in planning"

### Temporal Patterns
- "Career goals have remained consistent"
- "Living situation changes frequently"

## CREATING OBSERVATIONS

Use `create_observations_inductive`.

```json
{{
  "observations": [{{
    "content": "The pattern or generalization",
    "source_ids": ["id1", "id2", "id3"],
    "sources": ["evidence 1", "evidence 2"],
    "pattern_type": "tendency", // preference|behavior|personality|tendency|correlation
    "confidence": "medium" // low (2 sources), medium (3-4), high (5+)
  }}]
}}
```

## RULES

1. Minimum 2 source observations required - patterns need evidence
2. Don't just restate a single fact as a pattern
3. Confidence based on evidence count: 2=low, 3-4=medium, 5+=high
4. Look for HOW things change over time, not just static facts
5. Include source_ids - always link back to evidence
6. Copy source_ids exactly from the [id:xxx] shown in observation results - `search_messages` results have no ID and cannot be cited; invented IDs are discarded
7. Empty or missing source_ids will be rejected
8. When you are finished, do not output a summary of what you did - output only the token DONE"""

    def build_user_prompt(
        self,
        observed: str,
        hints: list[str] | None,
        peer_card: list[str] | None = None,
        *,
        work_list_shown: bool = False,
    ) -> str:
        target_observee_context = self._build_target_observee_context(observed)
        # Induction does not consume peer card context — it produces inductive
        # observations, not identity-marker updates.
        _ = peer_card

        if hints:
            hints_str = "\n".join(f"- {q}" for q in hints[:5])
            return f"""{target_observee_context}Explore and find patterns. These areas may be worth investigating:

{hints_str}

But follow the evidence - if you find patterns elsewhere, pursue those.

{WORK_LIST_START if work_list_shown else "Start with `get_recent_observations`."}"""

        start = f"{WORK_LIST_START}\n\n" if work_list_shown else ""
        return f"""{target_observee_context}Explore the observation space and identify patterns.

{start}Remember: patterns need 2+ sources. Look for tendencies, preferences, and behavioral regularities.

Go."""


class CardRefreshSpecialist(BaseSpecialist):
    """
    Card-only maintenance specialist for the ``card_refresh`` dream type.

    Restricted to peer-card work: it may discover observations
    (get_recent_observations, search_memory) and rewrite the peer card
    (update_peer_card). It has NO observation-mutating tools — a card refresh
    must never create or delete observations.

    Two modes:
    - refresh (default): the current card is injected into the prompt and the
      specialist folds in new identity markers.
    - rebuild: the current card is NOT injected; the specialist reconstructs
      the card solely from observations present in the collection. Used after
      removals, where the old card may contain facts whose support was deleted.

    Not a singleton — instantiated per run because ``rebuild`` is per-dream
    state.
    """

    name: str = "card_refresh"
    uses_working_memory: bool = False
    peer_card_update_instruction: str = "Update this with `update_peer_card`. See the PEER CARD section in the system prompt for the allowed entry kinds and rules."

    # Low iteration ceiling for this lightweight, single-purpose run.
    MAX_ITERATIONS_CEILING: int = 6

    def __init__(self, *, rebuild: bool = False) -> None:
        self.rebuild: bool = rebuild
        # In rebuild mode the existing card is withheld from the prompt.
        self.inject_peer_card: bool = not rebuild

    def get_tools(self, *, peer_card_enabled: bool = True) -> list[dict[str, Any]]:
        if peer_card_enabled:
            return CARD_REFRESH_SPECIALIST_TOOLS
        # Defensive: a card refresh without card write access is a no-op, and
        # the orchestrator skips the run entirely when peer cards are disabled.
        return [t for t in CARD_REFRESH_SPECIALIST_TOOLS if t["name"] not in PEER_CARD_TOOL_NAMES]

    def get_max_tokens(self) -> int:
        return 8192

    def get_max_iterations(self) -> int:
        return min(self.MAX_ITERATIONS_CEILING, DREAM.MAX_TOOL_ITERATIONS)

    def build_system_prompt(self, observed: str, *, peer_card_enabled: bool = True) -> str:
        _ = observed
        _ = peer_card_enabled
        rebuild_section = ""
        if self.rebuild:
            rebuild_section = """

## REBUILD MODE

The existing peer card is deliberately NOT shown to you: it may contain entries whose supporting observations have since been removed. Build the card solely from the observations you find in the collection right now. Do not carry over or guess at prior card content — if an identity marker is not supported by a current observation, it does not go on the card."""

        return f"""You are a peer-card maintenance agent for the target observee.

## YOUR JOB

Refresh the peer card and nothing else. You cannot create or delete observations — you have no tools for that. Your only write operation is `update_peer_card`.

## PROCESS

1. Survey the observation space: start with `get_recent_observations`, then use `search_memory` for targeted follow-ups (names, roles, locations, standing instructions).
2. Extract stable identity markers supported by the observations you found.
3. Call `update_peer_card` once with the complete deduplicated list.

Keep it short — a handful of tool calls at most.{rebuild_section}
{PEER_CARD_SYSTEM_SECTION}"""

    def build_user_prompt(
        self,
        observed: str,
        hints: list[str] | None,
        peer_card: list[str] | None = None,
        *,
        work_list_shown: bool = False,
    ) -> str:
        _ = hints
        _ = work_list_shown
        target_observee_context = self._build_target_observee_context(observed)
        peer_card_context = self._build_peer_card_context(peer_card)

        if self.rebuild:
            return f"""{target_observee_context}Rebuild the peer card from scratch.

The previous card is not shown and must not be assumed: reconstruct the card solely from observations currently in the collection. Start with `get_recent_observations`, verify with `search_memory` where needed, then call `update_peer_card` with the complete list.

Go."""

        return f"""{target_observee_context}{peer_card_context}Refresh the peer card.

Review recent observations with `get_recent_observations` (and `search_memory` for targeted checks), then call `update_peer_card` with the complete deduplicated list if there is anything to add, correct, or migrate. If the card is already accurate and complete, finish without updating it.

Go."""


# Singleton instances
SPECIALISTS: dict[str, BaseSpecialist] = {
    "deduction": DeductionSpecialist(),
    "induction": InductionSpecialist(),
}
