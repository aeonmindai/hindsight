# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: tests/deriver/test_representation_crud.py
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: tests/integration/test_representation.py
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: tests/crud/test_representation_manager.py
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: tests/crud/test_conclusion_source_messages.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""Representation and conclusion tests, ported from Honcho's tests/deriver/test_representation_crud.py,
tests/integration/test_representation.py, tests/crud/test_representation_manager.py and
tests/crud/test_conclusion_source_messages.py.
"""

import datetime
import logging
import sys
import types
from datetime import UTC
from typing import Any

import pytest
from pydantic import ValidationError

from hindsight_api.people import schemas
from hindsight_api.people.conclusions import (
    Conclusion,
    delete_document,
    delete_document_by_id,
    delete_documents,
    get_all_documents,
    get_child_observations,
    get_documents_by_ids,
    query_documents,
    query_documents_most_derived,
    query_documents_recent,
    query_explicit_documents_after,
)
from hindsight_api.people.context import PeopleContext
from hindsight_api.people.exceptions import ResourceNotFoundException, ValidationException
from hindsight_api.people.representation import RepresentationManager, get_working_representation
from hindsight_api.people.representation_utils import (
    ALLOWLIST_SAFE_LEVELS,
    ContradictionObservation,
    DeductiveObservation,
    ExplicitObservation,
    ExplicitObservationBase,
    InductiveObservation,
    PromptRepresentation,
    Representation,
    allowlist_safe_levels,
    flatten_message_ids,
)
from tests.people.people_helpers import add_message, add_session, make_doc, save

NOW = datetime.datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)


# --- Representation classes (pure) ---------------------------------------------------------------


def test_prompt_representation_schema_orders_citations_before_content() -> None:
    schema = PromptRepresentation.model_json_schema()
    explicit_properties = schema["$defs"]["ExplicitObservationBase"]["properties"]
    assert list(explicit_properties) == ["source_indices", "content"]


def test_representation_is_empty_and_diff():
    """is_empty and diff_representation behave per the new definitions."""
    shared_time = NOW - datetime.timedelta(seconds=10)
    exp_shared_1 = ExplicitObservation(content="A", created_at=shared_time, message_ids=[1], session_name="s")
    exp_shared_2 = ExplicitObservation(content="B", created_at=shared_time, message_ids=[1], session_name="s")
    rep1 = Representation(explicit=[exp_shared_1], deductive=[])
    rep2 = Representation(
        explicit=[
            ExplicitObservation(content="A", created_at=shared_time, message_ids=[1], session_name="s"),
            exp_shared_2,
        ]
    )

    assert not rep1.is_empty()
    assert Representation().is_empty()

    diff = rep1.diff_representation(rep2)
    assert [e.content for e in diff.explicit] == ["B"]
    assert diff.deductive == []


def test_representation_formatting_methods():
    """__str__ and format_as_markdown produce expected section headers and content."""
    e = ExplicitObservation(content="has a dog", created_at=NOW, message_ids=[1], session_name="s")
    d = DeductiveObservation(
        created_at=NOW, message_ids=[1], session_name="s", conclusion="owns a pet", premises=[e.content]
    )
    rep = Representation(explicit=[e], deductive=[d])

    s = str(rep)
    assert "EXPLICIT:" in s
    assert "DEDUCTIVE:" in s
    assert "owns a pet" in s

    md = rep.format_as_markdown()
    assert "## Explicit Observations" in md
    assert "## Deductive Observations" in md
    assert "owns a pet" in md
    assert "Premises:" in md


def test_representation_class_creation_and_operations():
    explicit_obs1 = ExplicitObservation(
        content="User likes dogs", created_at=datetime.datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC), message_ids=[1]
    )
    explicit_obs2 = ExplicitObservation(
        content="User has a pet named Rover",
        created_at=datetime.datetime(2025, 1, 1, 12, 1, 0, tzinfo=UTC),
        message_ids=[2],
    )
    deductive_obs1 = DeductiveObservation(
        conclusion="User probably has a dog named Rover",
        premises=["User likes dogs", "User has a pet named Rover"],
        created_at=datetime.datetime(2025, 1, 1, 12, 2, 0, tzinfo=UTC),
        message_ids=[3],
    )
    representation = Representation(explicit=[explicit_obs1, explicit_obs2], deductive=[deductive_obs1])

    assert not representation.is_empty()
    assert representation.len() == 3
    assert "[2025-01-01 12:00:00] User likes dogs" in str(representation)
    no_timestamp_output = representation.str_no_timestamps()
    assert "User likes dogs" in no_timestamp_output
    assert "[" not in no_timestamp_output
    markdown_output = representation.format_as_markdown()
    assert "## Explicit Observations" in markdown_output
    assert "Premises:" in markdown_output


def test_representation_merging_and_diffing():
    def obs(content: str, hour: int, message_id: int) -> ExplicitObservation:
        return ExplicitObservation(
            content=content,
            created_at=datetime.datetime(2025, 1, 1, hour, 0, 0, tzinfo=UTC),
            message_ids=[message_id],
            session_name="session1",
        )

    rep1 = Representation(explicit=[obs("User likes cats", 10, 1)])
    rep2 = Representation(explicit=[obs("User likes cats", 10, 1), obs("User likes dogs", 11, 2)])

    diff = rep1.diff_representation(rep2)
    assert [o.content for o in diff.explicit] == ["User likes dogs"]

    # merge deduplicates and sorts by created_at
    rep1.merge_representation(rep2)
    assert [o.content for o in rep1.explicit] == ["User likes cats", "User likes dogs"]

    # merge with max_observations keeps the most recent (FIFO)
    rep1.merge_representation(Representation(explicit=[obs("User likes birds", 12, 3)]), max_observations=2)
    assert [o.content for o in rep1.explicit] == ["User likes dogs", "User likes birds"]


def test_str_with_ids_and_every_level_formats():
    explicit = ExplicitObservation(id="E" * 21, content="likes tea", created_at=NOW, message_ids=[1])
    deductive = DeductiveObservation(
        id="D" * 21, conclusion="drinks tea", premises=["likes tea"], created_at=NOW, message_ids=[1]
    )
    inductive = InductiveObservation(
        id="I" * 21,
        conclusion="is a tea person",
        sources=["likes tea"],
        confidence="high",
        created_at=NOW,
        message_ids=[],
    )
    contradiction = ContradictionObservation(
        id="C" * 21, content="likes and hates tea", sources=["likes tea", "hates tea"], created_at=NOW, message_ids=[]
    )
    rep = Representation(
        explicit=[explicit], deductive=[deductive], inductive=[inductive], contradiction=[contradiction]
    )

    with_ids = rep.str_with_ids()
    assert f"[id:{'E' * 21}] [2025-01-01 12:00:00] likes tea" in with_ids
    assert f"[id:{'I' * 21}] [2025-01-01 12:00:00] [high] is a tea person" in with_ids
    assert "CONTRADICTION: likes and hates tea" in with_ids
    markdown = rep.format_as_markdown(include_ids=True)
    assert f"[id:{'D' * 21}] [2025-01-01 12:00:00] drinks tea" in markdown
    assert "**Pattern** [high]: is a tea person" in markdown
    assert "**CONTRADICTION**: likes and hates tea" in markdown
    assert "CONTRADICTION:" in rep.str_no_timestamps()


def test_observation_equality_and_hash():
    a = ExplicitObservation(content="x", created_at=NOW, message_ids=[1], session_name="s")
    b = ExplicitObservation(content="x", created_at=NOW, message_ids=[2], session_name="s")
    c = ExplicitObservation(content="y", created_at=NOW, message_ids=[1], session_name="s")
    assert a == b and hash(a) == hash(b)
    assert a != c

    d1 = DeductiveObservation(conclusion="z", premises=["p1"], created_at=NOW, message_ids=[1], session_name="s")
    d2 = DeductiveObservation(conclusion="z", premises=["other"], created_at=NOW, message_ids=[1], session_name="s")
    assert d1 == d2 and len({d1, d2}) == 1


def test_observations_cite_their_source():
    said = datetime.datetime(2025, 1, 1, 9, 30, 0, tzinfo=UTC)
    cited = ExplicitObservation(
        content="moved to Berlin", created_at=NOW, message_ids=[1], source_url="https://slack/x/1", said_at=said
    )
    plain = ExplicitObservation(content="likes tea", created_at=NOW, message_ids=[1])
    assert str(cited) == "[2025-01-01 12:00:00] moved to Berlin (source: https://slack/x/1, 2025-01-01 09:30:00)"
    assert str(plain) == "[2025-01-01 12:00:00] likes tea"
    rep = Representation(explicit=[cited, plain])
    assert "(source: https://slack/x/1" in rep.format_as_markdown()
    assert "(source:" in rep.str_with_ids()

    derived = DeductiveObservation(
        conclusion="lives in Germany",
        premises=["moved to Berlin"],
        created_at=NOW,
        message_ids=[1],
        source_url="https://slack/x/1",
    )
    assert "lives in Germany (source: https://slack/x/1)" in str(derived)
    # the no-timestamp form is for comparing text, so it carries no citation
    assert "source" not in derived.str_no_timestamps()


def test_prompt_representation_conversion():
    """Only explicit observations are converted: the deriver produces explicit ones."""
    pr = PromptRepresentation(explicit=[ExplicitObservationBase(content="A")])
    rep = Representation.from_prompt_representation(
        pr, message_ids=[1], prompt_message_ids=["msg_public_id_one___"], session_name="s", created_at=NOW
    )
    assert [e.content for e in rep.explicit] == ["A"]
    assert len(rep.deductive) == 0
    assert rep.explicit[0].created_at == NOW


def test_empty_prompt_representation_conversion():
    rep = Representation.from_prompt_representation(
        PromptRepresentation(), message_ids=[1], prompt_message_ids=[], session_name="s", created_at=NOW
    )
    assert rep.is_empty()


def test_mixed_peer_source_indices_resolve_against_prompt_order(caplog: pytest.LogCaptureFixture) -> None:
    prompt_message_ids = ["bob_asks_______000010", "alice_picks____000020", "bob_acks_______000030"]
    prompt_representation = PromptRepresentation(
        explicit=[ExplicitObservationBase(content="Alice chose the first option", source_indices=[0, 1, 3])]
    )

    with caplog.at_level(logging.WARNING, logger="hindsight_api.people.representation_utils"):
        representation = Representation.from_prompt_representation(
            prompt_representation,
            message_ids=[20],
            prompt_message_ids=prompt_message_ids,
            session_name="s",
            created_at=NOW,
        )

    # bob's question is cited as context; alice's answer is the fact.
    assert representation.explicit[0].source_message_ids == ["bob_asks_______000010", "alice_picks____000020"]
    assert "Dropping out-of-range source_indices [3]" in caplog.text


def test_prompt_representation_accepts_string_explicit():
    pr = PromptRepresentation.model_validate({"explicit": "User works remotely"})
    assert [e.content for e in pr.explicit] == ["User works remotely"]


def test_prompt_representation_accepts_list_of_strings_explicit():
    pr = PromptRepresentation.model_validate({"explicit": ["User prefers Korean communication", "User likes coffee"]})
    assert [e.content for e in pr.explicit] == ["User prefers Korean communication", "User likes coffee"]


def test_prompt_representation_accepts_mixed_list_explicit():
    pr = PromptRepresentation.model_validate({"explicit": ["User works remotely", {"content": "User likes coffee"}]})
    assert [e.content for e in pr.explicit] == ["User works remotely", "User likes coffee"]


def test_prompt_representation_drops_blank_string_items():
    pr = PromptRepresentation.model_validate({"explicit": ["  ", "", "User works remotely", "   "]})
    assert [e.content for e in pr.explicit] == ["User works remotely"]


def test_prompt_representation_bare_blank_and_none_explicit():
    assert PromptRepresentation.model_validate({"explicit": "   "}).explicit == []
    assert PromptRepresentation.model_validate({"explicit": None}).explicit == []


def test_prompt_representation_proper_dicts_explicit_pass_through():
    pr = PromptRepresentation.model_validate({"explicit": [{"content": "A"}, {"content": "B"}]})
    assert [e.content for e in pr.explicit] == ["A", "B"]


@pytest.mark.parametrize("explicit", [123, {"content": "A"}, [123]])
def test_prompt_representation_rejects_unsupported_explicit(explicit: Any):
    with pytest.raises(ValidationError):
        PromptRepresentation.model_validate({"explicit": explicit})


def test_flatten_message_ids_handles_old_and_new_shapes():
    assert flatten_message_ids([3, 1, 3]) == [1, 3]
    assert flatten_message_ids([[1, 2], [3, 4]]) == [1, 2, 3, 4]
    assert flatten_message_ids([(105, 105)]) == [105]


def test_allowlist_safe_levels():
    assert ALLOWLIST_SAFE_LEVELS == ("explicit",)
    assert allowlist_safe_levels(None) == ["explicit"]
    assert allowlist_safe_levels(["deductive", "explicit"]) == ["explicit"]
    assert allowlist_safe_levels(["deductive"]) == []


def _conclusion(**overrides: Any) -> Conclusion:
    values: dict[str, Any] = {
        "id": "x" * 21,
        "bank_id": "b",
        "observer": "alice",
        "observed": "alice",
        "level": "explicit",
        "content": "likes tea",
        "times_derived": 1,
        "session_name": "s",
        "internal_metadata": {"message_ids": [4]},
        "created_at": NOW,
    }
    values.update(overrides)
    return Conclusion(**values)


def test_from_documents_builds_every_level_with_its_source():
    said = datetime.datetime(2024, 12, 31, 8, 0, 0, tzinfo=UTC)
    rep = Representation.from_documents(
        [
            _conclusion(source_message_ids=["m" * 21], source_url="https://slack/x/1", said_at=said),
            _conclusion(
                id="d" * 21,
                level="deductive",
                content="drinks tea",
                source_ids=["x" * 21],
                internal_metadata={"premises": ["likes tea"], "message_ids": [[4, 5]]},
            ),
            _conclusion(
                id="i" * 21,
                level="inductive",
                content="tea person",
                internal_metadata={"sources": ["likes tea"], "pattern_type": "preference", "confidence": "high"},
            ),
            _conclusion(id="c" * 21, level="contradiction", content="likes and hates tea"),
        ]
    )
    [explicit] = rep.explicit
    assert explicit.id == "x" * 21
    assert explicit.message_ids == [4]
    assert explicit.source_message_ids == ["m" * 21]
    assert explicit.source_url == "https://slack/x/1"
    assert explicit.said_at == said
    [deductive] = rep.deductive
    assert deductive.premises == ["likes tea"]
    assert deductive.source_ids == ["x" * 21]
    assert deductive.message_ids == [4, 5]
    [inductive] = rep.inductive
    assert (inductive.pattern_type, inductive.confidence) == ("preference", "high")
    assert len(rep.contradiction) == 1


def test_from_documents_uses_message_created_at_when_present():
    rep = Representation.from_documents(
        [_conclusion(internal_metadata={"message_ids": [1], "message_created_at": "2024-06-01T10:20:30Z"})]
    )
    assert rep.explicit[0].created_at == datetime.datetime(2024, 6, 1, 10, 20, 30)


def test_from_old_document_defaults_source_message_ids():
    rep = Representation.from_documents([_conclusion(source_message_ids=None)])
    assert rep.explicit[0].source_message_ids == []


# --- conclusions ---------------------------------------------------------------------------------


async def _live(ctx: PeopleContext, bank_id: str, observer: str = "alice", observed: str = "alice") -> list[Conclusion]:
    conclusions, _ = await get_all_documents(ctx, bank_id, observer=observer, observed=observed, reverse=True)
    return conclusions


async def _seed_messages(ctx: PeopleContext, bank_id: str, count: int = 3) -> list[str]:
    await add_session(ctx, bank_id, "s1", ["alice"])
    return [
        await add_message(
            ctx,
            bank_id,
            "s1",
            "alice",
            f"message {i}",
            seq=i,
            source_url=f"https://slack/x/{i}",
            said_at=NOW + datetime.timedelta(minutes=i),
        )
        for i in range(count)
    ]


async def _edges(ctx: PeopleContext, bank_id: str, conclusion_id: str, kind: str = "message") -> list[str]:
    async with ctx.conn() as db:
        rows = await db.fetch(
            f"SELECT source_id FROM {ctx.table('conclusion_sources')} "
            "WHERE bank_id = $1 AND conclusion_id = $2 AND kind = $3 ORDER BY position",
            bank_id,
            conclusion_id,
            kind,
        )
    return [r["source_id"] for r in rows]


@pytest.mark.asyncio
async def test_create_documents_writes_citation_edges_and_source(ctx: PeopleContext, bank_id: str):
    ids = await _seed_messages(ctx, bank_id)
    result = await save(ctx, bank_id, [await make_doc(ctx, "likes tea", source_message_ids=[ids[1], ids[0]])])

    [doc] = await _live(ctx, bank_id)
    assert result.created_conclusions[0].id == doc.id
    assert doc.source_message_ids == [ids[1], ids[0]]
    assert await _edges(ctx, bank_id, doc.id) == [ids[1], ids[0]]
    # the source of the first cited message rides on the conclusion
    assert doc.source_url == "https://slack/x/1"
    assert doc.said_at == NOW + datetime.timedelta(minutes=1)


@pytest.mark.asyncio
async def test_conclusion_source_comes_from_the_batch_messages_when_nothing_is_cited(ctx: PeopleContext, bank_id: str):
    await add_session(ctx, bank_id, "s1", ["alice"])
    async with ctx.conn() as db:
        await add_message(ctx, bank_id, "s1", "alice", "hello", seq=1, source_url="https://slack/x/9", said_at=NOW)
        internal_id = await db.fetchval(
            f"SELECT id FROM {ctx.table('messages')} WHERE bank_id = $1 AND seq_in_session = 1", bank_id
        )
    await save(ctx, bank_id, [await make_doc(ctx, "likes tea", message_ids=[internal_id])])
    [doc] = await _live(ctx, bank_id)
    assert (doc.source_url, doc.said_at) == ("https://slack/x/9", NOW)


@pytest.mark.asyncio
async def test_derived_conclusions_inherit_the_source_of_their_premise(ctx: PeopleContext, bank_id: str):
    ids = await _seed_messages(ctx, bank_id)
    first = await save(ctx, bank_id, [await make_doc(ctx, "likes tea", source_message_ids=[ids[2]])])
    premise = first.created_conclusions[0]
    await save(ctx, bank_id, [await make_doc(ctx, "is a tea drinker", level="deductive", source_ids=[premise.id])])

    derived = next(c for c in await _live(ctx, bank_id) if c.level == "deductive")
    assert derived.source_ids == [premise.id]
    assert derived.source_url == "https://slack/x/2"
    assert await _edges(ctx, bank_id, derived.id, kind="conclusion") == [premise.id]
    assert [c.id for c in await get_child_observations(ctx, bank_id, premise.id)] == [derived.id]


@pytest.mark.asyncio
async def test_explicit_conclusions_need_a_session(ctx: PeopleContext, bank_id: str):
    result = await save(ctx, bank_id, [await make_doc(ctx, "likes tea", session=None)])
    assert result.created_documents == []
    assert await _live(ctx, bank_id) == []


@pytest.mark.asyncio
async def test_exact_duplicates_reinforce_the_existing_conclusion(ctx: PeopleContext, bank_id: str):
    ids = await _seed_messages(ctx, bank_id)
    await save(ctx, bank_id, [await make_doc(ctx, "Likes Tea", source_message_ids=[ids[0]])])
    result = await save(ctx, bank_id, [await make_doc(ctx, "  likes tea ", source_message_ids=[ids[1], ids[0]])])

    assert result.exact_dup_existing_count == 1
    assert result.created_documents == []
    [doc] = await _live(ctx, bank_id)
    assert doc.times_derived == 2
    # Existing citations keep their positions; new ones follow.
    assert await _edges(ctx, bank_id, doc.id) == [ids[0], ids[1]]


@pytest.mark.asyncio
async def test_exact_dedup_never_crosses_sessions_for_explicit_conclusions(ctx: PeopleContext, bank_id: str):
    await save(ctx, bank_id, [await make_doc(ctx, "likes tea", session="s1")])
    result = await save(ctx, bank_id, [await make_doc(ctx, "likes tea", session="s2")])
    assert result.exact_dup_existing_count == 0
    assert len(await _live(ctx, bank_id)) == 2


@pytest.mark.asyncio
async def test_exact_dedup_never_crosses_levels(ctx: PeopleContext, bank_id: str):
    await save(ctx, bank_id, [await make_doc(ctx, "likes tea")])
    result = await save(ctx, bank_id, [await make_doc(ctx, "likes tea", level="deductive", source_ids=["a" * 21])])
    assert result.exact_dup_existing_count == 0
    assert {c.level for c in await _live(ctx, bank_id)} == {"explicit", "deductive"}


@pytest.mark.asyncio
async def test_in_batch_duplicates_merge_their_citations(ctx: PeopleContext, bank_id: str):
    ids = await _seed_messages(ctx, bank_id)
    result = await save(
        ctx,
        bank_id,
        [
            await make_doc(ctx, "likes tea", source_message_ids=[ids[1]]),
            await make_doc(ctx, "likes tea", source_message_ids=[ids[0], ids[1]]),
        ],
    )
    assert result.exact_dup_in_batch_count == 1
    [doc] = await _live(ctx, bank_id)
    assert await _edges(ctx, bank_id, doc.id) == [ids[1], ids[0]]


@pytest.mark.asyncio
async def test_malformed_citations_are_dropped(ctx: PeopleContext, bank_id: str):
    ids = await _seed_messages(ctx, bank_id)
    await save(
        ctx,
        bank_id,
        [
            await make_doc(ctx, "likes tea", source_message_ids=[ids[0]]),
            await make_doc(ctx, "likes tea", source_message_ids=["not-a-message-id", ids[1]]),
        ],
    )
    [doc] = await _live(ctx, bank_id)
    assert await _edges(ctx, bank_id, doc.id) == [ids[0], ids[1]]


@pytest.mark.asyncio
async def test_in_batch_duplicates_of_an_existing_conclusion_merge_into_it(ctx: PeopleContext, bank_id: str):
    ids = await _seed_messages(ctx, bank_id)
    await save(ctx, bank_id, [await make_doc(ctx, "likes tea", source_message_ids=[ids[0]])])
    result = await save(
        ctx,
        bank_id,
        [
            await make_doc(ctx, "likes tea", source_message_ids=[ids[0]]),
            await make_doc(ctx, "likes tea", source_message_ids=[ids[1]]),
        ],
    )
    assert result.exact_dup_existing_count == 1
    assert result.exact_dup_in_batch_count == 1
    [doc] = await _live(ctx, bank_id)
    assert await _edges(ctx, bank_id, doc.id) == [ids[0], ids[1]]


async def _same_vector(ctx: PeopleContext) -> list[float]:
    [vector] = await ctx.embed(["a shared meaning"])
    return vector


@pytest.mark.asyncio
async def test_semantic_rejection_merges_citations_into_the_kept_conclusion(ctx: PeopleContext, bank_id: str):
    ids = await _seed_messages(ctx, bank_id)
    vector = await _same_vector(ctx)
    await save(
        ctx,
        bank_id,
        [await make_doc(ctx, "Ann likes coffee and tea", source_message_ids=[ids[0]], embedding=vector)],
    )
    result = await save(
        ctx,
        bank_id,
        [await make_doc(ctx, "Ann likes coffee", source_message_ids=[ids[1]], embedding=vector)],
        deduplicate=True,
    )

    assert result.semantic_dup_rejected_count == 1
    [doc] = await _live(ctx, bank_id)
    assert doc.content == "Ann likes coffee and tea"
    assert doc.times_derived == 2
    assert await _edges(ctx, bank_id, doc.id) == [ids[0], ids[1]]


@pytest.mark.asyncio
async def test_semantic_replacement_inherits_the_replaced_citations(ctx: PeopleContext, bank_id: str):
    ids = await _seed_messages(ctx, bank_id)
    vector = await _same_vector(ctx)
    first = await save(
        ctx, bank_id, [await make_doc(ctx, "Ann likes coffee", source_message_ids=[ids[0]], embedding=vector)]
    )
    result = await save(
        ctx,
        bank_id,
        [await make_doc(ctx, "Ann likes coffee and tea", source_message_ids=[ids[1]], embedding=vector)],
        deduplicate=True,
    )

    assert result.semantic_dup_replaced_count == 1
    [doc] = await _live(ctx, bank_id)
    assert doc.content == "Ann likes coffee and tea"
    assert doc.times_derived == 2
    assert await _edges(ctx, bank_id, doc.id) == [ids[0], ids[1]]
    # the replaced row is reported with the id of the row that replaced it
    [(old, replacement_id)] = result.replaced_conclusions
    assert old.id == first.created_conclusions[0].id
    assert replacement_id == doc.id


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement_first", [False, True])
async def test_replacement_keeps_citations_from_a_reinforce_of_the_same_row(
    ctx: PeopleContext, bank_id: str, replacement_first: bool
):
    ids = await _seed_messages(ctx, bank_id)
    vector = await _same_vector(ctx)
    await save(ctx, bank_id, [await make_doc(ctx, "Ann likes coffee", source_message_ids=[ids[0]], embedding=vector)])
    # One batch both reinforces the existing row (exact match) and replaces it (a richer semantic
    # match), in either order.
    reinforce = await make_doc(ctx, "Ann likes coffee", source_message_ids=[ids[1]], embedding=vector)
    replace = await make_doc(ctx, "Ann likes coffee and tea", embedding=vector)
    batch = [replace, reinforce] if replacement_first else [reinforce, replace]
    result = await save(ctx, bank_id, batch, deduplicate=True)

    assert result.semantic_dup_replaced_count == 1
    assert result.exact_dup_existing_count == 1
    [doc] = await _live(ctx, bank_id)
    assert doc.content == "Ann likes coffee and tea"
    assert await _edges(ctx, bank_id, doc.id) == [ids[0], ids[1]]


@pytest.mark.asyncio
async def test_semantic_dedup_stays_inside_the_level_and_session(ctx: PeopleContext, bank_id: str):
    vector = await _same_vector(ctx)
    await save(ctx, bank_id, [await make_doc(ctx, "likes coffee", session="s1", embedding=vector)])
    result = await save(
        ctx, bank_id, [await make_doc(ctx, "enjoys coffee", session="s2", embedding=vector)], deduplicate=True
    )
    assert result.semantic_dup_rejected_count == 0 and result.semantic_dup_replaced_count == 0
    assert len(await _live(ctx, bank_id)) == 2


@pytest.mark.asyncio
async def test_delete_documents_is_soft_and_scoped(ctx: PeopleContext, bank_id: str):
    result = await save(
        ctx, bank_id, [await make_doc(ctx, "likes tea"), await make_doc(ctx, "likes jam", session="s2")]
    )
    tea, jam = (c.id for c in result.created_conclusions)

    # wrong session or pair deletes nothing
    assert await delete_documents(ctx, bank_id, [tea], observer="alice", observed="alice", session_name="s2") == []
    assert await delete_documents(ctx, bank_id, [tea], observer="bob", observed="alice") == []

    assert await delete_documents(ctx, bank_id, [tea, "missing"], observer="alice", observed="alice") == [
        (tea, "explicit")
    ]
    assert [c.id for c in await _live(ctx, bank_id)] == [jam]
    # already deleted: deleting again finds nothing
    with pytest.raises(ResourceNotFoundException):
        await delete_document(ctx, bank_id, tea, observer="alice", observed="alice")
    await delete_document(ctx, bank_id, jam, observer="alice", observed="alice", session_name="s2")
    with pytest.raises(ResourceNotFoundException):
        await delete_document_by_id(ctx, bank_id, jam)
    assert await get_documents_by_ids(ctx, bank_id, [tea, jam]) == []


@pytest.mark.asyncio
async def test_delete_documents_can_join_a_callers_transaction(ctx: PeopleContext, bank_id: str):
    result = await save(ctx, bank_id, [await make_doc(ctx, "likes tea")])
    [created] = result.created_conclusions
    async with ctx.conn() as db:
        with pytest.raises(RuntimeError):
            async with db.transaction():
                await delete_documents(ctx, bank_id, [created.id], observer="alice", observed="alice", conn=db)
                raise RuntimeError("roll back")
    assert [c.id for c in await _live(ctx, bank_id)] == [created.id]


@pytest.mark.asyncio
async def test_query_helpers_order_and_filter(ctx: PeopleContext, bank_id: str):
    await save(ctx, bank_id, [await make_doc(ctx, "first")])
    await save(ctx, bank_id, [await make_doc(ctx, "second")])
    await save(ctx, bank_id, [await make_doc(ctx, "third", session="s2")])
    await save(ctx, bank_id, [await make_doc(ctx, "second")])  # reinforces "second"

    recent = await query_documents_recent(ctx, bank_id, observer="alice", observed="alice", limit=2)
    assert [c.content for c in recent] == ["third", "second"]
    in_session = await query_documents_recent(ctx, bank_id, observer="alice", observed="alice", session_name="s1")
    assert [c.content for c in in_session] == ["second", "first"]
    most_derived = await query_documents_most_derived(ctx, bank_id, observer="alice", observed="alice", limit=1)
    assert [c.content for c in most_derived] == ["second"]

    page, total = await get_all_documents(ctx, bank_id, observer="alice", observed="alice", limit=2, offset=1)
    assert total == 3 and len(page) == 2
    only_s2, total = await get_all_documents(
        ctx, bank_id, observer="alice", observed="alice", filters={"session_name": "s2"}
    )
    assert [c.content for c in only_s2] == ["third"] and total == 1
    with pytest.raises(ValidationException):
        await get_all_documents(ctx, bank_id, observer="alice", observed="alice", filters={"content": "x"})

    rows, total = await query_explicit_documents_after(
        ctx, bank_id, observer="alice", observed="alice", after=None, since=None, limit=2
    )
    assert [c.content for c in rows] == ["first", "second"] and total == 3
    rows, total = await query_explicit_documents_after(
        ctx, bank_id, observer="alice", observed="alice", after=(rows[-1].created_at, rows[-1].id), since=None, limit=5
    )
    assert [c.content for c in rows] == ["third"] and total == 1


@pytest.mark.asyncio
async def test_query_documents_is_a_cosine_search_within_the_pair(ctx: PeopleContext, bank_id: str):
    await save(ctx, bank_id, [await make_doc(ctx, "likes tea"), await make_doc(ctx, "likes jam")])
    await save(ctx, bank_id, [await make_doc(ctx, "likes tea")], observer="bob", observed="alice")

    found = await query_documents(ctx, bank_id, "likes tea", observer="alice", observed="alice", top_k=1)
    assert [c.content for c in found] == ["likes tea"]
    both = await query_documents(ctx, bank_id, "likes tea", observer="alice", observed="alice", top_k=5)
    assert len(both) == 2
    near = await query_documents(
        ctx, bank_id, "likes tea", observer="alice", observed="alice", top_k=5, max_distance=0.001
    )
    assert [c.content for c in near] == ["likes tea"]
    assert await query_documents(ctx, bank_id, "likes tea", observer="alice", observed="alice", top_k=0) == []
    nothing = await query_documents(
        ctx, bank_id, "likes tea", observer="alice", observed="alice", filters={"session_name": {"in": []}}
    )
    assert nothing == []


# --- RepresentationManager -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recent_excludes_soft_deleted_and_selected_conclusions(ctx: PeopleContext, bank_id: str):
    a = (await save(ctx, bank_id, [await make_doc(ctx, "one")])).created_conclusions[0]
    b = (await save(ctx, bank_id, [await make_doc(ctx, "two")])).created_conclusions[0]
    c = (await save(ctx, bank_id, [await make_doc(ctx, "three")])).created_conclusions[0]
    await delete_documents(ctx, bank_id, [b.id], observer="alice", observed="alice")

    manager = RepresentationManager(ctx, bank_id, observer="alice", observed="alice")
    recent = await manager._query_documents_recent(top_k=10)
    assert [x.content for x in recent] == ["three", "one"]
    recent = await manager._query_documents_recent(top_k=10, excluded_document_ids={c.id})
    assert [x.id for x in recent] == [a.id]
    assert await manager._query_documents_recent(top_k=0) == []


@pytest.mark.asyncio
async def test_most_derived_ties_break_by_recency_and_skip_deleted(ctx: PeopleContext, bank_id: str):
    await save(ctx, bank_id, [await make_doc(ctx, "old")])
    await save(ctx, bank_id, [await make_doc(ctx, "newer")])
    popular = (await save(ctx, bank_id, [await make_doc(ctx, "popular")])).created_conclusions[0]
    await save(ctx, bank_id, [await make_doc(ctx, "popular")])
    await save(ctx, bank_id, [await make_doc(ctx, "popular")])
    gone = (await save(ctx, bank_id, [await make_doc(ctx, "gone", times_derived=9)])).created_conclusions[0]
    await delete_documents(ctx, bank_id, [gone.id], observer="alice", observed="alice")

    manager = RepresentationManager(ctx, bank_id, observer="alice", observed="alice")
    derived = await manager._query_documents_most_derived(top_k=3)
    assert [x.content for x in derived] == ["popular", "newer", "old"]
    assert derived[0].id == popular.id and derived[0].times_derived == 3


async def _seed_two_sessions(ctx: PeopleContext, bank_id: str) -> None:
    await save(ctx, bank_id, [await make_doc(ctx, "s1 explicit", session="s1")])
    await save(ctx, bank_id, [await make_doc(ctx, "s2 explicit", session="s2")])
    await save(
        ctx, bank_id, [await make_doc(ctx, "s1 deduced", session="s1", level="deductive", source_ids=["a" * 21])]
    )


@pytest.mark.asyncio
async def test_session_allowlist_scopes_every_query_path(ctx: PeopleContext, bank_id: str):
    await _seed_two_sessions(ctx, bank_id)
    manager = RepresentationManager(ctx, bank_id, observer="alice", observed="alice")

    # only explicit conclusions carry a trustworthy session stamp, so only they are served
    assert [c.content for c in await manager._query_documents_recent(top_k=10, session_allowlist=["s1"])] == [
        "s1 explicit"
    ]
    assert [c.content for c in await manager._query_documents_most_derived(top_k=10, session_allowlist=["s1"])] == [
        "s1 explicit"
    ]
    semantic = await manager.query_documents("s1 explicit", top_k=10, session_allowlist=["s1", "s2"])
    assert sorted(c.content for c in semantic) == ["s1 explicit", "s2 explicit"]
    by_level = await manager.query_documents("s1 deduced", top_k=10, level="deductive", session_allowlist=["s1"])
    assert by_level == []

    rep = await manager.get_working_representation(session_allowlist=["s2"], include_most_derived=True)
    assert [o.content for o in rep.explicit] == ["s2 explicit"]
    assert rep.deductive == []


@pytest.mark.asyncio
async def test_empty_allowlist_fails_closed(ctx: PeopleContext, bank_id: str):
    await _seed_two_sessions(ctx, bank_id)
    manager = RepresentationManager(ctx, bank_id, observer="alice", observed="alice")
    assert (await manager.get_working_representation(session_allowlist=[])).is_empty()
    assert (await manager.get_working_representation(session_allowlist=[], include_semantic_query="x")).is_empty()
    # without an allowlist everything is served
    assert (await manager.get_working_representation()).len() == 3


def test_build_filter_conditions_empty_allowlist_fails_closed():
    manager = RepresentationManager(None, "b", observer="a", observed="a")  # type: ignore[arg-type]
    assert manager._build_filter_conditions(session_allowlist=[]) == {
        "session_name": {"in": []},
        "level": {"in": ["explicit"]},
    }
    # an unscopeable level narrows to nothing rather than widening
    assert manager._build_filter_conditions("deductive", session_allowlist=["s"])["level"] == {"in": []}
    assert manager._build_filter_conditions("deductive") == {"level": "deductive"}
    assert manager._build_filter_conditions() == {}


@pytest.mark.asyncio
async def test_working_representation_blends_semantic_derived_and_recent(ctx: PeopleContext, bank_id: str):
    for text in ["likes tea", "likes jam", "likes ham"]:
        await save(ctx, bank_id, [await make_doc(ctx, text)])
    await save(ctx, bank_id, [await make_doc(ctx, "likes jam")])  # most derived

    rep = await get_working_representation(
        ctx,
        bank_id,
        observer="alice",
        observed="alice",
        include_semantic_query="likes tea",
        semantic_search_top_k=1,
        semantic_search_max_distance=0.001,
        include_most_derived=True,
        max_observations=3,
    )
    assert sorted(o.content for o in rep.explicit) == ["likes ham", "likes jam", "likes tea"]

    capped = await get_working_representation(ctx, bank_id, observer="alice", observed="alice", max_observations=2)
    assert [o.content for o in capped.explicit] == ["likes jam", "likes ham"]


@pytest.mark.asyncio
async def test_requested_semantic_search_always_gets_budget(ctx: PeopleContext, bank_id: str, monkeypatch):
    """max_observations < 3 must not allocate 0 to an explicitly requested search."""
    seen: list[int] = []

    async def fake_query(self: RepresentationManager, **kwargs: Any) -> list[Conclusion]:
        seen.append(kwargs["top_k"])
        return []

    monkeypatch.setattr(RepresentationManager, "query_documents", fake_query)
    manager = RepresentationManager(ctx, bank_id, observer="alice", observed="alice")
    for max_observations in (1, 2, 100):
        await manager._get_working_representation_internal(
            include_semantic_query="what do they like?", embedding=[0.1], max_observations=max_observations
        )
    assert all(1 <= top_k <= m for top_k, m in zip(seen, (1, 2, 100), strict=True))


@pytest.mark.asyncio
async def test_unused_semantic_budget_goes_to_recent_observations(ctx: PeopleContext, bank_id: str, monkeypatch):
    semantic = _conclusion(id="s" * 21, content="semantic result", created_at=NOW)
    recent = _conclusion(id="r" * 21, content="recent result", created_at=NOW + datetime.timedelta(days=1))
    recent_calls: list[dict[str, Any]] = []

    async def fake_semantic(self: RepresentationManager, **kwargs: Any) -> list[Conclusion]:
        return [semantic]

    async def fake_recent(self: RepresentationManager, **kwargs: Any) -> list[Conclusion]:
        recent_calls.append(kwargs)
        return [recent]

    monkeypatch.setattr(RepresentationManager, "query_documents", fake_semantic)
    monkeypatch.setattr(RepresentationManager, "_query_documents_recent", fake_recent)
    manager = RepresentationManager(ctx, bank_id, observer="alice", observed="alice")
    rep = await manager._get_working_representation_internal(
        include_semantic_query="what do they like?", embedding=[0.1], semantic_search_top_k=5, max_observations=5
    )
    assert recent_calls[0]["top_k"] == 4
    assert recent_calls[0]["excluded_document_ids"] == {"s" * 21}
    assert [o.content for o in rep.explicit] == ["semantic result", "recent result"]


@pytest.mark.asyncio
async def test_query_failure_degrades_to_no_semantic_results(ctx: PeopleContext, bank_id: str, monkeypatch):
    await save(ctx, bank_id, [await make_doc(ctx, "likes tea")])

    async def boom(*args: Any, **kwargs: Any) -> list[Conclusion]:
        raise RuntimeError("the index is down")

    monkeypatch.setattr("hindsight_api.people.representation.query_documents", boom)
    manager = RepresentationManager(ctx, bank_id, observer="alice", observed="alice")
    assert await manager.query_documents("likes tea", top_k=3) == []
    rep = await manager.get_working_representation(include_semantic_query="likes tea")
    assert [o.content for o in rep.explicit] == ["likes tea"]


@pytest.mark.asyncio
async def test_query_embedding_failure_degrades_to_recent(ctx: PeopleContext, bank_id: str, monkeypatch):
    await save(ctx, bank_id, [await make_doc(ctx, "likes tea")])

    async def boom(text: str) -> list[float]:
        raise RuntimeError("embedding provider down")

    monkeypatch.setattr(PeopleContext, "embed_query", lambda self, text: boom(text))
    manager = RepresentationManager(ctx, bank_id, observer="alice", observed="alice")
    rep = await manager.get_working_representation(include_semantic_query="likes tea")
    assert [o.content for o in rep.explicit] == ["likes tea"]


def _observation(content: str, **kwargs: Any) -> ExplicitObservation:
    return ExplicitObservation(
        content=content, created_at=NOW, message_ids=kwargs.pop("message_ids", [1]), session_name="s1", **kwargs
    )


@pytest.mark.asyncio
async def test_save_representation_threads_source_message_ids_and_source(ctx: PeopleContext, bank_id: str):
    ids = await _seed_messages(ctx, bank_id)
    manager = RepresentationManager(ctx, bank_id, observer="alice", observed="alice")
    representation = Representation(
        explicit=[_observation("Alice chose the first option", source_message_ids=[ids[0], ids[2]])]
    )

    result = await manager.save_representation(
        representation, message_ids=[1], session_name="s1", message_created_at=NOW
    )

    [doc] = await _live(ctx, bank_id)
    assert len(result.created_documents) == 1
    assert result.created_documents[0].metadata.message_ids == [1]
    assert "source_message_ids" not in result.created_documents[0].metadata.model_dump()
    assert doc.source_message_ids == [ids[0], ids[2]]
    assert (doc.source_url, doc.said_at) == ("https://slack/x/0", NOW)
    assert doc.internal_metadata["message_created_at"] == "2025-01-01T12:00:00Z"
    assert doc.internal_metadata["message_ids"] == [1]


@pytest.mark.asyncio
async def test_save_representation_filters_blank_observations_before_embedding(
    ctx: PeopleContext, bank_id: str, monkeypatch
):
    embedded: list[list[str]] = []
    real_embed = PeopleContext.embed

    async def spy(self: PeopleContext, texts: list[str]) -> list[list[float]]:
        embedded.append(list(texts))
        return await real_embed(self, texts)

    monkeypatch.setattr(PeopleContext, "embed", spy)
    manager = RepresentationManager(ctx, bank_id, observer="alice", observed="alice")
    representation = Representation(
        explicit=[_observation("   "), _observation(" useful observation ")],
        deductive=[
            DeductiveObservation(conclusion="  ", premises=["x"], created_at=NOW, message_ids=[1], session_name="s1")
        ],
    )
    await get_or_create_peers_for(ctx, bank_id, "alice")
    saved = await manager.save_representation(
        representation, message_ids=[1], session_name="s1", message_created_at=NOW
    )

    assert embedded == [["useful observation"]]
    assert [d.content for d in saved.created_documents] == ["useful observation"]


async def get_or_create_peers_for(ctx: PeopleContext, bank_id: str, *names: str) -> None:
    from hindsight_api.people.peers import get_or_create_peers

    await get_or_create_peers(ctx, bank_id, list(names))


@pytest.mark.asyncio
async def test_save_representation_skips_when_every_observation_is_blank(ctx: PeopleContext, bank_id: str, monkeypatch):
    async def fail(self: PeopleContext, texts: list[str]) -> list[list[float]]:
        raise AssertionError("nothing should be embedded")

    monkeypatch.setattr(PeopleContext, "embed", fail)
    manager = RepresentationManager(ctx, bank_id, observer="alice", observed="alice")
    for representation in (
        Representation(),
        Representation(explicit=[_observation("   "), _observation("\x00")]),
    ):
        saved = await manager.save_representation(
            representation, message_ids=[1], session_name="s1", message_created_at=NOW
        )
        assert saved.created_documents == []


@pytest.mark.asyncio
async def test_save_representation_strips_nul_bytes(ctx: PeopleContext, bank_id: str):
    await get_or_create_peers_for(ctx, bank_id, "alice")
    manager = RepresentationManager(ctx, bank_id, observer="alice", observed="alice")
    representation = Representation(
        explicit=[_observation("likes\x00 tea")],
        deductive=[
            DeductiveObservation(
                conclusion="drinks\x00 tea",
                premises=["likes\x00 tea"],
                source_ids=["a" * 21],
                created_at=NOW,
                message_ids=[1],
                session_name="s1",
            )
        ],
    )
    await manager.save_representation(representation, message_ids=[1], session_name="s1", message_created_at=NOW)

    by_level = {c.level: c for c in await _live(ctx, bank_id)}
    assert by_level["explicit"].content == "likes tea"
    assert by_level["deductive"].content == "drinks tea"
    assert by_level["deductive"].internal_metadata["premises"] == ["likes tea"]


@pytest.mark.asyncio
async def test_save_representation_dedups_a_repeated_fact(ctx: PeopleContext, bank_id: str):
    await get_or_create_peers_for(ctx, bank_id, "alice")
    manager = RepresentationManager(ctx, bank_id, observer="alice", observed="alice")
    representation = Representation(explicit=[_observation("likes tea")])
    await manager.save_representation(representation, message_ids=[1], session_name="s1", message_created_at=NOW)
    second = await manager.save_representation(
        representation, message_ids=[2], session_name="s1", message_created_at=NOW
    )
    assert second.exact_dup_existing_count == 1
    [doc] = await _live(ctx, bank_id)
    assert doc.times_derived == 2


@pytest.mark.asyncio
async def test_save_representation_schedules_a_dream_only_when_enabled(ctx: PeopleContext, bank_id: str, monkeypatch):
    calls: list[tuple[Any, ...]] = []

    async def check_and_schedule_dream(*args: Any) -> bool:
        calls.append(args)
        return False

    fake = types.ModuleType("hindsight_api.people.dreamer.dream_scheduler")
    fake.check_and_schedule_dream = check_and_schedule_dream  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hindsight_api.people.dreamer.dream_scheduler", fake)

    await get_or_create_peers_for(ctx, bank_id, "alice")
    manager = RepresentationManager(ctx, bank_id, observer="alice", observed="alice")
    representation = Representation(explicit=[_observation("likes tea")])

    await manager.save_representation(representation, message_ids=[1], session_name="s1", message_created_at=NOW)
    assert calls == [(ctx, bank_id, "alice", "alice")]

    disabled = schemas.get_configuration(None, {"dream": {"enabled": False}})
    await manager.save_representation(
        Representation(explicit=[_observation("likes jam")]),
        message_ids=[2],
        session_name="s1",
        message_created_at=NOW,
        message_level_configuration=disabled,
    )
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_a_failing_dream_check_does_not_fail_the_save(ctx: PeopleContext, bank_id: str, monkeypatch):
    async def check_and_schedule_dream(*args: Any) -> bool:
        raise RuntimeError("scheduler down")

    fake = types.ModuleType("hindsight_api.people.dreamer.dream_scheduler")
    fake.check_and_schedule_dream = check_and_schedule_dream  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hindsight_api.people.dreamer.dream_scheduler", fake)
    await get_or_create_peers_for(ctx, bank_id, "alice")
    manager = RepresentationManager(ctx, bank_id, observer="alice", observed="alice")
    saved = await manager.save_representation(
        Representation(explicit=[_observation("likes tea")]), message_ids=[1], session_name="s1", message_created_at=NOW
    )
    assert len(saved.created_documents) == 1


@pytest.mark.asyncio
async def test_conclusions_of_a_bank_do_not_leak_into_another(ctx: PeopleContext, bank_id: str, people_engine):
    other = bank_id + "-other"
    await people_engine.ensure_bank_profile(bank_id=other, request_context=ctx.request_context)
    await save(ctx, bank_id, [await make_doc(ctx, "likes tea")])
    assert await _live(ctx, other) == []
    assert (await get_working_representation(ctx, other, observer="alice", observed="alice")).is_empty()
