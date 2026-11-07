"""Source metadata and pointer fields (RUN-2355 6 and contract 7)."""

from datetime import datetime, timezone

import pytest

from hindsight_api.api.http import MemoryItem
from hindsight_api.engine.retain.fact_extraction import _build_user_message
from hindsight_api.engine.retain.source_metadata import missing_source_fields

SOURCE = {
    "source_url": "https://discord.test/channels/1/2/3",
    "said_by": "asha",
    "said_at": "2026-11-03T09:10:00+00:00",
    "channel": "discord",
}


# ----------------------------------------------------------------------------- source metadata


def test_named_source_fields_travel_in_metadata_and_set_the_time():
    item = MemoryItem(content="I will send the contract tomorrow", **SOURCE)

    assert item.metadata == SOURCE
    assert item.timestamp == datetime(2026, 11, 3, 9, 10, tzinfo=timezone.utc)


def test_an_explicit_timestamp_wins_over_said_at():
    item = MemoryItem(content="x", timestamp="2026-11-04T00:00:00Z", **SOURCE)

    assert item.timestamp == datetime(2026, 11, 4, tzinfo=timezone.utc)
    assert item.metadata is not None and item.metadata["said_at"] == SOURCE["said_at"]


def test_existing_metadata_is_kept_beside_the_source_fields():
    item = MemoryItem(content="x", metadata={"topic": "pricing"}, **SOURCE)

    assert item.metadata == {"topic": "pricing", **SOURCE}


def test_a_source_field_that_disagrees_with_metadata_is_refused():
    with pytest.raises(ValueError, match="channel is given twice"):
        MemoryItem(content="x", metadata={"channel": "slack"}, **SOURCE)


def test_an_item_without_source_fields_is_unchanged():
    item = MemoryItem(content="x")

    assert item.metadata is None
    assert item.timestamp is None
    assert missing_source_fields(item.metadata) == ["source_url", "said_by", "said_at", "channel"]


def test_missing_source_fields_lists_only_what_is_absent():
    assert missing_source_fields({"source_url": "u", "said_by": "a"}) == ["said_at", "channel"]


def test_a_bad_said_at_is_refused():
    with pytest.raises(ValueError, match="said_at"):
        MemoryItem(content="x", said_at="yesterday-ish")


# ----------------------------------------------------------------------------- pointers


@pytest.mark.parametrize("kind", ["crm", "linear", "stripe"])
def test_a_pointer_carries_its_kind_and_id_in_metadata(kind):
    item = MemoryItem(content="the deal moved", pointer_kind=kind, pointer_id="rec_1")

    assert item.metadata == {"pointer_kind": kind, "pointer_id": "rec_1"}


def test_a_pointer_needs_an_id():
    with pytest.raises(ValueError, match="pointer_id is required"):
        MemoryItem(content="x", pointer_kind="crm")


def test_an_id_needs_a_pointer_kind():
    with pytest.raises(ValueError, match="pointer_kind must be one of"):
        MemoryItem(content="x", metadata={"pointer_id": "rec_1"})


def test_an_unknown_pointer_kind_is_refused():
    with pytest.raises(ValueError):
        MemoryItem(content="x", pointer_kind="jira", pointer_id="X-1")


def _message(metadata):
    return _build_user_message(
        chunk="Asha: the Acme deal is in Negotiation now",
        chunk_index=0,
        total_chunks=1,
        event_date=None,
        context="sales channel",
        metadata=metadata,
    )


def test_the_extraction_prompt_names_the_speaker_and_forbids_copying_a_pointed_value():
    message = _message({**SOURCE, "pointer_kind": "crm", "pointer_id": "deal_42"})

    assert "SPEAKER: asha said this on discord" in message
    assert "POINTER: this content mentions a record that crm owns (id deal_42)" in message
    assert "Do NOT store such a value as a fact" in message


def test_the_extraction_prompt_has_no_pointer_rule_without_a_pointer():
    message = _message(SOURCE)

    assert "SPEAKER:" in message
    assert "POINTER:" not in message
