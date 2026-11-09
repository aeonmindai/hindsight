"""Holding consolidation back is a deadline, never an off switch: it ends by itself, a broken file holds nothing,
and only consolidation is held (retain, extraction and refreshes are claimed as always)."""

import json
import time
from datetime import datetime, timedelta, timezone

from hindsight_api.worker.poller import WorkerPoller


def _poller(hold_file: str | None, min_gap: float = 0.0) -> WorkerPoller:
    poller = WorkerPoller.__new__(WorkerPoller)
    poller._consolidation_hold_file = hold_file
    poller._consolidation_min_gap_seconds = min_gap
    poller._last_consolidation_end = float("-inf")
    return poller


def _write(path, until: datetime) -> str:
    path.write_text(json.dumps({"until": until.isoformat()}))
    return str(path)


def test_a_hold_in_the_future_holds_and_one_in_the_past_does_not(tmp_path):
    now = datetime.now(timezone.utc)
    assert "held until" in _poller(_write(tmp_path / "h.json", now + timedelta(minutes=5)))._consolidation_hold_reason()
    assert _poller(_write(tmp_path / "old.json", now - timedelta(seconds=1)))._consolidation_hold_reason() is None


def test_a_missing_or_broken_file_holds_nothing(tmp_path):
    assert _poller(str(tmp_path / "missing.json"))._consolidation_hold_reason() is None
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    assert _poller(str(bad))._consolidation_hold_reason() is None
    no_until = tmp_path / "no-until.json"
    no_until.write_text("{}")
    assert _poller(str(no_until))._consolidation_hold_reason() is None
    assert _poller(None)._consolidation_hold_reason() is None


def test_the_gap_after_a_consolidation_holds_only_until_it_has_passed():
    poller = _poller(None, min_gap=60)
    assert poller._consolidation_hold_reason() is None
    poller._last_consolidation_end = time.monotonic()
    assert "gap" in poller._consolidation_hold_reason()
    poller._last_consolidation_end = time.monotonic() - 61
    assert poller._consolidation_hold_reason() is None
