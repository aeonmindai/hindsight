# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/dreamer/__init__.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""The dreamer: consolidation of what is known about a person, run as a dream over an (observer, observed) pair.

Entry points live in the submodules, so importing this package stays cheap and cycle-free:

* ``dream_scheduler.DreamScheduler`` (``start``/``stop``), ``check_and_schedule_dream``
* ``orchestrator.run_dream``, ``orchestrator.process_dream``
* ``dream_due.list_due_dreams``
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "get_dream_scheduler",
    "check_and_schedule_dream",
    "process_dream",
]


def __getattr__(name: str) -> Any:
    if name in ("get_dream_scheduler", "check_and_schedule_dream"):
        from . import dream_scheduler

        return getattr(dream_scheduler, name)
    if name == "process_dream":
        from .orchestrator import process_dream

        return process_dream
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
