"""Deriver settings as module constants.

Honcho reads these from ``settings.DERIVER`` (config.toml and environment). The fork has no such
settings object: the values below are Honcho's defaults, kept as one mutable dataclass instance so a
test or a caller can change one in place (``DERIVER.FLUSH_ENABLED = True``). Honcho's settings that
belong to what is not copied (tenancy, the reconciler, metrics, the model transport) are dropped:
every deriver call goes through the engine's Claude-configured background provider
(``ctx.llm_background``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DeriverSettings:
    # Concurrency within one deriver process.
    WORKERS: int = 1

    POLLING_SLEEP_INTERVAL_SECONDS: float = 1.0
    # Adaptive polling: when the queue is idle (or the loop is erroring) the sleep interval grows from
    # POLLING_SLEEP_INTERVAL_SECONDS toward POLLING_SLEEP_MAX_INTERVAL_SECONDS by POLLING_BACKOFF_MULTIPLIER each
    # cycle, then snaps back to the base interval as soon as work is found.
    POLLING_BACKOFF_ENABLED: bool = True
    POLLING_SLEEP_MAX_INTERVAL_SECONDS: float = 30.0
    POLLING_BACKOFF_MULTIPLIER: float = 2.0
    # Sleep a uniform-random delay in [0, POLLING_STARTUP_JITTER_SECONDS] before the first poll so
    # instances that start together do not poll in lockstep. 0.0 disables it.
    POLLING_STARTUP_JITTER_SECONDS: float = 30.0
    # Multiply every poll sleep by a random factor in [1 - ratio, 1 + ratio]. 0.0 disables it.
    POLLING_JITTER_RATIO: float = 0.5
    STALE_SESSION_TIMEOUT_MINUTES: int = 5
    # Minimum (jittered) spacing between stale-work-unit cleanup runs.
    STALE_WORK_UNIT_CLEANUP_INTERVAL_SECONDS: float = 60.0

    # Output cap of one deriver call, in tokens.
    MAX_OUTPUT_TOKENS: int = 4096

    # Minimum tokens a representation work unit must accumulate (summed over its own unprocessed
    # messages) before it becomes claimable. Bypassed by FLUSH_ENABLED and by age flushing. 0 disables
    # the accumulation gate (work units are claimable as soon as anything is pending).
    REPRESENTATION_BATCH_WORK_UNIT_TARGET_TOKENS: int = 512
    # Cumulative-token cap on the conversation window (queued messages plus interleaved context) fed
    # to a single deriver call when draining a claimed work unit. The first unprocessed message is
    # always included, even if it alone exceeds the cap.
    REPRESENTATION_BATCH_TARGET_INPUT_TOKENS: int = 1024
    # Sub-threshold work units become eligible once their oldest unprocessed item exceeds this age.
    # 0 disables age-based flushing.
    REPRESENTATION_BATCH_MAX_AGE_SECONDS: int = 1800
    # Bypasses the batch token threshold and processes work immediately.
    FLUSH_ENABLED: bool = False


DERIVER = DeriverSettings()
