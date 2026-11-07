"""People deriver queue: the work queue and the work-unit claims.

Revision ID: c7b3e5a9f124
Revises: b6a2d4f8e013
Create Date: 2026-11-07

The queue behind the people deriver (ported from Honcho's ``queue`` and ``active_queue_sessions``
tables). ``people_queue`` holds one row per task: a ``representation`` task for each retained
message that needs conclusions drawn, and a ``dream`` task for each scheduled dream. Rows that share
a ``work_unit_key`` are one work unit; a worker takes a work unit by inserting a row in
``people_active_queue_sessions`` (unique per ``work_unit_key``), so two workers never drain the
same unit at once. A claim that is not refreshed for a few minutes is stale and is deleted, which
frees its work unit for another worker.

Honcho keeps a third table, ``queue_item_batches``, that triggers keep in step with the queue so the
claim does not re-aggregate it. The people queue is small, so the claim aggregates ``people_queue``
(joined to ``people_messages`` for token counts) directly and no trigger or batch table exists.

PG only, like the other people migrations: Oracle does not run the people features.
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "c7b3e5a9f124"
down_revision: str | Sequence[str] | None = "b6a2d4f8e013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    s = _pg_schema_prefix()

    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {s}people_queue (
            bank_id TEXT NOT NULL,
            id BIGINT GENERATED ALWAYS AS IDENTITY,
            session_id TEXT,
            work_unit_key TEXT NOT NULL,
            task_type TEXT NOT NULL,
            payload JSONB NOT NULL,
            processed BOOLEAN NOT NULL DEFAULT false,
            error TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            message_id BIGINT,
            PRIMARY KEY (bank_id, id),
            FOREIGN KEY (bank_id) REFERENCES {s}banks(bank_id) ON DELETE CASCADE,
            CONSTRAINT people_queue_task_type_check CHECK (task_type IN ('representation', 'dream'))
        )
        """
    )
    # One work unit's pending items, oldest first: the claim, the batch fetch and the retry counter.
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_people_queue_work_unit "
        f"ON {s}people_queue (bank_id, work_unit_key, processed, id)"
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_people_queue_pending ON {s}people_queue (bank_id, created_at) "
        f"WHERE processed = false"
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_people_queue_message ON {s}people_queue (bank_id, message_id) "
        f"WHERE message_id IS NOT NULL"
    )
    # A dream is enqueued once per work unit while one is pending (Honcho: uq_queue_dream_pending_work_unit_key).
    op.execute(
        f"CREATE UNIQUE INDEX IF NOT EXISTS uq_people_queue_dream_pending "
        f"ON {s}people_queue (bank_id, work_unit_key) WHERE task_type = 'dream' AND processed = false"
    )

    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {s}people_active_queue_sessions (
            bank_id TEXT NOT NULL,
            id TEXT NOT NULL,
            work_unit_key TEXT NOT NULL,
            last_updated TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (bank_id, id),
            UNIQUE (bank_id, work_unit_key),
            FOREIGN KEY (bank_id) REFERENCES {s}banks(bank_id) ON DELETE CASCADE
        )
        """
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_people_active_queue_sessions_updated "
        f"ON {s}people_active_queue_sessions (last_updated)"
    )


def _pg_downgrade() -> None:
    s = _pg_schema_prefix()
    op.execute(f"DROP TABLE IF EXISTS {s}people_active_queue_sessions")
    op.execute(f"DROP TABLE IF EXISTS {s}people_queue")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)  # oracle slot intentionally absent: see the module docstring


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
