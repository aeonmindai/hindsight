"""People dreamer tables: dream runs, the changes each run made, and per-pair dream state.

Revision ID: d8c4f6b0a235
Revises: c7b3e5a9f124
Create Date: 2026-11-07

``people_dream_runs`` and ``people_dream_run_changes`` are Honcho's Representation History (``dream_runs``
and ``dream_run_changes``): one row per dream over an (observer, observed) pair, and one row per
committed change (a created conclusion, a removed conclusion with a copy of its content, or a peer-card
update). Conclusion ids are ids of ``people_conclusions`` rows but carry no foreign key: a removed
conclusion's id must stay readable in history after its row is hard-deleted.

``people_dream_state`` holds what Honcho keeps in ``Collection.internal_metadata`` for a pair: when the
last dream ran, how many explicit conclusions it had seen, the work-list backlog and the working notes.

PG only, like the other people migrations: Oracle does not run the people features.
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "d8c4f6b0a235"
down_revision: str | Sequence[str] | None = "c7b3e5a9f124"
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
        CREATE TABLE IF NOT EXISTS {s}people_dream_runs (
            bank_id TEXT NOT NULL,
            id TEXT NOT NULL,
            observer TEXT NOT NULL,
            observed TEXT NOT NULL,
            dream_type TEXT,
            trigger_reason TEXT,
            status TEXT NOT NULL DEFAULT 'running',
            created_count INTEGER NOT NULL DEFAULT 0,
            removed_count INTEGER NOT NULL DEFAULT 0,
            peer_card_updated BOOLEAN NOT NULL DEFAULT false,
            started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            finished_at TIMESTAMPTZ,
            pruned_at TIMESTAMPTZ,
            PRIMARY KEY (bank_id, id),
            FOREIGN KEY (bank_id) REFERENCES {s}banks(bank_id) ON DELETE CASCADE,
            CONSTRAINT people_dream_runs_status_valid
                CHECK (status IN ('running', 'completed', 'failed', 'interrupted'))
        )
        """
    )
    # Listing a peer's history, newest first.
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_people_dream_runs_observed_started "
        f"ON {s}people_dream_runs (bank_id, observed, started_at, id)"
    )
    # Retention and the stale-run sweep, which run across banks.
    op.execute(f"CREATE INDEX IF NOT EXISTS idx_people_dream_runs_started_at ON {s}people_dream_runs (started_at)")

    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {s}people_dream_run_changes (
            bank_id TEXT NOT NULL,
            id BIGINT GENERATED ALWAYS AS IDENTITY,
            run_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            specialist TEXT,
            conclusion_id TEXT,
            level TEXT,
            content TEXT,
            session_name TEXT,
            source_ids JSONB,
            replaced_by TEXT,
            peer_card_before JSONB,
            peer_card_after JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (bank_id, id),
            FOREIGN KEY (bank_id, run_id) REFERENCES {s}people_dream_runs (bank_id, id) ON DELETE CASCADE,
            CONSTRAINT people_dream_run_changes_kind_valid CHECK (kind IN ('created', 'removed', 'peer_card'))
        )
        """
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_people_dream_run_changes_run ON {s}people_dream_run_changes (bank_id, run_id, id)"
    )
    # Resolving a source that a later dream removed.
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_people_dream_run_changes_removed "
        f"ON {s}people_dream_run_changes (bank_id, conclusion_id) WHERE kind = 'removed'"
    )
    # Session deletion removes the copies that session produced.
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_people_dream_run_changes_session "
        f"ON {s}people_dream_run_changes (bank_id, session_name) WHERE session_name IS NOT NULL"
    )

    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {s}people_dream_state (
            bank_id TEXT NOT NULL,
            observer TEXT NOT NULL,
            observed TEXT NOT NULL,
            internal_metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (bank_id, observer, observed),
            FOREIGN KEY (bank_id) REFERENCES {s}banks(bank_id) ON DELETE CASCADE
        )
        """
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_people_dream_state_backlog ON {s}people_dream_state (bank_id) "
        f"WHERE COALESCE(((internal_metadata -> 'dream') ->> 'work_list_backlog')::integer, 0) > 0"
    )


def _pg_downgrade() -> None:
    s = _pg_schema_prefix()
    for table in ("people_dream_state", "people_dream_run_changes", "people_dream_runs"):
        op.execute(f"DROP TABLE IF EXISTS {s}{table}")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)  # oracle slot intentionally absent: see the module docstring


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
