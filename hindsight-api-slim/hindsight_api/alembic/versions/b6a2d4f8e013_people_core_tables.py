"""People core tables: peers, sessions, session peers, messages, conclusions and their sources.

Revision ID: b6a2d4f8e013
Revises: a5f1c3e7d902
Create Date: 2026-11-07

The tables behind the people package (ported from Honcho): who is who (``people_peers``, one row
per person handle), the conversations they spoke in (``people_sessions``, ``people_session_peers``,
``people_messages``), and what is concluded about each person (``people_conclusions`` with its
reasoning and evidence edges in ``people_conclusion_sources``).

Every row carries ``bank_id``. A message keeps where it came from (``source_url``, ``said_by``,
``said_at``, ``channel``), and a conclusion keeps ``source_url`` and ``said_at`` copied from the
message it was drawn from, so every line a person model shows can cite its source.

``people_conclusion_sources`` holds both kinds of edge Honcho keeps in two tables:
``kind = 'conclusion'`` (a reasoning-tree edge: the conclusion was derived from another conclusion,
Honcho's ``document_sources``) and ``kind = 'message'`` (an evidence edge: the conclusion cites the
public id of a message, Honcho's ``conclusion_source_messages``).

PG only, like the other people migrations: Oracle does not run the people features.
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "b6a2d4f8e013"
down_revision: str | Sequence[str] | None = "a5f1c3e7d902"
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
        CREATE TABLE IF NOT EXISTS {s}people_peers (
            bank_id TEXT NOT NULL,
            id TEXT NOT NULL,
            name TEXT NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            internal_metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            configuration JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            deleted_at TIMESTAMPTZ,
            PRIMARY KEY (bank_id, id),
            UNIQUE (bank_id, name),
            FOREIGN KEY (bank_id) REFERENCES {s}banks(bank_id) ON DELETE CASCADE,
            CONSTRAINT people_peers_name_length CHECK (length(name) <= 512),
            CONSTRAINT people_peers_id_length CHECK (length(id) = 21)
        )
        """
    )
    op.execute(f"CREATE INDEX IF NOT EXISTS idx_people_peers_bank_created ON {s}people_peers (bank_id, created_at, id)")

    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {s}people_sessions (
            bank_id TEXT NOT NULL,
            id TEXT NOT NULL,
            name TEXT NOT NULL,
            is_active BOOLEAN NOT NULL DEFAULT true,
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            internal_metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            deleted_at TIMESTAMPTZ,
            PRIMARY KEY (bank_id, id),
            UNIQUE (bank_id, name),
            FOREIGN KEY (bank_id) REFERENCES {s}banks(bank_id) ON DELETE CASCADE,
            CONSTRAINT people_sessions_name_length CHECK (length(name) <= 512),
            CONSTRAINT people_sessions_id_length CHECK (length(id) = 21)
        )
        """
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_people_sessions_bank_created ON {s}people_sessions (bank_id, created_at, id)"
    )

    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {s}people_session_peers (
            bank_id TEXT NOT NULL,
            session_name TEXT NOT NULL,
            peer_name TEXT NOT NULL,
            configuration JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            internal_metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            joined_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            left_at TIMESTAMPTZ,
            PRIMARY KEY (bank_id, session_name, peer_name),
            FOREIGN KEY (bank_id, session_name) REFERENCES {s}people_sessions (bank_id, name) ON DELETE CASCADE,
            FOREIGN KEY (bank_id, peer_name) REFERENCES {s}people_peers (bank_id, name) ON DELETE CASCADE
        )
        """
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_people_session_peers_peer ON {s}people_session_peers (bank_id, peer_name)"
    )

    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {s}people_messages (
            bank_id TEXT NOT NULL,
            id BIGINT GENERATED ALWAYS AS IDENTITY,
            public_id TEXT NOT NULL,
            session_name TEXT NOT NULL,
            peer_name TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            internal_metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            token_count INTEGER NOT NULL DEFAULT 0,
            seq_in_session BIGINT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            deleted_at TIMESTAMPTZ,
            source_url TEXT,
            said_by TEXT,
            said_at TIMESTAMPTZ,
            channel TEXT,
            PRIMARY KEY (bank_id, id),
            UNIQUE (bank_id, public_id),
            CONSTRAINT uq_people_messages_session_seq UNIQUE (bank_id, session_name, seq_in_session),
            FOREIGN KEY (bank_id, session_name) REFERENCES {s}people_sessions (bank_id, name) ON DELETE CASCADE,
            FOREIGN KEY (bank_id, peer_name) REFERENCES {s}people_peers (bank_id, name) ON DELETE CASCADE,
            CONSTRAINT people_messages_public_id_length CHECK (length(public_id) = 21),
            CONSTRAINT people_messages_content_length CHECK (length(content) <= 65535)
        )
        """
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_people_messages_session_tokens "
        f"ON {s}people_messages (bank_id, session_name, id) INCLUDE (created_at, token_count)"
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_people_messages_peer ON {s}people_messages (bank_id, peer_name, created_at)"
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_people_messages_fts ON {s}people_messages "
        f"USING gin (to_tsvector('english', content))"
    )

    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {s}people_conclusions (
            bank_id TEXT NOT NULL,
            id TEXT NOT NULL,
            observer TEXT NOT NULL,
            observed TEXT NOT NULL,
            level TEXT NOT NULL DEFAULT 'explicit',
            content TEXT NOT NULL,
            embedding vector,
            times_derived INTEGER NOT NULL DEFAULT 1,
            session_name TEXT,
            internal_metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            source_url TEXT,
            said_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            deleted_at TIMESTAMPTZ,
            PRIMARY KEY (bank_id, id),
            FOREIGN KEY (bank_id) REFERENCES {s}banks(bank_id) ON DELETE CASCADE,
            CONSTRAINT people_conclusions_id_length CHECK (length(id) = 21),
            CONSTRAINT people_conclusions_content_length CHECK (length(content) <= 65535),
            CONSTRAINT people_conclusions_level_check
                CHECK (level IN ('explicit', 'deductive', 'inductive', 'contradiction'))
        )
        """
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_people_conclusions_pair_created "
        f"ON {s}people_conclusions (bank_id, observer, observed, created_at DESC) WHERE deleted_at IS NULL"
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_people_conclusions_pair_level "
        f"ON {s}people_conclusions (bank_id, observer, observed, level) WHERE deleted_at IS NULL"
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_people_conclusions_session "
        f"ON {s}people_conclusions (bank_id, session_name) WHERE deleted_at IS NULL"
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_people_conclusions_deleted "
        f"ON {s}people_conclusions (bank_id, deleted_at) WHERE deleted_at IS NOT NULL"
    )

    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {s}people_conclusion_sources (
            bank_id TEXT NOT NULL,
            conclusion_id TEXT NOT NULL,
            source_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            position INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (bank_id, conclusion_id, kind, source_id),
            FOREIGN KEY (bank_id, conclusion_id) REFERENCES {s}people_conclusions (bank_id, id) ON DELETE CASCADE,
            CONSTRAINT people_conclusion_sources_kind_check CHECK (kind IN ('conclusion', 'message'))
        )
        """
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_people_conclusion_sources_source "
        f"ON {s}people_conclusion_sources (bank_id, kind, source_id)"
    )


def _pg_downgrade() -> None:
    s = _pg_schema_prefix()
    for table in (
        "people_conclusion_sources",
        "people_conclusions",
        "people_messages",
        "people_session_peers",
        "people_sessions",
        "people_peers",
    ):
        op.execute(f"DROP TABLE IF EXISTS {s}{table}")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)  # oracle slot intentionally absent: see the module docstring


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
