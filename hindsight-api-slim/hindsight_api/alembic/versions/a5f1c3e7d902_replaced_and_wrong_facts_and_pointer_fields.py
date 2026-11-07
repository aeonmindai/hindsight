"""A fact can be replaced or wrong, and a pointer field names a record, never copies its value.

Revision ID: a5f1c3e7d902
Revises: c7e2f9a4d1b8
Create Date: 2026-11-07

Two store-level rules, both enforced by the database so no prompt and no caller can skip them.

1. Replaced or wrong facts are never current.
   Curation already *moves* an invalidated fact from ``memory_units`` into
   ``invalidated_memory_units`` (recall reads only the live table). The archive now also records
   why: ``invalid_kind`` is ``replaced`` (a later fact took its place, named by ``replaced_by``,
   dated by ``replaced_at``), ``wrong`` (it was never true), or ``other`` (any older invalidation).
   A trigger on ``memory_units`` refuses to insert a row whose id sits in the archive as
   ``replaced`` or ``wrong``, so no restore, import or revert path can bring one back as current
   without first clearing the kind on purpose (``engine.memories`` revert does).

2. Pointer fields.
   A value the CRM, Linear or Stripe holds is stored as a pointer, never as a copy. A fact carries
   ``pointer_kind`` (``crm``, ``linear``, ``stripe``) and ``pointer_id`` in its metadata; a check
   constraint on ``memory_units`` rejects a pointer kind outside that list and a pointer with no id.

PG only, like the other curation migrations: Oracle takes its schema from the baseline snapshot and
does not run the people features.
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "a5f1c3e7d902"
down_revision: str | Sequence[str] | None = "c7e2f9a4d1b8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(
        f"ALTER TABLE {schema}invalidated_memory_units "
        "ADD COLUMN IF NOT EXISTS invalid_kind TEXT NOT NULL DEFAULT 'other', "
        "ADD COLUMN IF NOT EXISTS replaced_by UUID, "
        "ADD COLUMN IF NOT EXISTS replaced_at TIMESTAMPTZ"
    )
    op.execute(
        f"""
        DO $$ BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'invalidated_mu_kind_check'
                           AND conrelid = '{schema}invalidated_memory_units'::regclass) THEN
            ALTER TABLE {schema}invalidated_memory_units
                ADD CONSTRAINT invalidated_mu_kind_check
                CHECK (invalid_kind IN ('replaced', 'wrong', 'other')
                       AND (invalid_kind <> 'replaced' OR replaced_by IS NOT NULL));
        END IF; END $$;
        """
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_invalidated_mu_replaced_by "
        f"ON {schema}invalidated_memory_units (replaced_by) WHERE replaced_by IS NOT NULL"
    )

    # The routine lives in the tenant schema next to the tables it reads, so it follows the
    # search_path-free qualified names below.
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION {schema}refuse_dead_fact_as_current() RETURNS trigger AS $fn$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM {schema}invalidated_memory_units a
                WHERE a.id = NEW.id AND a.invalid_kind IN ('replaced', 'wrong')
            ) THEN
                RAISE EXCEPTION 'memory % is marked % and cannot be current', NEW.id,
                    (SELECT a.invalid_kind FROM {schema}invalidated_memory_units a WHERE a.id = NEW.id)
                    USING ERRCODE = 'check_violation';
            END IF;
            RETURN NEW;
        END;
        $fn$ LANGUAGE plpgsql
        """
    )
    op.execute(f"DROP TRIGGER IF EXISTS trg_refuse_dead_fact_as_current ON {schema}memory_units")
    op.execute(
        f"CREATE TRIGGER trg_refuse_dead_fact_as_current BEFORE INSERT ON {schema}memory_units "
        f"FOR EACH ROW EXECUTE FUNCTION {schema}refuse_dead_fact_as_current()"
    )

    op.execute(
        f"""
        DO $$ BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'memory_units_pointer_check'
                           AND conrelid = '{schema}memory_units'::regclass) THEN
            ALTER TABLE {schema}memory_units
                ADD CONSTRAINT memory_units_pointer_check
                CHECK (
                    NOT (metadata ? 'pointer_kind' OR metadata ? 'pointer_id')
                    OR (COALESCE(metadata->>'pointer_kind', '') IN ('crm', 'linear', 'stripe')
                        AND COALESCE(metadata->>'pointer_id', '') <> '')
                ) NOT VALID;
        END IF; END $$;
        """
    )
    op.execute(f"ALTER TABLE {schema}memory_units VALIDATE CONSTRAINT memory_units_pointer_check")


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"ALTER TABLE {schema}memory_units DROP CONSTRAINT IF EXISTS memory_units_pointer_check")
    op.execute(f"DROP TRIGGER IF EXISTS trg_refuse_dead_fact_as_current ON {schema}memory_units")
    op.execute(f"DROP FUNCTION IF EXISTS {schema}refuse_dead_fact_as_current()")
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_invalidated_mu_replaced_by")
    op.execute(f"ALTER TABLE {schema}invalidated_memory_units DROP CONSTRAINT IF EXISTS invalidated_mu_kind_check")
    op.execute(
        f"ALTER TABLE {schema}invalidated_memory_units "
        "DROP COLUMN IF EXISTS replaced_at, DROP COLUMN IF EXISTS replaced_by, DROP COLUMN IF EXISTS invalid_kind"
    )


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)  # oracle slot intentionally absent: see the module docstring


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
