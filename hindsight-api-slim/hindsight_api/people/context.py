"""Shared plumbing for the people package (peers, representations, deriver, dreamer, dialectic).

Honcho's code reaches its database through SQLAlchemy sessions, its LLM through its own LLM layer
and its vectors through its vector-store backends. None of that is copied. In the fork every
people module takes a ``PeopleContext`` and gets from it:

* a database connection from the memory engine's pool (``async with ctx.conn() as conn``),
  used with plain asyncpg SQL (``$1`` parameters, ``conn.fetch`` / ``fetchrow`` / ``execute``);
* fully qualified names for the ``people_*`` tables (``ctx.table("peers")``);
* the engine's embeddings (``await ctx.embed(texts)``), which are local when the server runs in
  the broker-only profile;
* the engine's configured LLM providers: ``ctx.llm_background`` (consolidation settings,
  for the deriver and the dreamer) and ``ctx.llm_interactive`` (reflect settings, for the dialectic).

The people tables never hold or name memories-store tables (``memory_units`` and friends); a people
module that needs memories goes through the memory engine (``ctx.engine.recall_async`` and so on).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..engine.db_utils import acquire_with_retry
from ..engine.retain.types import embedding_to_pgvector
from ..engine.schema import fq_table
from ..models import RequestContext

if TYPE_CHECKING:
    from ..engine.memory_engine import MemoryEngine

# Tables created by the people migrations. ``ctx.table`` refuses any other name so a typo cannot
# silently reach another table.
PEOPLE_TABLES = frozenset(
    {
        "people_peers",
        "people_sessions",
        "people_session_peers",
        "people_messages",
        "people_conclusions",
        "people_conclusion_sources",
        "people_queue",
        "people_active_queue_sessions",
        "people_dream_runs",
        "people_dream_run_changes",
        "people_dream_state",
    }
)


@dataclass(frozen=True)
class PeopleContext:
    """The memory engine plus the request context the people modules run under."""

    engine: "MemoryEngine"
    request_context: RequestContext

    @classmethod
    def internal(cls, engine: "MemoryEngine") -> "PeopleContext":
        """A context for background work (the deriver, the dreamer, consolidation hooks)."""
        return cls(engine=engine, request_context=RequestContext(internal=True))

    @asynccontextmanager
    async def conn(self) -> AsyncIterator[Any]:
        """A connection from the engine's pool. Use ``conn.transaction()`` for multi-statement writes."""
        backend = await self.engine._get_backend()
        async with acquire_with_retry(backend) as conn:
            yield conn

    @staticmethod
    def table(name: str) -> str:
        """Schema-qualified people table: ``ctx.table("peers")`` is ``<schema>.people_peers``."""
        full = name if name.startswith("people_") else f"people_{name}"
        if full not in PEOPLE_TABLES:
            raise KeyError(f"{full!r} is not a people table")
        return fq_table(full)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed documents with the engine's embeddings (local in the broker-only profile)."""
        encode = getattr(self.engine.embeddings, "encode_documents", None) or self.engine.embeddings.encode
        return await encode(texts)

    async def embed_query(self, text: str) -> list[float]:
        """Embed one search query."""
        encode = getattr(self.engine.embeddings, "encode_query", None) or self.engine.embeddings.encode
        return (await encode([text]))[0]

    @staticmethod
    def vector(embedding: Any) -> str:
        """The pgvector literal asyncpg binds to a ``vector`` parameter (``$1::vector``)."""
        return embedding_to_pgvector(embedding)

    @property
    def llm_background(self) -> Any:
        """Claude-configured provider for background work (the engine's consolidation settings)."""
        return self.engine._consolidation_llm_config

    @property
    def llm_interactive(self) -> Any:
        """Claude-configured provider for questions asked about a person (the engine's reflect settings)."""
        return self.engine._reflect_llm_config
