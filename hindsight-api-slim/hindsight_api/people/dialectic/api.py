# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/routers/peers.py
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/utils/sse.py
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/schemas/api.py
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/utils/filter.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""FastAPI routes for the dialectic: ask a question about one person, or across a bank.

Paths live under ``/v1/default/banks/{bank_id}/people``:

* ``POST /people/{peer_id}/chat`` answers a question about a person (Honcho's peer chat).
* ``POST /people/chat`` answers a question across everyone in the bank (Honcho's workspace chat).

Both answer with the text and the sources it can cite (``sources``: the URL and time each conclusion,
message and fact the agent read was said). ``stream: true`` answers as a ``text/event-stream``; the terminal
event carries the sources. ``include_evidence`` adds what the agent read in full. Authentication follows
``create_people_router``: the request is authenticated against the engine's tenant extension first.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Annotated, Any, TypeVar

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Path, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ...models import RequestContext
from .. import peers as peers_crud
from .. import schemas
from ..context import PeopleContext
from ..exceptions import ConflictException, PeopleError, ResourceNotFoundException, ValidationException
from ..sanitization import NulStripped
from .chat import agentic_chat, agentic_chat_stream, workspace_chat, workspace_chat_stream
from .evidence import Evidence, EvidenceAccumulator, Source
from .schema_conversion import json_response_schema_to_pydantic
from .settings import MAX_QUERY_LENGTH, ReasoningLevel

if TYPE_CHECKING:
    from ...engine.memory_engine import MemoryEngine

logger = logging.getLogger(__name__)

MAX_SESSION_ALLOWLIST_ENTRIES = 100

T = TypeVar("T")

_INCLUDE_EVIDENCE_DESCRIPTION = (
    "Return what the answer was built from: the conclusions and messages the agent read and the tools it called. "
    "It over-reports: it lists what the agent read, not only what the answer relied on."
)


class DialecticOptions(BaseModel):
    session_id: str | None = Field(None, description="ID of the session to scope the query to")
    filters: dict[str, Any] | None = Field(
        None,
        description=(
            "Optional filters to scope recall. This endpoint supports only the 'session_id' key: a session id, a "
            'list of session ids, or {"in": [...]}. Recall (conclusions and messages) is restricted to the '
            "allowlist; unsupported keys are rejected. When session_id is also set, it must be included in the "
            "allowlist."
        ),
    )
    target: str | None = Field(None, description="Optional person to ask about, from the perspective of this person")
    query: Annotated[
        str, Field(min_length=1, max_length=MAX_QUERY_LENGTH, description="Dialectic API Prompt"), NulStripped
    ]
    stream: bool = Field(
        default=False,
        description=(
            "Return the answer as a text/event-stream. The answer arrives when the agent finishes gathering "
            "context, in pieces that rejoin to the answer; the terminal event carries the sources."
        ),
    )
    reasoning_level: ReasoningLevel = Field(
        default="low", description="Level of reasoning to apply: minimal, low, medium, high, or max"
    )
    response_format: dict[str, Any] | None = Field(
        None,
        description=(
            "Optional JSON Schema (root type 'object') the response must conform to. When provided, `content` is a "
            "JSON string matching this schema. Only a conservative subset of JSON Schema is supported; unsupported "
            "schemas are rejected with 422. Constraint keywords (minItems, maxLength, ...) are hints to the model, "
            "not enforced server-side."
        ),
    )
    include_evidence: bool = Field(default=False, description=_INCLUDE_EVIDENCE_DESCRIPTION)


class WorkspaceChatOptions(BaseModel):
    """Options for bank-level chat (no anchor person; see DialecticOptions)."""

    session_id: str | None = Field(None, description="Optional session to scope message tools to")
    filters: dict[str, Any] | None = Field(
        None,
        description="Optional filters to scope recall; only the 'session_id' key is supported (see DialecticOptions)",
    )
    query: Annotated[str, Field(min_length=1, max_length=MAX_QUERY_LENGTH, description="Bank chat prompt"), NulStripped]
    stream: bool = Field(default=False, description="Return the answer as a text/event-stream (see DialecticOptions)")
    reasoning_level: ReasoningLevel = Field(
        default="low", description="Level of reasoning to apply: minimal, low, medium, high, or max"
    )
    response_format: dict[str, Any] | None = Field(
        None, description="Optional JSON Schema the response must conform to (see DialecticOptions)"
    )
    include_evidence: bool = Field(default=False, description=_INCLUDE_EVIDENCE_DESCRIPTION)


class DialecticResponse(BaseModel):
    content: str | None
    sources: list[Source] = Field(
        default_factory=list,
        description="Everything the agent read, each with the URL and time it was said, for citing the answer.",
    )
    evidence: Evidence | None = Field(
        default=None, description="What the answer was built from. Present only when `include_evidence` is true."
    )


async def format_dialectic_sse_stream(
    chunks: AsyncIterator[str],
    evidence: EvidenceAccumulator,
    include_evidence: bool = False,
) -> AsyncIterator[str]:
    """Frame answer chunks as SSE events, then a terminal event.

    Sources and evidence can only be known once the answer has finished streaming, so they ride on the terminal
    event rather than a frame of their own. The terminal event always carries ``sources``; ``evidence`` only when
    the caller asked for it.
    """
    async for chunk in chunks:
        yield f"data: {json.dumps({'delta': {'content': chunk}, 'done': False})}\n\n"

    built = evidence.build()
    final: dict[str, object] = {
        "done": True,
        "sources": json.loads(json.dumps([s.model_dump() for s in built.sources()])),
    }
    if include_evidence:
        final["evidence"] = json.loads(built.model_dump_json())
    yield f"data: {json.dumps(final)}\n\n"


def extract_session_allowlist(filters: dict[str, Any] | None, must_include: str | None = None) -> list[str] | None:
    """Parse the session allowlist from chat filters (Honcho ``utils/filter.py``).

    The only key supported is ``session_id``: a session id, a list of session ids, or ``{"in": [...]}``.
    Returns None when no allowlist was asked for.

    Raises:
        ValidationException: On an unsupported key or shape, an over-long list, or a ``must_include``
            session the allowlist does not cover.
    """
    if not filters:
        return None
    unsupported = sorted(set(filters) - {"session_id"})
    if unsupported:
        raise ValidationException(f"Unsupported filter key(s) {unsupported}: only 'session_id' is supported")
    value = filters["session_id"]
    if isinstance(value, str):
        allowlist = [value]
    elif isinstance(value, list):
        allowlist = [str(v) for v in value]
    elif isinstance(value, dict) and set(value) == {"in"} and isinstance(value["in"], list):
        allowlist = [str(v) for v in value["in"]]
    else:
        raise ValidationException("'session_id' must be a session id, a list of session ids, or {\"in\": [...]}")
    if len(allowlist) > MAX_SESSION_ALLOWLIST_ENTRIES:
        raise ValidationException(f"A session allowlist holds at most {MAX_SESSION_ALLOWLIST_ENTRIES} sessions")
    if must_include is not None and must_include not in allowlist:
        raise ValidationException(f"session_id {must_include!r} is not included in the filters' session allowlist")
    return allowlist


def _http_error(error: PeopleError) -> HTTPException:
    if isinstance(error, ResourceNotFoundException):
        return HTTPException(status_code=404, detail=str(error))
    if isinstance(error, ConflictException):
        return HTTPException(status_code=409, detail=str(error))
    return HTTPException(status_code=422, detail=str(error))


def _response_model(response_format: dict[str, Any] | None) -> type[BaseModel] | None:
    """Convert the caller's JSON Schema so malformed schemas fail immediately with 422."""
    if response_format is None:
        return None
    try:
        return json_response_schema_to_pydantic(response_format)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid response_format: {e}") from None


def create_dialectic_router(get_engine: Callable[[], MemoryEngine]) -> APIRouter:
    """Build the dialectic router. ``get_engine`` returns the running memory engine."""
    router = APIRouter(prefix="/v1/default/banks/{bank_id}", tags=["People"])

    def request_context(request: Request, authorization: str | None = Header(default=None)) -> RequestContext:
        """The request's context, shared with the other routes of the request (one authentication)."""
        cached = getattr(request.state, "hs_request_context", None)
        if cached is not None:
            return cached
        api_key = None
        if authorization:
            api_key = (
                authorization[7:].strip() if authorization.lower().startswith("bearer ") else authorization.strip()
            )
        context = RequestContext(api_key=api_key)
        request.state.hs_request_context = context
        return context

    async def writable_context(
        bank_id: str = Path(...), rc: RequestContext = Depends(request_context)
    ) -> PeopleContext:
        """Authenticate the caller (this sets the tenant schema); the bank is created with defaults when missing."""
        engine = get_engine()
        await engine._authenticate_tenant(rc)
        await engine.ensure_bank_profile(bank_id, request_context=rc)
        return PeopleContext(engine=engine, request_context=rc)

    async def run(call: Awaitable[T]) -> T:
        try:
            return await call
        except PeopleError as e:
            raise _http_error(e) from e

    def allowlist(options: DialecticOptions | WorkspaceChatOptions) -> list[str] | None:
        try:
            return extract_session_allowlist(options.filters, must_include=options.session_id)
        except PeopleError as e:
            raise _http_error(e) from e

    async def sse(
        chunks: AsyncIterator[str], evidence: EvidenceAccumulator, include_evidence: bool
    ) -> AsyncIterator[str]:
        try:
            async for frame in format_dialectic_sse_stream(chunks, evidence, include_evidence):
                yield frame
        except PeopleError as e:
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

    def response_for(content: str, evidence: EvidenceAccumulator, include_evidence: bool) -> DialecticResponse:
        built = evidence.build()
        return DialecticResponse(
            content=content if content else None,
            sources=built.sources(),
            evidence=built if include_evidence else None,
        )

    @router.post(
        "/people/chat",
        summary="Bank Chat",
        response_model=DialecticResponse,
        responses={200: {"content": {"application/json": {}, "text/event-stream": {}}}},
    )
    async def bank_chat(
        bank_id: str = Path(...),
        options: WorkspaceChatOptions = Body(...),
        ctx: PeopleContext = Depends(writable_context),
    ) -> Any:
        """
        Query everyone in a bank using natural language. Performs agentic search and reasoning across the bank's
        people, messages and retained facts, and answers citing where each claim was said.
        """
        response_model = _response_model(options.response_format)
        session_allowlist = allowlist(options)
        evidence = EvidenceAccumulator()
        if options.stream:
            return StreamingResponse(
                sse(
                    workspace_chat_stream(
                        ctx,
                        bank_id,
                        session_name=options.session_id,
                        query=options.query,
                        reasoning_level=options.reasoning_level,
                        response_model=response_model,
                        evidence=evidence,
                        session_allowlist=session_allowlist,
                    ),
                    evidence,
                    options.include_evidence,
                ),
                media_type="text/event-stream",
            )
        content = await run(
            workspace_chat(
                ctx,
                bank_id,
                session_name=options.session_id,
                query=options.query,
                reasoning_level=options.reasoning_level,
                response_model=response_model,
                evidence=evidence,
                session_allowlist=session_allowlist,
            )
        )
        return response_for(content, evidence, options.include_evidence)

    @router.post(
        "/people/{peer_id}/chat",
        summary="Person Chat",
        response_model=DialecticResponse,
        responses={200: {"content": {"application/json": {}, "text/event-stream": {}}}},
    )
    async def chat(
        bank_id: str = Path(...),
        peer_id: str = Path(...),
        options: DialecticOptions = Body(...),
        ctx: PeopleContext = Depends(writable_context),
    ) -> Any:
        """
        Query a person's representation using natural language. Performs agentic search and reasoning to
        comprehensively answer the query based on all latent knowledge gathered about the person from their
        messages and conclusions, and answers citing where each claim was said.
        """
        response_model = _response_model(options.response_format)
        session_allowlist = allowlist(options)

        # Get or create the person to ensure they exist
        await run(peers_crud.get_or_create_peers(ctx, bank_id, [schemas.PeerSpec(name=peer_id)]))

        evidence = EvidenceAccumulator()
        observed = options.target if options.target is not None else peer_id
        if observed != peer_id:
            # Checked before streaming starts, so a missing target answers 404 and not an error event.
            await run(peers_crud.get_peer(ctx, bank_id, observed))
        if options.stream:
            return StreamingResponse(
                sse(
                    agentic_chat_stream(
                        ctx,
                        bank_id,
                        session_name=options.session_id,
                        query=options.query,
                        observer=peer_id,
                        observed=observed,
                        reasoning_level=options.reasoning_level,
                        session_allowlist=session_allowlist,
                        response_model=response_model,
                        evidence=evidence,
                    ),
                    evidence,
                    options.include_evidence,
                ),
                media_type="text/event-stream",
            )
        content = await run(
            agentic_chat(
                ctx,
                bank_id,
                session_name=options.session_id,
                query=options.query,
                observer=peer_id,
                # if target is given, that's the observed person. otherwise, observer==observed
                # and it's answered from the omniscient perspective
                observed=observed,
                reasoning_level=options.reasoning_level,
                session_allowlist=session_allowlist,
                response_model=response_model,
                evidence=evidence,
            )
        )
        return response_for(content, evidence, options.include_evidence)

    return router
