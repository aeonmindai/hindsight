# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/routers/peers.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""FastAPI routes for people: peers (one per person handle), peer cards, representations, the memory brief,
and the sessions and messages the conclusions were drawn from.

Paths live under ``/v1/default/banks/{bank_id}/people``. The router is built by ``create_people_router``
from a function returning the memory engine, so the HTTP app owns the engine and this module owns the
routes.
"""

import datetime
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, TypeVar

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Path, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..models import RequestContext
from . import peer_card as peer_card_crud
from . import peers as peers_crud
from . import schemas
from .conclusions import summarize_documents_by_level
from .context import PeopleContext
from .exceptions import ConflictException, PeopleError, ResourceNotFoundException, ValidationException
from .memory_brief import format_memory_counts
from .representation import get_working_representation

if TYPE_CHECKING:
    from ..engine.memory_engine import MemoryEngine

logger = logging.getLogger(__name__)

MAX_SESSION_ALLOWLIST_ENTRIES = 100

T = TypeVar("T")


class Page(BaseModel):
    """One page of a list, shaped like Honcho's paginated responses."""

    items: list[Any]
    total: int
    page: int
    size: int
    pages: int


class MemoryBriefResponse(BaseModel):
    """What the memory brief says about one observer/observed pair."""

    observer: str
    observed: str
    counts_by_level: dict[str, int]
    earliest: datetime.datetime | None = None
    latest: datetime.datetime | None = None
    brief: str = Field(description="The one-line counts summary agents start a run with")


def _page(items: list[Any], total: int, page: int, size: int) -> Page:
    return Page(items=items, total=total, page=page, size=size, pages=(total + size - 1) // size if size else 0)


def extract_session_allowlist(filters: dict[str, Any] | None, must_include: str | None = None) -> list[str] | None:
    """Parse the session allowlist from representation filters (Honcho ``utils/filter.py``).

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


def create_people_router(get_engine: "Callable[[], MemoryEngine]") -> APIRouter:
    """Build the people router. ``get_engine`` returns the running memory engine."""
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

    async def people_context(bank_id: str = Path(...), rc: RequestContext = Depends(request_context)) -> PeopleContext:
        """Authenticate the caller (this sets the tenant schema) and return the people context."""
        engine = get_engine()
        await engine._authenticate_tenant(rc)
        return PeopleContext(engine=engine, request_context=rc)

    async def writable_context(bank_id: str = Path(...), ctx: PeopleContext = Depends(people_context)) -> PeopleContext:
        """Like ``people_context``, and the bank is created with defaults when it does not exist yet."""
        await ctx.engine.ensure_bank_profile(bank_id, request_context=ctx.request_context)
        return ctx

    async def run(call: "Awaitable[T]") -> T:
        try:
            return await call
        except PeopleError as e:
            raise _http_error(e) from e

    @router.post("/people/list", response_model=Page, summary="List people")
    async def get_peers(
        bank_id: str = Path(...),
        options: schemas.PeerGet | None = Body(None, description="Filtering options for the peers list"),
        reverse: bool = Query(False, description="Whether to reverse the order of results"),
        page: int = Query(1, ge=1),
        size: int = Query(50, ge=1, le=100),
        ctx: PeopleContext = Depends(people_context),
    ) -> Page:
        """Get all peers of a bank, paginated with optional metadata filters."""
        filter_param = options.filters if options and options.filters else None
        items, total = await run(
            peers_crud.get_peers(ctx, bank_id, filter_param, reverse, limit=size, offset=(page - 1) * size)
        )
        return _page([p.model_dump(by_alias=True, mode="json") for p in items], total, page, size)

    @router.post("/people", summary="Get or create a person")
    async def get_or_create_peer(
        bank_id: str = Path(...),
        peer: schemas.PeerCreate = Body(..., description="Peer creation parameters"),
        ctx: PeopleContext = Depends(writable_context),
    ) -> Any:
        """Get a peer by handle or create a new one with the given handle (201 when created, else 200)."""
        result = await run(peers_crud.get_or_create_peers(ctx, bank_id, [peer]))
        return JSONResponse(
            status_code=201 if result.created else 200,
            content=result.resource[0].model_dump(by_alias=True, mode="json"),
        )

    @router.put("/people/{peer_id}", summary="Update a person")
    async def update_peer(
        bank_id: str = Path(...),
        peer_id: str = Path(...),
        peer: schemas.PeerUpdate = Body(..., description="Updated peer parameters"),
        ctx: PeopleContext = Depends(writable_context),
    ) -> Any:
        """Update a peer's metadata and/or configuration."""
        updated = await run(peers_crud.update_peer(ctx, bank_id, peer_id, peer))
        return updated.model_dump(by_alias=True, mode="json")

    @router.post("/people/{peer_id}/sessions", response_model=Page, summary="List the sessions of a person")
    async def get_sessions_for_peer(
        bank_id: str = Path(...),
        peer_id: str = Path(...),
        options: schemas.SessionGet | None = Body(None, description="Filtering options for the sessions list"),
        reverse: bool = Query(False, description="Whether to reverse the order of results"),
        page: int = Query(1, ge=1),
        size: int = Query(50, ge=1, le=100),
        ctx: PeopleContext = Depends(people_context),
    ) -> Page:
        """Get all sessions for a peer, paginated with optional filters."""
        filter_param = options.filters if options and options.filters else None
        items, total = await run(
            peers_crud.get_sessions_for_peer(
                ctx, bank_id, peer_id, filter_param, reverse, limit=size, offset=(page - 1) * size
            )
        )
        return _page([s.model_dump(by_alias=True, mode="json") for s in items], total, page, size)

    @router.post("/people/{peer_id}/representation", response_model=schemas.RepresentationResponse)
    async def get_representation(
        bank_id: str = Path(...),
        peer_id: str = Path(...),
        options: schemas.PeerRepresentationGet = Body(..., description="Options for getting the peer representation"),
        ctx: PeopleContext = Depends(people_context),
    ) -> schemas.RepresentationResponse:
        """Get a curated subset of a peer's representation.

        If a session_id is provided in the body, the representation is scoped to that session. If a target
        is provided, it is the target's representation from the perspective of the peer. If no target is
        provided, it is the peer's own.
        """
        # Parse the session allowlist from filters (422 on unsupported keys/shapes, and on a session_id
        # the allowlist doesn't cover).
        session_allowlist = _guard(extract_session_allowlist, options.filters, options.session_id)

        embedding: list[float] | None = None
        if options.search_query:
            try:
                embedding = await ctx.embed_query(options.search_query)
            except Exception:
                # Swallowed on purpose, but not silently: without this a provider outage degrades every
                # search request to derived+recent retrieval with no signal anywhere.
                logger.warning(
                    "Representation search embedding failed for bank %s, degrading to non-semantic retrieval",
                    bank_id,
                    exc_info=True,
                )

        observed = options.target if options.target is not None else peer_id
        representation = await run(
            get_working_representation(
                ctx,
                bank_id,
                observer=peer_id,
                observed=observed,
                session_allowlist=[options.session_id] if options.session_id is not None else session_allowlist,
                # Only ask for the semantic branch when we actually have an embedding.
                include_semantic_query=options.search_query if embedding is not None else None,
                embedding=embedding,
                semantic_search_top_k=options.search_top_k,
                semantic_search_max_distance=options.search_max_distance,
                include_most_derived=options.include_most_frequent
                if options.include_most_frequent is not None
                else False,
                max_observations=options.max_conclusions
                if options.max_conclusions is not None
                else schemas.WORKING_REPRESENTATION_MAX_OBSERVATIONS,
            )
        )
        return schemas.RepresentationResponse(representation=representation.format_as_markdown())

    @router.get("/people/{peer_id}/card", response_model=schemas.PeerCardResponse)
    async def get_peer_card(
        bank_id: str = Path(...),
        peer_id: str = Path(..., description="Handle of the observer peer"),
        target: str | None = Query(
            None,
            description=(
                "Optional target peer to retrieve a card for, from the observer's perspective. If not provided, "
                "returns the observer's own card"
            ),
        ),
        ctx: PeopleContext = Depends(people_context),
    ) -> schemas.PeerCardResponse:
        """Get the peer card the observer peer has for the target peer, if it exists."""
        observed = target if target is not None else peer_id
        peer_card = await run(peer_card_crud.get_peer_card(ctx, bank_id, peer_id, observed))
        return schemas.PeerCardResponse(peer_card=peer_card)

    @router.put("/people/{peer_id}/card", response_model=schemas.PeerCardResponse)
    async def set_peer_card(
        bank_id: str = Path(...),
        peer_id: str = Path(..., description="Handle of the observer peer"),
        peer_card_data: schemas.PeerCardSet = Body(..., description="Peer card data to set"),
        target: str | None = Query(
            None,
            description=(
                "Optional target peer to set a card for, from the observer's perspective. If not provided, sets "
                "the observer's own card"
            ),
        ),
        ctx: PeopleContext = Depends(writable_context),
    ) -> schemas.PeerCardResponse:
        """Set the peer card the observer peer has for the target peer."""
        observed = target if target is not None else peer_id
        await run(peer_card_crud.set_peer_card(ctx, bank_id, peer_card_data.peer_card, peer_id, observed))
        return schemas.PeerCardResponse(peer_card=peer_card_data.peer_card)

    @router.get("/people/{peer_id}/context", response_model=schemas.PeerContext)
    async def get_peer_context(
        bank_id: str = Path(...),
        peer_id: str = Path(..., description="Handle of the observer peer"),
        target: str | None = Query(
            None,
            description=(
                "Optional target peer to get context for, from the observer's perspective. If not provided, "
                "returns the observer's own context (self-observation)"
            ),
        ),
        search_query: str | None = Query(
            None, description="Optional query to curate the representation around semantic search results"
        ),
        search_top_k: int | None = Query(None, ge=1, le=100),
        search_max_distance: float | None = Query(None, ge=0.0, le=1.0),
        include_most_frequent: bool = Query(default=True),
        max_conclusions: int | None = Query(None, ge=1, le=100),
        ctx: PeopleContext = Depends(people_context),
    ) -> schemas.PeerContext:
        """Get context for a peer, including their representation and peer card, in one call."""
        # If no target specified, get the peer's own context (self-observation)
        observed = target if target is not None else peer_id
        embedding: list[float] | None = None
        if search_query:
            try:
                embedding = await ctx.embed_query(search_query)
            except Exception:
                logger.warning("Context search embedding failed for bank %s", bank_id, exc_info=True)

        representation = await run(
            get_working_representation(
                ctx,
                bank_id,
                observer=peer_id,
                observed=observed,
                session_allowlist=None,  # Peer context is global, not session-scoped
                include_semantic_query=search_query if embedding is not None else None,
                embedding=embedding,
                semantic_search_top_k=search_top_k,
                semantic_search_max_distance=search_max_distance,
                include_most_derived=include_most_frequent,
                max_observations=max_conclusions
                if max_conclusions is not None
                else schemas.WORKING_REPRESENTATION_MAX_OBSERVATIONS,
            )
        )
        try:
            peer_card = await peer_card_crud.get_peer_card(ctx, bank_id, peer_id, observed)
        except ResourceNotFoundException:
            peer_card = None
        return schemas.PeerContext(
            peer_id=peer_id,
            target_id=observed,
            representation=representation.format_as_markdown(),
            peer_card=peer_card,
        )

    @router.get("/people/{peer_id}/memory-brief", response_model=MemoryBriefResponse)
    async def get_memory_brief(
        bank_id: str = Path(...),
        peer_id: str = Path(..., description="Handle of the observer peer"),
        target: str | None = Query(None, description="Handle of the observed peer (default: the observer)"),
        ctx: PeopleContext = Depends(people_context),
    ) -> MemoryBriefResponse:
        """Conclusion counts by level and the dates they span for one observer/observed pair."""
        observed = target if target is not None else peer_id
        summary = await run(summarize_documents_by_level(ctx, bank_id, observer=peer_id, observed=observed))
        return MemoryBriefResponse(
            observer=peer_id,
            observed=observed,
            counts_by_level=summary.counts_by_level,
            earliest=summary.earliest,
            latest=summary.latest,
            brief=format_memory_counts(summary),
        )

    @router.post("/people/{peer_id}/search", summary="Search the messages of a person")
    async def search_peer(
        bank_id: str = Path(...),
        peer_id: str = Path(...),
        body: schemas.MessageSearchOptions = Body(
            ..., description="Message search parameters. Use `limit` to control the number of results returned."
        ),
        ctx: PeopleContext = Depends(people_context),
    ) -> list[Any]:
        """Search a peer's messages, optionally filtered by session."""
        messages = await run(peers_crud.search_messages(ctx, bank_id, peer_id, body.query, body.filters, body.limit))
        return [m.model_dump(by_alias=True, mode="json") for m in messages]

    @router.post("/people-sessions/list", response_model=Page, summary="List the sessions of a bank")
    async def list_sessions(
        bank_id: str = Path(...),
        options: schemas.SessionGet | None = Body(None, description="Filtering options for the sessions list"),
        reverse: bool = Query(False),
        page: int = Query(1, ge=1),
        size: int = Query(50, ge=1, le=100),
        ctx: PeopleContext = Depends(people_context),
    ) -> Page:
        """Get all sessions of a bank, paginated with optional metadata filters."""
        filter_param = options.filters if options and options.filters else None
        items, total = await run(
            peers_crud.get_sessions(ctx, bank_id, filter_param, reverse, limit=size, offset=(page - 1) * size)
        )
        return _page([s.model_dump(by_alias=True, mode="json") for s in items], total, page, size)

    @router.get("/people-sessions/{session_id}/messages", response_model=Page, summary="List the messages of a session")
    async def list_messages(
        bank_id: str = Path(...),
        session_id: str = Path(...),
        reverse: bool = Query(False),
        page: int = Query(1, ge=1),
        size: int = Query(50, ge=1, le=100),
        ctx: PeopleContext = Depends(people_context),
    ) -> Page:
        """Get the messages of a session in the order they were said, each with where it came from."""
        items, total = await run(
            peers_crud.get_messages(ctx, bank_id, session_id, reverse, limit=size, offset=(page - 1) * size)
        )
        return _page([m.model_dump(by_alias=True, mode="json") for m in items], total, page, size)

    return router


def _guard(fn: "Callable[..., T]", *args: Any) -> T:
    """Call a plain function, mapping its people errors to HTTP errors like ``run`` does."""
    try:
        return fn(*args)
    except PeopleError as e:
        raise _http_error(e) from e
