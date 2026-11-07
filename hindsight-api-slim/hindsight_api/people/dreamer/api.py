# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/routers/dreams.py
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/routers/workspaces.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""Representation History and dream scheduling routes.

Paths live under ``/v1/default/banks/{bank_id}/dreams``. The router is built by ``create_dream_router``
from a function returning the memory engine, the same way ``create_people_router`` is.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Path, Query, Request
from pydantic import BaseModel

from ...models import RequestContext
from ..context import PeopleContext
from ..dream_history import count_runs, get_run_detail, list_runs
from ..exceptions import PeopleError, ResourceNotFoundException
from .schemas import DreamRun, DreamRunDetail, DreamRunList, ScheduleDreamRequest
from .settings import DREAM

if TYPE_CHECKING:
    from ...engine.memory_engine import MemoryEngine

logger = logging.getLogger(__name__)


class DreamRunPage(BaseModel):
    """One page of dream runs, shaped like the people router's pages."""

    items: list[DreamRun]
    total: int
    page: int
    size: int
    pages: int


def create_dream_router(get_engine: "Callable[[], MemoryEngine]") -> APIRouter:
    """Build the dream router. ``get_engine`` returns the running memory engine."""
    router = APIRouter(prefix="/v1/default/banks/{bank_id}/dreams", tags=["Dreams"])

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

    @router.post("/list", response_model=DreamRunPage, summary="List dream runs, newest first")
    async def list_dreams(
        bank_id: str = Path(...),
        options: DreamRunList | None = Body(None, description="Filtering options for the dream run list"),
        page: int = Query(1, ge=1),
        size: int = Query(50, ge=1, le=100),
        ctx: PeopleContext = Depends(people_context),
    ) -> DreamRunPage:
        """
        List dream runs, newest first. Results are paginated.

        Each run records what one dream changed in a representation. Filter by `observed_id` for a
        peer's history, and add `observer_id` for a single representation. Runs are kept for the
        retention window (30 days by default).
        """
        observed = options.observed_id if options else None
        observer = options.observer_id if options else None
        runs = await list_runs(ctx, bank_id, observed=observed, observer=observer, limit=size, offset=(page - 1) * size)
        total = await count_runs(ctx, bank_id, observed=observed, observer=observer)
        return DreamRunPage(items=runs, total=total, page=page, size=size, pages=(total + size - 1) // size)

    @router.get("/{run_id}", response_model=DreamRunDetail, summary="Get one dream run with its changes")
    async def get_dream(
        bank_id: str = Path(...),
        run_id: str = Path(..., description="The dream's run ID"),
        ctx: PeopleContext = Depends(people_context),
    ) -> DreamRunDetail:
        """
        Get one dream run with every change it committed.

        The response lists the conclusions the dream created with the conclusions each was derived from,
        the conclusions it removed as they read when removed, and the peer card before and after.
        """
        try:
            return await get_run_detail(ctx, bank_id, run_id)
        except ResourceNotFoundException as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except PeopleError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    @router.post("/schedule", status_code=202, summary="Schedule a dream now")
    async def schedule_dream(
        bank_id: str = Path(...),
        request: ScheduleDreamRequest = Body(..., description="Dream scheduling parameters"),
        ctx: PeopleContext = Depends(people_context),
    ) -> dict[str, Any]:
        """
        Manually schedule a dream task for a specific pair of people.

        This endpoint bypasses all automatic dream conditions (document threshold, minimum hours between
        dreams) and enqueues the dream for immediate processing.
        """
        from ..deriver.enqueue import enqueue_dream

        if not DREAM.ENABLED:
            raise HTTPException(status_code=400, detail="Dreams are not enabled in the system configuration")

        # Default observed to observer if not provided
        observer = request.observer
        observed = request.observed if request.observed is not None else request.observer
        await enqueue_dream(
            ctx,
            bank_id,
            observer,
            observed,
            request.dream_type,
            session_name=request.session_id,
            rebuild=request.rebuild,
            # Manual route: explicit sentinels for the scheduling-context fields.
            trigger_reason="manual",
            delay_reason="immediate",
        )
        logger.info(
            "Manually scheduled dream: %s for %s/%s/%s (session: %s)",
            request.dream_type.value,
            bank_id,
            observer,
            observed,
            request.session_id,
        )
        return {"scheduled": True, "observer": observer, "observed": observed, "dream_type": request.dream_type.value}

    return router
