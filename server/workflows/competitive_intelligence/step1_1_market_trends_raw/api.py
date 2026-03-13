from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .market_trends_raw import run_step_1_1_market_trends_raw
from .models import Step11MarketTrendsRawRequest, Step11MarketTrendsRawResponse


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    # @router.post("/competitive-intelligence/step-1-1-market-trends-raw/run", response_model=Step11MarketTrendsRawResponse)
    def step1_1_market_trends_raw_run(
        req: Step11MarketTrendsRawRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> Step11MarketTrendsRawResponse:
        ensure_user_dirs(s, user_id)
        result = run_step_1_1_market_trends_raw(req=req)
        return Step11MarketTrendsRawResponse(market_trends_raw=result)

    return router
