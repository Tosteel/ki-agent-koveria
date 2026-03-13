from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .market_trends_summary import run_step_1_3_market_trends_summary
from .models import Step13MarketTrendsSummaryRequest, Step13MarketTrendsSummaryResponse


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    # @router.post("/competitive-intelligence/step-1-3-market-trends-summary/run", response_model=Step13MarketTrendsSummaryResponse)
    def step1_3_market_trends_summary_run(
        req: Step13MarketTrendsSummaryRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> Step13MarketTrendsSummaryResponse:
        ensure_user_dirs(s, user_id)
        result = run_step_1_3_market_trends_summary(
            req=req,
            user_root=s.user_dir(user_id),
            work_root=s.user_work_dir(user_id),
        )
        return Step13MarketTrendsSummaryResponse(market_trends_summary=result)

    return router
