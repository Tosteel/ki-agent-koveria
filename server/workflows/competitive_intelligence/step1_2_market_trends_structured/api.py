from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .market_trends_structured import run_step_1_2_market_trends_structured
from .models import Step12MarketTrendsStructuredRequest, Step12MarketTrendsStructuredResponse


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    # @router.post("/competitive-intelligence/step-1-2-market-trends-structured/run", response_model=Step12MarketTrendsStructuredResponse)
    def step1_2_market_trends_structured_run(
        req: Step12MarketTrendsStructuredRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> Step12MarketTrendsStructuredResponse:
        ensure_user_dirs(s, user_id)
        result = run_step_1_2_market_trends_structured(
            req=req,
            user_root=s.user_dir(user_id),
            work_root=s.user_work_dir(user_id),
        )
        return Step12MarketTrendsStructuredResponse(market_trends_structured=result)

    return router
