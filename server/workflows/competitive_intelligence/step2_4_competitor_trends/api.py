from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .competitor_trends import run_step_2_4_competitor_trends
from .models import Step24CompetitorTrendsRequest, Step24CompetitorTrendsResponse


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    # @router.post("/competitive-intelligence/step-2-4-competitor-trends/run", response_model=Step24CompetitorTrendsResponse)
    def step2_4_competitor_trends_run(
        req: Step24CompetitorTrendsRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> Step24CompetitorTrendsResponse:
        ensure_user_dirs(s, user_id)
        result = run_step_2_4_competitor_trends(
            req=req,
            user_root=s.user_dir(user_id),
            work_root=s.user_work_dir(user_id),
        )
        return Step24CompetitorTrendsResponse(competitor_trends=result)

    return router
