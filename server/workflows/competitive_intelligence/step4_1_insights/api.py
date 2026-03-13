from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .models import Step41InsightsRequest, Step41InsightsResponse
from .step4_1_insights import run_step_4_1_insights


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    # @router.post("/competitive-intelligence/step-4-1-insights/run", response_model=Step41InsightsResponse)
    def step4_1_insights_run(
        req: Step41InsightsRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> Step41InsightsResponse:
        ensure_user_dirs(s, user_id)
        result = run_step_4_1_insights(
            req=req,
            user_root=s.user_dir(user_id),
            work_root=s.user_work_dir(user_id),
        )
        return Step41InsightsResponse(insights=result)

    return router

