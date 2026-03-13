from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .models import Step51RecommendationsRequest, Step51RecommendationsResponse
from .step5_1_recommendations import run_step_5_1_recommendations


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    # @router.post("/competitive-intelligence/step-5-1-recommendations/run", response_model=Step51RecommendationsResponse)
    def step5_1_recommendations_run(
        req: Step51RecommendationsRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> Step51RecommendationsResponse:
        ensure_user_dirs(s, user_id)
        result = run_step_5_1_recommendations(
            req=req,
            user_root=s.user_dir(user_id),
            work_root=s.user_work_dir(user_id),
        )
        return Step51RecommendationsResponse(recommendations=result)

    return router

