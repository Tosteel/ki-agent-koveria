from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .competitor_profile_structured import run_step_2_3_competitor_profile_structured
from .models import Step23CompetitorProfileStructuredRequest, Step23CompetitorProfileStructuredResponse


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    # @router.post("/competitive-intelligence/step-2-3-competitor-profile-structured/run", response_model=Step23CompetitorProfileStructuredResponse)
    def step2_3_competitor_profile_structured_run(
        req: Step23CompetitorProfileStructuredRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> Step23CompetitorProfileStructuredResponse:
        ensure_user_dirs(s, user_id)
        result = run_step_2_3_competitor_profile_structured(
            req=req,
            user_root=s.user_dir(user_id),
            work_root=s.user_work_dir(user_id),
        )
        return Step23CompetitorProfileStructuredResponse(competitor_profile_structured=result)

    return router
