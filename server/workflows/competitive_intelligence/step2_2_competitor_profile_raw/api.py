from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .competitor_profile_raw import run_step_2_2_competitor_profile_raw
from .models import Step22CompetitorProfileRawRequest, Step22CompetitorProfileRawResponse


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    # @router.post("/competitive-intelligence/step-2-2-competitor-profile-raw/run", response_model=Step22CompetitorProfileRawResponse)
    def step2_2_competitor_profile_raw_run(
        req: Step22CompetitorProfileRawRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> Step22CompetitorProfileRawResponse:
        ensure_user_dirs(s, user_id)
        result = run_step_2_2_competitor_profile_raw(
            req=req,
            user_root=s.user_dir(user_id),
            work_root=s.user_work_dir(user_id),
        )
        return Step22CompetitorProfileRawResponse(competitor_profile_raw=result)

    return router
