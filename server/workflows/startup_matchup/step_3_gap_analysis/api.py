from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .models import StartupMatchupStep3Request, StartupMatchupStep3Response
from .startup_matchup_step import run_step_3


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    #@router.post("/startup-matchup/step-3/run", response_model=StartupMatchupStep3Response)
    def startup_matchup_step_3_run(
        req: StartupMatchupStep3Request,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> StartupMatchupStep3Response:
        ensure_user_dirs(s, user_id)
        result = run_step_3(
            req=req,
            user_root=s.user_dir(user_id),
            work_root=s.user_work_dir(user_id),
        )
        return StartupMatchupStep3Response(gap_analysis=result)

    return router
