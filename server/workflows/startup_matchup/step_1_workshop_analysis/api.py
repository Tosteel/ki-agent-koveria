from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .models import StartupMatchupStep1Request, StartupMatchupStep1Response
from .startup_matchup_step import run_step_1


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    #@router.post("/startup-matchup/step-1/run", response_model=StartupMatchupStep1Response)
    def startup_matchup_step_1_run(
        req: StartupMatchupStep1Request,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> StartupMatchupStep1Response:
        ensure_user_dirs(s, user_id)
        result = run_step_1(
            req=req,
            user_root=s.user_dir(user_id),
            work_root=s.user_work_dir(user_id),
        )
        return StartupMatchupStep1Response(workshop_analysis=result)

    return router
