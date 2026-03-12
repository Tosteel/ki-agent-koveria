from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .models import StartupMatchupStep7Request, StartupMatchupStep7Response
from .startup_matchup_step import run_step_7


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    #@router.post("/startup-matchup/step-7/run", response_model=StartupMatchupStep7Response)
    def startup_matchup_step_7_run(
        req: StartupMatchupStep7Request,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> StartupMatchupStep7Response:
        ensure_user_dirs(s, user_id)
        result = run_step_7(
            req=req,
            user_root=s.user_dir(user_id),
            work_root=s.user_work_dir(user_id),
        )
        return StartupMatchupStep7Response(startup_profiles=result)

    return router
