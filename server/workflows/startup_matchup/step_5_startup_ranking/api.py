from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .models import StartupMatchupStep5Request, StartupMatchupStep5Response
from .startup_matchup_step import run_step_5


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    @router.post("/startup-matchup/step-5/run", response_model=StartupMatchupStep5Response)
    def startup_matchup_step_5_run(
        req: StartupMatchupStep5Request,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> StartupMatchupStep5Response:
        ensure_user_dirs(s, user_id)
        result = run_step_5(
            req=req,
            user_root=s.user_dir(user_id),
            work_root=s.user_work_dir(user_id),
        )
        return StartupMatchupStep5Response(startup_ranked_list=result)

    return router
