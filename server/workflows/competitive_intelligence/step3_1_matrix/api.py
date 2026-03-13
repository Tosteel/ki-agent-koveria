from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .models import Step31MatrixRequest, Step31MatrixResponse
from .step3_1_matrix import run_step_3_1_matrix


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    # @router.post("/competitive-intelligence/step-3-1-matrix/run", response_model=Step31MatrixResponse)
    def step3_1_matrix_run(
        req: Step31MatrixRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> Step31MatrixResponse:
        ensure_user_dirs(s, user_id)
        result = run_step_3_1_matrix(
            req=req,
            user_root=s.user_dir(user_id),
            work_root=s.user_work_dir(user_id),
        )
        return Step31MatrixResponse(matrix=result)

    return router

