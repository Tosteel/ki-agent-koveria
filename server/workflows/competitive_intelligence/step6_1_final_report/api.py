from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .models import Step61FinalReportRequest, Step61FinalReportResponse
from .step6_1_final_report import run_step_6_1_final_report


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    # @router.post("/competitive-intelligence/step-6-1-final-report/run", response_model=Step61FinalReportResponse)
    def step6_1_final_report_run(
        req: Step61FinalReportRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> Step61FinalReportResponse:
        ensure_user_dirs(s, user_id)
        result = run_step_6_1_final_report(
            req=req,
            user_root=s.user_dir(user_id),
            work_root=s.user_work_dir(user_id),
        )
        return Step61FinalReportResponse(final_report=result)

    return router

