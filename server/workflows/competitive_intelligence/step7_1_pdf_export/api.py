from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .models import Step71PdfExportRequest, Step71PdfExportResponse
from .step7_1_pdf_export import run_step_7_1_pdf_export


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    # @router.post("/competitive-intelligence/step-7-1-pdf-export/run", response_model=Step71PdfExportResponse)
    def step7_1_pdf_export_run(
        req: Step71PdfExportRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> Step71PdfExportResponse:
        ensure_user_dirs(s, user_id)
        result = run_step_7_1_pdf_export(
            req=req,
            user_root=s.user_dir(user_id),
            work_root=s.user_work_dir(user_id),
        )
        return Step71PdfExportResponse(pdf_export=result)

    return router

