from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .final_report_generator_v0_6 import build_final_report_v0_6
from .models import FinalReportRequest, FinalReportResponse


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    #@router.post('/competitive/final-report/v0.6', response_model=FinalReportResponse)
    def competitive_generate_final_report_v0_6(
        req: FinalReportRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> FinalReportResponse:
        ensure_user_dirs(s, user_id)
        result = build_final_report_v0_6(
            artifacts=req.artifacts,
            artifact_paths=req.artifact_paths,
            provider=req.provider,
            max_chars_per_artifact=req.max_chars_per_artifact,
            user_root=s.user_dir(user_id),
            work_root=s.user_work_dir(user_id),
        )
        return FinalReportResponse(
            final_report=result.final_report.model_dump(),
            validation=result.validation.model_dump(),
            report_context=result.report_context,
            artifact_chunks=[c.model_dump() for c in result.artifact_chunks],
            extraction_warnings=result.extraction_warnings,
        )

    return router
