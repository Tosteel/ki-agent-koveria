from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .models import CompetitiveQualityGateRequest, CompetitiveQualityGateResponse
from .quality_gate import run_competitive_quality_gate


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    #@router.post("/competitive/quality-gate", response_model=CompetitiveQualityGateResponse)
    def competitive_quality_gate(
        req: CompetitiveQualityGateRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> CompetitiveQualityGateResponse:
        ensure_user_dirs(s, user_id)
        out_artifact, report = run_competitive_quality_gate(
            artifact=req.artifact,
            artifact_path=req.artifact_path,
            step=req.step,
            mode=req.mode,
            provider=req.provider,
            max_context_chars=req.max_context_chars,
            user_root=s.user_dir(user_id),
            work_root=s.user_work_dir(user_id),
        )
        return CompetitiveQualityGateResponse(
            artifact=out_artifact,
            quality_report=report,
        )

    return router

