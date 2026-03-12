from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .models import StrategicAnalysisRequest, StrategicAnalysisResponse
from .strategic_analysis_swot_positioning import run_strategic_analysis


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    #@router.post('/competitive/strategic-analysis', response_model=StrategicAnalysisResponse)
    def competitive_strategic_analysis(
        req: StrategicAnalysisRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> StrategicAnalysisResponse:
        ensure_user_dirs(s, user_id)
        result = run_strategic_analysis(
            gaps_and_usps=req.gaps_and_usps,
            gaps_and_usps_path=req.gaps_and_usps_path,
            evidences=req.evidences,
            provider=req.provider,
            user_root=s.user_dir(user_id),
            work_root=s.user_work_dir(user_id),
        )
        return StrategicAnalysisResponse(
            swot=result.swot.model_dump(),
            positioning_data=result.positioning_data.model_dump(),
            strategic_implications=[x.model_dump() for x in result.strategic_implications],
            extraction_warnings=result.extraction_warnings,
        )

    return router
