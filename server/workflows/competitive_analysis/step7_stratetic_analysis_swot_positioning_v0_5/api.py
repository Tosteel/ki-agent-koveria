from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .models import StrateticAnalysisSwotPositioningV05Request, StrateticAnalysisSwotPositioningV05Response
from .stratetic_analysis_swot_positioning_v0_5 import run_stratetic_analysis_swot_positioning_v0_5


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    #@router.post('/competitive/strategic-analysis/v0.5', response_model=StrateticAnalysisSwotPositioningV05Response)
    def competitive_stratetic_analysis_swot_positioning_v0_5(
        req: StrateticAnalysisSwotPositioningV05Request,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> StrateticAnalysisSwotPositioningV05Response:
        ensure_user_dirs(s, user_id)
        result = run_stratetic_analysis_swot_positioning_v0_5(
            feature_matrix_gap=req.feature_matrix_gap,
            feature_matrix_gap_path=req.feature_matrix_gap_path,
            comparison_matrix=req.comparison_matrix,
            gaps_and_usps=req.gaps_and_usps,
            evidences=req.evidences,
            provider=req.provider,
            user_root=s.user_dir(user_id),
            work_root=s.user_work_dir(user_id),
        )
        return StrateticAnalysisSwotPositioningV05Response(
            swot=result.swot.model_dump(),
            positioning_data=result.positioning_data.model_dump(),
            strategic_implications=[x.model_dump() for x in result.strategic_implications],
            extraction_warnings=result.extraction_warnings,
        )

    return router
