from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .feature_matrx_gap_analysis_v0_5 import run_feature_matrx_gap_analysis_v0_5
from .models import FeatureMatrxGapAnalysisV05Request, FeatureMatrxGapAnalysisV05Response


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    @router.post('/competitive/feature-matrx-gap/v0.5', response_model=FeatureMatrxGapAnalysisV05Response)
    def competitive_feature_matrx_gap_v0_5(
        req: FeatureMatrxGapAnalysisV05Request,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> FeatureMatrxGapAnalysisV05Response:
        ensure_user_dirs(s, user_id)
        result = run_feature_matrx_gap_analysis_v0_5(
            product_profile=req.product_profile,
            product_profile_path=req.product_profile_path,
            competitor_profile_results=req.competitor_profile_results,
            competitor_profile_results_path=req.competitor_profile_results_path,
            provider=req.provider,
            user_root=s.user_dir(user_id),
            work_root=s.user_work_dir(user_id),
        )
        return FeatureMatrxGapAnalysisV05Response(
            feature_matrix_gap={
                "comparison_matrix": result.comparison_matrix.model_dump(),
                "gaps_and_usps": result.gaps_and_usps.model_dump(),
                "cluster_assignment": [c.model_dump() for c in result.cluster_assignment],
            }
        )

    return router
