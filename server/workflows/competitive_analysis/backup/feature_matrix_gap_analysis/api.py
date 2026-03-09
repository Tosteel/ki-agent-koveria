from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .feature_matrix_gap_analysis import run_feature_matrix_gap_analysis
from .models import FeatureMatrixGapAnalysisRequest, FeatureMatrixGapAnalysisResponse


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    @router.post('/competitive/feature-matrix-gap', response_model=FeatureMatrixGapAnalysisResponse)
    def competitive_feature_matrix_gap(
        req: FeatureMatrixGapAnalysisRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> FeatureMatrixGapAnalysisResponse:
        ensure_user_dirs(s, user_id)
        result = run_feature_matrix_gap_analysis(
            product_profile=req.product_profile,
            product_profile_path=req.product_profile_path,
            competitor_profiles=req.competitor_profiles,
            competitor_profiles_path=req.competitor_profiles_path,
            provider=req.provider,
            user_root=s.user_dir(user_id),
            work_root=s.user_work_dir(user_id),
        )
        return FeatureMatrixGapAnalysisResponse(
            comparison_matrix=result.comparison_matrix.model_dump(),
            gaps_and_usps=result.gaps_and_usps.model_dump(),
            cluster_assignment=[c.model_dump() for c in result.cluster_assignment],
        )

    return router
