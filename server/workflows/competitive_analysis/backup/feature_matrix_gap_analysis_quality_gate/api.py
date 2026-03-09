from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .feature_matrix_gap_analysis_quality_gate import run_feature_matrix_gap_analysis_quality_gate
from .models import (
    FeatureMatrixGapAnalysisQualityGateRequest,
    FeatureMatrixGapAnalysisQualityGateResponse,
)


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    @router.post('/competitive/feature-matrix-gap/quality-gate', response_model=FeatureMatrixGapAnalysisQualityGateResponse)
    def competitive_feature_matrix_gap_quality_gate(
        req: FeatureMatrixGapAnalysisQualityGateRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> FeatureMatrixGapAnalysisQualityGateResponse:
        ensure_user_dirs(s, user_id)
        cleaned, report = run_feature_matrix_gap_analysis_quality_gate(
            feature_matrix_gap=req.feature_matrix_gap,
            feature_matrix_gap_path=req.feature_matrix_gap_path,
            provider=req.provider,
            max_missing_features_per_competitor=req.max_missing_features_per_competitor,
            max_urls_per_feature=req.max_urls_per_feature,
            max_llm_calls=req.max_llm_calls,
            min_confidence=req.min_confidence,
            verbose_progress=req.verbose_progress,
            user_root=s.user_dir(user_id),
            work_root=s.user_work_dir(user_id),
        )
        return FeatureMatrixGapAnalysisQualityGateResponse(
            comparison_matrix=cleaned.get("comparison_matrix") if isinstance(cleaned.get("comparison_matrix"), dict) else {},
            gaps_and_usps=cleaned.get("gaps_and_usps") if isinstance(cleaned.get("gaps_and_usps"), dict) else {},
            cluster_assignment=[x for x in (cleaned.get("cluster_assignment") or []) if isinstance(x, dict)],
            extraction_warnings=[str(x) for x in (cleaned.get("extraction_warnings") or [])],
            quality_report=report,
        )

    return router
