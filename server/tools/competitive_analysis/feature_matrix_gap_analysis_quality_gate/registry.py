from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .feature_matrix_gap_analysis_quality_gate import run_feature_matrix_gap_analysis_quality_gate
from .models import (
    FeatureMatrixGapAnalysisQualityGateRequest,
    FeatureMatrixGapAnalysisQualityGateResponse,
)


TOOL_NAME = "competitive_feature_matrix_gap_analysis_quality_gate"


def register(registry: ToolRegistry) -> None:
    def tool_feature_matrix_gap_analysis_quality_gate(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = FeatureMatrixGapAnalysisQualityGateRequest(**args)
        cleaned, report = run_feature_matrix_gap_analysis_quality_gate(
            feature_matrix_gap=req.feature_matrix_gap,
            feature_matrix_gap_path=req.feature_matrix_gap_path,
            provider=req.provider,
            max_missing_features_per_competitor=req.max_missing_features_per_competitor,
            max_urls_per_feature=req.max_urls_per_feature,
            max_llm_calls=req.max_llm_calls,
            min_confidence=req.min_confidence,
            verbose_progress=req.verbose_progress,
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return FeatureMatrixGapAnalysisQualityGateResponse(
            comparison_matrix=cleaned.get("comparison_matrix") if isinstance(cleaned.get("comparison_matrix"), dict) else {},
            gaps_and_usps=cleaned.get("gaps_and_usps") if isinstance(cleaned.get("gaps_and_usps"), dict) else {},
            cluster_assignment=[x for x in (cleaned.get("cluster_assignment") or []) if isinstance(x, dict)],
            extraction_warnings=[str(x) for x in (cleaned.get("extraction_warnings") or [])],
            quality_report=report,
        ).model_dump()

    registry.register(TOOL_NAME, tool_feature_matrix_gap_analysis_quality_gate, request_model=FeatureMatrixGapAnalysisQualityGateRequest)
