from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .feature_matrix_gap_analysis import run_feature_matrix_gap_analysis
from .models import FeatureMatrixGapAnalysisRequest, FeatureMatrixGapAnalysisResponse


TOOL_NAME = "competitive_feature_matrix_gap_analysis"


def register(registry: ToolRegistry) -> None:
    def tool_feature_matrix_gap_analysis(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = FeatureMatrixGapAnalysisRequest(**args)
        result = run_feature_matrix_gap_analysis(
            product_profile=req.product_profile,
            product_profile_path=req.product_profile_path,
            competitor_profiles=req.competitor_profiles,
            competitor_profiles_path=req.competitor_profiles_path,
            provider=req.provider,
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return FeatureMatrixGapAnalysisResponse(
            comparison_matrix=result.comparison_matrix.model_dump(),
            gaps_and_usps=result.gaps_and_usps.model_dump(),
            cluster_assignment=[c.model_dump() for c in result.cluster_assignment],
        ).model_dump()

    registry.register(TOOL_NAME, tool_feature_matrix_gap_analysis, request_model=FeatureMatrixGapAnalysisRequest)
