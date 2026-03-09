from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .feature_matrix_gap_analysis_v0_5 import run_feature_matrix_gap_analysis_v0_5
from .models import FeatureMatrixGapAnalysisV05Request, FeatureMatrixGapAnalysisV05Response


TOOL_NAME = "feature_matrix_gap_analysis_v0_5"


def register(registry: ToolRegistry) -> None:
    def tool_feature_matrix_gap_analysis_v0_5(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = FeatureMatrixGapAnalysisV05Request(**args)
        result = run_feature_matrix_gap_analysis_v0_5(
            product_profile=req.product_profile,
            product_profile_path=req.product_profile_path,
            competitor_profile_results=req.competitor_profile_results,
            competitor_profile_results_path=req.competitor_profile_results_path,
            provider=req.provider,
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return FeatureMatrixGapAnalysisV05Response(
            feature_matrix_gap={
                "comparison_matrix": result.comparison_matrix.model_dump(),
                "gaps_and_usps": result.gaps_and_usps.model_dump(),
                "cluster_assignment": [c.model_dump() for c in result.cluster_assignment],
            }
        ).model_dump()

    registry.register(TOOL_NAME, tool_feature_matrix_gap_analysis_v0_5, request_model=FeatureMatrixGapAnalysisV05Request)
