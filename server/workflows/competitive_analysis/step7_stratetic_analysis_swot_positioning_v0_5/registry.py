from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .models import StrateticAnalysisSwotPositioningV05Request, StrateticAnalysisSwotPositioningV05Response
from .stratetic_analysis_swot_positioning_v0_5 import run_stratetic_analysis_swot_positioning_v0_5


TOOL_NAME = "stratetic_analysis_swot_positioning_v0_5"


def register(registry: ToolRegistry) -> None:
    def tool_stratetic_analysis_swot_positioning_v0_5(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = StrateticAnalysisSwotPositioningV05Request(**args)
        result = run_stratetic_analysis_swot_positioning_v0_5(
            feature_matrix_gap=req.feature_matrix_gap,
            feature_matrix_gap_path=req.feature_matrix_gap_path,
            comparison_matrix=req.comparison_matrix,
            gaps_and_usps=req.gaps_and_usps,
            evidences=req.evidences,
            provider=req.provider,
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return StrateticAnalysisSwotPositioningV05Response(
            swot=result.swot.model_dump(),
            positioning_data=result.positioning_data.model_dump(),
            strategic_implications=[x.model_dump() for x in result.strategic_implications],
            extraction_warnings=result.extraction_warnings,
        ).model_dump()

    registry.register(TOOL_NAME, tool_stratetic_analysis_swot_positioning_v0_5, request_model=StrateticAnalysisSwotPositioningV05Request)
