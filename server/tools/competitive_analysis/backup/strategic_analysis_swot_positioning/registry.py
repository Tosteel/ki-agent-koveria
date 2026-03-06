from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .models import StrategicAnalysisRequest, StrategicAnalysisResponse
from .strategic_analysis_swot_positioning import run_strategic_analysis


TOOL_NAME = "competitive_strategic_analysis"


def register(registry: ToolRegistry) -> None:
    def tool_competitive_strategic_analysis(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = StrategicAnalysisRequest(**args)
        result = run_strategic_analysis(
            gaps_and_usps=req.gaps_and_usps,
            gaps_and_usps_path=req.gaps_and_usps_path,
            evidences=req.evidences,
            provider=req.provider,
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return StrategicAnalysisResponse(
            swot=result.swot.model_dump(),
            positioning_data=result.positioning_data.model_dump(),
            strategic_implications=[x.model_dump() for x in result.strategic_implications],
            extraction_warnings=result.extraction_warnings,
        ).model_dump()

    registry.register(TOOL_NAME, tool_competitive_strategic_analysis, request_model=StrategicAnalysisRequest)
