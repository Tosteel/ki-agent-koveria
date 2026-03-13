from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .models import Step61FinalReportRequest, Step61FinalReportResponse
from .step6_1_final_report import run_step_6_1_final_report


TOOL_NAME = "step6_1_final_report"


def register(registry: ToolRegistry) -> None:
    def tool_step6_1_final_report(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = Step61FinalReportRequest(**args)
        result = run_step_6_1_final_report(
            req=req,
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return Step61FinalReportResponse(final_report=result).model_dump()

    registry.register(TOOL_NAME, tool_step6_1_final_report, request_model=Step61FinalReportRequest)

