from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .final_report_generator_v0_6 import build_final_report_v0_6
from .models import FinalReportRequest, FinalReportResponse


TOOL_NAME = "final_report_generator_v0_6"


def register(registry: ToolRegistry) -> None:
    def tool_final_report_v0_6(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = FinalReportRequest(**args)
        result = build_final_report_v0_6(
            artifacts=req.artifacts,
            artifact_paths=req.artifact_paths,
            provider=req.provider,
            max_chars_per_artifact=req.max_chars_per_artifact,
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return FinalReportResponse(
            final_report=result.final_report.model_dump(),
            validation=result.validation.model_dump(),
            report_context=result.report_context,
            artifact_chunks=[c.model_dump() for c in result.artifact_chunks],
            extraction_warnings=result.extraction_warnings,
        ).model_dump()

    registry.register(TOOL_NAME, tool_final_report_v0_6, request_model=FinalReportRequest)
