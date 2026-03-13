from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .models import Step71PdfExportRequest, Step71PdfExportResponse
from .step7_1_pdf_export import run_step_7_1_pdf_export


TOOL_NAME = "step7_1_pdf_export"


def register(registry: ToolRegistry) -> None:
    def tool_step7_1_pdf_export(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = Step71PdfExportRequest(**args)
        result = run_step_7_1_pdf_export(
            req=req,
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return Step71PdfExportResponse(pdf_export=result).model_dump()

    registry.register(TOOL_NAME, tool_step7_1_pdf_export, request_model=Step71PdfExportRequest)

