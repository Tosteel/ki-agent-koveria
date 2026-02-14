from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .models import PdfExportRequest, PdfExportResponse
from .pdf import export_text_pdf


def register(registry: ToolRegistry) -> None:
    def tool_pdf_export(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = PdfExportRequest(**args)
        out = (ctx.settings.user_work_dir(ctx.user_id) / req.output_path).resolve()
        size = export_text_pdf(out, title=req.title, text=req.text)
        return PdfExportResponse(output_path=req.output_path, bytes_written=size).model_dump()

    registry.register("pdf_export", tool_pdf_export, request_model=PdfExportRequest)
