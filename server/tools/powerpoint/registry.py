from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .models import PptExportRequest, PptExportResponse
from .powerpoint import export_text_pptx


def register(registry: ToolRegistry) -> None:
    def tool_ppt_export(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = PptExportRequest(**args)
        out = (ctx.settings.user_work_dir(ctx.user_id) / req.output_path).resolve()
        result = export_text_pptx(
            out,
            title=req.title,
            text=req.text,
            use_llm_layout=req.use_llm_layout,
            allow_heuristic_fallback=req.allow_heuristic_fallback,
            goal=req.goal or ctx.goal,
            instruction=req.instruction,
            max_slides=req.max_slides,
            max_boxes_per_slide=req.max_boxes_per_slide,
        )
        return PptExportResponse(
            output_path=req.output_path,
            bytes_written=int(result.get("bytes_written") or 0),
            layout_mode=str(result.get("layout_mode") or "heuristic"),
        ).model_dump()

    registry.register("ppt_export", tool_ppt_export, request_model=PptExportRequest)
