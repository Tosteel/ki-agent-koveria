from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .models import ProfessionalPdfPublisherRequest, ProfessionalPdfPublisherResponse
from .professional_pdf_publischer_v0_6 import publish_competition_pdf_v0_6


TOOL_NAME = "professional_pdf_publischer_v0_6"


def register(registry: ToolRegistry) -> None:
    def tool_publish_pdf(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = ProfessionalPdfPublisherRequest(**args)
        result = publish_competition_pdf_v0_6(
            final_report=req.final_report,
            final_report_path=req.final_report_path,
            output_path=req.output_path,
            logo_path=req.logo_path,
            report_config_path=req.report_config_path,
            chart_paths=req.chart_paths,
            include_render_log=req.include_render_log,
            render_log_path=req.render_log_path,
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return ProfessionalPdfPublisherResponse(**result).model_dump()

    registry.register(TOOL_NAME, tool_publish_pdf, request_model=ProfessionalPdfPublisherRequest)
