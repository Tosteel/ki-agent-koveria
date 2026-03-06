from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .models import CompetitiveQualityGateRequest, CompetitiveQualityGateResponse
from .quality_gate import run_competitive_quality_gate


TOOL_NAME = "competitive_quality_gate"


def register(registry: ToolRegistry) -> None:
    def tool_competitive_quality_gate(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = CompetitiveQualityGateRequest(**args)
        out_artifact, report = run_competitive_quality_gate(
            artifact=req.artifact,
            artifact_path=req.artifact_path,
            step=req.step,
            mode=req.mode,
            provider=req.provider,
            max_context_chars=req.max_context_chars,
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return CompetitiveQualityGateResponse(
            artifact=out_artifact,
            quality_report=report,
        ).model_dump()

    registry.register(
        TOOL_NAME,
        tool_competitive_quality_gate,
        request_model=CompetitiveQualityGateRequest,
    )

