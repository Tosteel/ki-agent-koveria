from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .models import Step31MatrixRequest, Step31MatrixResponse
from .step3_1_matrix import run_step_3_1_matrix


TOOL_NAME = "step3_1_matrix"


def register(registry: ToolRegistry) -> None:
    def tool_step3_1_matrix(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = Step31MatrixRequest(**args)
        result = run_step_3_1_matrix(
            req=req,
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return Step31MatrixResponse(matrix=result).model_dump()

    registry.register(TOOL_NAME, tool_step3_1_matrix, request_model=Step31MatrixRequest)

