from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .distance import distance_check
from .models import DistanceCheckRequest, DistanceCheckResponse


def register(registry: ToolRegistry) -> None:
    def tool_distance_check(_ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = DistanceCheckRequest(**args)
        result = distance_check(
            origin=req.origin,
            destination=req.destination,
            max_distance_km=req.max_distance_km,
        )
        return DistanceCheckResponse(**result).model_dump()

    registry.register(
        "distance_check",
        tool_distance_check,
        request_model=DistanceCheckRequest,
        response_model=DistanceCheckResponse,
    )
