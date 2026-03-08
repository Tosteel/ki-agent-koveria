from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .competitor_profile_text_v0_6 import build_competitor_profile_text_v0_6
from .models import CompetitorProfileTextV06Request, CompetitorProfileTextV06Response


TOOL_NAME = "competitor_profile_text_v0_6"


def register(registry: ToolRegistry) -> None:
    def tool_competitor_profile_text_v0_6(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = CompetitorProfileTextV06Request(**args)
        result = build_competitor_profile_text_v0_6(
            competitor_product_results=req.competitor_product_results,
            competitor_product_results_path=req.competitor_product_results_path,
            product_profile=req.product_profile,
            product_profile_path=req.product_profile_path,
            provider=req.provider,
            max_competitors=req.max_competitors,
            brave_enable_research=req.brave_enable_research,
            brave_stream=req.brave_stream,
            brave_language=req.brave_language,
            brave_country=req.brave_country,
            verbose_terminal=req.verbose_terminal,
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return CompetitorProfileTextV06Response(competitor_profile_text=result).model_dump()

    registry.register(TOOL_NAME, tool_competitor_profile_text_v0_6, request_model=CompetitorProfileTextV06Request)
