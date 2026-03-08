from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .competitor_profile_extraction_v0_6 import extract_competitor_profiles_v0_6
from .models import CompetitorProfileExtractionV06Request, CompetitorProfileExtractionV06Response


TOOL_NAME = "competitor_profile_extraction_v0_6"


def register(registry: ToolRegistry) -> None:
    def tool_competitor_profile_extraction_v0_6(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = CompetitorProfileExtractionV06Request(**args)
        result = extract_competitor_profiles_v0_6(
            competitor_profile_text=req.competitor_profile_text,
            competitor_profile_text_path=req.competitor_profile_text_path,
            product_profile=req.product_profile,
            product_profile_path=req.product_profile_path,
            provider=req.provider,
            max_competitors=req.max_competitors,
            verbose_terminal=req.verbose_terminal,
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return CompetitorProfileExtractionV06Response(competitor_profile_results=result).model_dump()

    registry.register(TOOL_NAME, tool_competitor_profile_extraction_v0_6, request_model=CompetitorProfileExtractionV06Request)

