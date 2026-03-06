from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .competitor_profile_extraction_quality_gate import run_competitor_profile_extraction_quality_gate
from .models import (
    CompetitorProfileExtractionQualityGateRequest,
)


TOOL_NAME = "competitor_profile_extraction_quality_gate"


def register(registry: ToolRegistry) -> None:
    def tool_competitor_profile_extraction_quality_gate(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = CompetitorProfileExtractionQualityGateRequest(**args)
        result = run_competitor_profile_extraction_quality_gate(
            competitor_profiles=req.competitor_profiles,
            competitor_profiles_path=req.competitor_profiles_path,
            provider=req.provider,
            enrich_prices=req.enrich_prices,
            max_price_pages_per_competitor=req.max_price_pages_per_competitor,
            require_model_token_hits=req.require_model_token_hits,
            verbose_progress=req.verbose_progress,
            drop_unverified_features=req.drop_unverified_features,
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return result.model_dump()

    registry.register(TOOL_NAME, tool_competitor_profile_extraction_quality_gate, request_model=CompetitorProfileExtractionQualityGateRequest)
