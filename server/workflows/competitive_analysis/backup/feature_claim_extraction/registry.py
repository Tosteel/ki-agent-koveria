from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .feature_claim_extraction import extract_feature_claim_profile
from .models import (
    CompetitiveFeatureClaimExtractionRequest,
    CompetitiveFeatureClaimExtractionResponse,
)


TOOL_NAME = "competitive_extract_product_profile"


def register(registry: ToolRegistry) -> None:
    def tool_competitive_extract_product_profile(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = CompetitiveFeatureClaimExtractionRequest(**args)
        profile = extract_feature_claim_profile(
            parsed_doc=req.parsed_doc,
            parsed_doc_path=req.parsed_doc_path,
            provider=req.provider,
            max_context_chars=req.max_context_chars,
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return CompetitiveFeatureClaimExtractionResponse(product_profile=profile).model_dump()

    registry.register(
        TOOL_NAME,
        tool_competitive_extract_product_profile,
        request_model=CompetitiveFeatureClaimExtractionRequest,
    )
