from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .feature_claim_extraction_v0_2 import extract_feature_claim_profile_v0_2
from .models import (
    CompetitiveFeatureClaimExtractionV2Request,
    CompetitiveFeatureClaimExtractionV2Response,
)


TOOL_NAME = "competitive_extract_product_profile_v0_2"


def register(registry: ToolRegistry) -> None:
    def tool_competitive_extract_product_profile_v0_2(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = CompetitiveFeatureClaimExtractionV2Request(**args)
        profile = extract_feature_claim_profile_v0_2(
            parsed_doc=req.parsed_doc,
            parsed_doc_path=req.parsed_doc_path,
            provider=req.provider,
            max_context_chars=req.max_context_chars,
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return CompetitiveFeatureClaimExtractionV2Response(product_profile=profile).model_dump()

    registry.register(
        TOOL_NAME,
        tool_competitive_extract_product_profile_v0_2,
        request_model=CompetitiveFeatureClaimExtractionV2Request,
    )
