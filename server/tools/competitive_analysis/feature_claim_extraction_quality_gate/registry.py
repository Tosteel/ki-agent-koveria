from __future__ import annotations

from typing import Any, Dict

from server.agent.tool_registry import ToolContext, ToolRegistry

from .feature_claim_extraction_quality_gate import run_feature_claim_extraction_quality_gate
from .models import (
    FeatureClaimExtractionQualityGateRequest,
    FeatureClaimExtractionQualityGateResponse,
)


TOOL_NAME = "feature_claim_extraction_quality_gate"


def register(registry: ToolRegistry) -> None:
    def tool_feature_claim_extraction_quality_gate(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = FeatureClaimExtractionQualityGateRequest(**args)
        cleaned_profile, quality_report = run_feature_claim_extraction_quality_gate(
            product_profile=req.product_profile,
            product_profile_path=req.product_profile_path,
            provider=req.provider,
            max_context_chars=req.max_context_chars,
            remove_nonsensical_features=req.remove_nonsensical_features,
            repair_feature_names=req.repair_feature_names,
            min_alpha_chars=req.min_alpha_chars,
            max_feature_name_length=req.max_feature_name_length,
            user_root=ctx.settings.user_dir(ctx.user_id),
            work_root=ctx.settings.user_work_dir(ctx.user_id),
        )
        return FeatureClaimExtractionQualityGateResponse(
            product_profile=cleaned_profile,
            quality_report=quality_report,
        ).model_dump()

    registry.register(
        TOOL_NAME,
        tool_feature_claim_extraction_quality_gate,
        request_model=FeatureClaimExtractionQualityGateRequest,
    )
