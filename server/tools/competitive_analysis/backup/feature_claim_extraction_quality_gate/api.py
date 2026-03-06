from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .feature_claim_extraction_quality_gate import run_feature_claim_extraction_quality_gate
from .models import (
    FeatureClaimExtractionQualityGateRequest,
    FeatureClaimExtractionQualityGateResponse,
)


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    @router.post("/competitive/profile/quality-gate", response_model=FeatureClaimExtractionQualityGateResponse)
    def competitive_profile_quality_gate(
        req: FeatureClaimExtractionQualityGateRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> FeatureClaimExtractionQualityGateResponse:
        ensure_user_dirs(s, user_id)
        cleaned_profile, quality_report = run_feature_claim_extraction_quality_gate(
            product_profile=req.product_profile,
            product_profile_path=req.product_profile_path,
            provider=req.provider,
            max_context_chars=req.max_context_chars,
            remove_nonsensical_features=req.remove_nonsensical_features,
            repair_feature_names=req.repair_feature_names,
            min_alpha_chars=req.min_alpha_chars,
            max_feature_name_length=req.max_feature_name_length,
            allow_llm_fallback=req.allow_llm_fallback,
            user_root=s.user_dir(user_id),
            work_root=s.user_work_dir(user_id),
        )
        return FeatureClaimExtractionQualityGateResponse(
            product_profile=cleaned_profile,
            quality_report=quality_report,
        )

    return router
