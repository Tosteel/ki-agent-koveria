from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .feature_claim_extraction_v0_2 import extract_feature_claim_profile_v0_2
from .models import (
    CompetitiveFeatureClaimExtractionV2Request,
    CompetitiveFeatureClaimExtractionV2Response,
)


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    #@router.post("/competitive/profile/extract/v0.2", response_model=CompetitiveFeatureClaimExtractionV2Response)
    def competitive_profile_extract_v0_2(
        req: CompetitiveFeatureClaimExtractionV2Request,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> CompetitiveFeatureClaimExtractionV2Response:
        ensure_user_dirs(s, user_id)
        profile = extract_feature_claim_profile_v0_2(
            parsed_doc=req.parsed_doc,
            parsed_doc_path=req.parsed_doc_path,
            provider=req.provider,
            max_context_chars=req.max_context_chars,
            user_root=s.user_dir(user_id),
            work_root=s.user_work_dir(user_id),
        )
        return CompetitiveFeatureClaimExtractionV2Response(product_profile=profile)

    return router
