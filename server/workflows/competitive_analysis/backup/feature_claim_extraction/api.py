from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .feature_claim_extraction import extract_feature_claim_profile
from .models import (
    CompetitiveFeatureClaimExtractionRequest,
    CompetitiveFeatureClaimExtractionResponse,
)


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    #@router.post("/competitive/profile/extract", response_model=CompetitiveFeatureClaimExtractionResponse)
    def competitive_profile_extract(
        req: CompetitiveFeatureClaimExtractionRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> CompetitiveFeatureClaimExtractionResponse:
        ensure_user_dirs(s, user_id)
        profile = extract_feature_claim_profile(
            parsed_doc=req.parsed_doc,
            parsed_doc_path=req.parsed_doc_path,
            provider=req.provider,
            max_context_chars=req.max_context_chars,
            user_root=s.user_dir(user_id),
            work_root=s.user_work_dir(user_id),
        )
        return CompetitiveFeatureClaimExtractionResponse(product_profile=profile)

    return router
