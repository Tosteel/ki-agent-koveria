from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .competitor_profile_extraction_quality_gate import run_competitor_profile_extraction_quality_gate
from .models import (
    CompetitorProfileExtractionQualityGateRequest,
    CompetitorProfileExtractionQualityGateResponse,
)


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    #@router.post('/competitive/competitors/profiles/quality-gate', response_model=CompetitorProfileExtractionQualityGateResponse)
    def competitive_competitor_profiles_quality_gate(
        req: CompetitorProfileExtractionQualityGateRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> CompetitorProfileExtractionQualityGateResponse:
        ensure_user_dirs(s, user_id)
        result = run_competitor_profile_extraction_quality_gate(
            competitor_profiles=req.competitor_profiles,
            competitor_profiles_path=req.competitor_profiles_path,
            provider=req.provider,
            enrich_prices=req.enrich_prices,
            max_price_pages_per_competitor=req.max_price_pages_per_competitor,
            require_model_token_hits=req.require_model_token_hits,
            verbose_progress=req.verbose_progress,
            drop_unverified_features=req.drop_unverified_features,
            user_root=s.user_dir(user_id),
            work_root=s.user_work_dir(user_id),
        )
        return CompetitorProfileExtractionQualityGateResponse(competitor_profiles=result)

    return router
