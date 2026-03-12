from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .competitor_identification_quality_gate import run_competitor_identification_quality_gate
from .models import (
    CompetitorIdentificationQualityGateRequest,
    CompetitorIdentificationQualityGateResponse,
)


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    #@router.post('/competitive/competitors/quality-gate', response_model=CompetitorIdentificationQualityGateResponse)
    def competitive_competitors_quality_gate(
        req: CompetitorIdentificationQualityGateRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> CompetitorIdentificationQualityGateResponse:
        ensure_user_dirs(s, user_id)
        cleaned, report = run_competitor_identification_quality_gate(
            competitor_list=req.competitor_list,
            competitor_list_path=req.competitor_list_path,
            product_profile=req.product_profile,
            product_profile_path=req.product_profile_path,
            provider=req.provider,
            min_relevance_score=req.min_relevance_score,
            drop_generic_listing_pages=req.drop_generic_listing_pages,
            drop_weak_unknown_candidates=req.drop_weak_unknown_candidates,
            drop_manufacturer_nodes_without_model_signal=req.drop_manufacturer_nodes_without_model_signal,
            dedupe_by_name_and_domain=req.dedupe_by_name_and_domain,
            enable_llm_snippet_validation=req.enable_llm_snippet_validation,
            llm_min_keep_confidence=req.llm_min_keep_confidence,
            max_llm_checks=req.max_llm_checks,
            user_root=s.user_dir(user_id),
            work_root=s.user_work_dir(user_id),
        )
        return CompetitorIdentificationQualityGateResponse(competitor_list=cleaned, quality_report=report)

    return router
