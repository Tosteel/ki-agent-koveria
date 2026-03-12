from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .competitor_identification_v0_2 import identify_competitors_v0_2
from .models import CompetitorIdentificationRequest, CompetitorIdentificationResponse


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    #@router.post('/competitive/competitors/identify/v0.2', response_model=CompetitorIdentificationResponse)
    def competitive_identify_v0_2(
        req: CompetitorIdentificationRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> CompetitorIdentificationResponse:
        ensure_user_dirs(s, user_id)
        result = identify_competitors_v0_2(
            analysis_plan=req.analysis_plan,
            analysis_plan_path=req.analysis_plan_path,
            product_profile=req.product_profile,
            product_profile_path=req.product_profile_path,
            provider=req.provider,
            max_queries=req.max_queries,
            per_query_results=req.per_query_results,
            shortlist_size=req.shortlist_size,
            min_relevance_score=req.min_relevance_score,
            min_similarity_score=req.min_similarity_score,
            url_priority_weight=req.url_priority_weight,
            datasheet_priority_weight=req.datasheet_priority_weight,
            exclude_below_threshold=req.exclude_below_threshold,
            exhaust_all_attempts=req.exhaust_all_attempts,
            verbose_terminal=req.verbose_terminal,
            user_root=s.user_dir(user_id),
            work_root=s.user_work_dir(user_id),
        )
        return CompetitorIdentificationResponse(competitor_list=result)

    return router
