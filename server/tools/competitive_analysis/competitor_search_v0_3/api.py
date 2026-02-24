from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .competitor_search_v0_3 import search_competitors_v0_3
from .models import CompetitorSearchRequest, CompetitorSearchResponse


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    @router.post('/competitive/competitors/search/v0.3', response_model=CompetitorSearchResponse)
    def competitive_company_search_v0_3(
        req: CompetitorSearchRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> CompetitorSearchResponse:
        ensure_user_dirs(s, user_id)
        result = search_competitors_v0_3(
            analysis_plan=req.analysis_plan,
            analysis_plan_path=req.analysis_plan_path,
            product_competitors=req.product_competitors,
            product_competitors_path=req.product_competitors_path,
            provider=req.provider,
            max_queries=req.max_queries,
            per_query_results=req.per_query_results,
            shortlist_size=req.shortlist_size,
            min_relevance_score=req.min_relevance_score,
            verbose_terminal=req.verbose_terminal,
            verbose_search_hits=req.verbose_search_hits,
            user_root=s.user_dir(user_id),
            work_root=s.user_work_dir(user_id),
        )
        return CompetitorSearchResponse(competitor_search_results=result)

    return router
