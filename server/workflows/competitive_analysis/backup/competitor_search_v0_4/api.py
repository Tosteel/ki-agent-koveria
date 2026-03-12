from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .competitor_search_v0_4 import search_competitors_v0_4
from .models import CompetitorSearchV04Request, CompetitorSearchV04Response


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    #@router.post("/competitive/competitors/search/v0.4", response_model=CompetitorSearchV04Response)
    def competitive_company_search_v0_4(
        req: CompetitorSearchV04Request,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> CompetitorSearchV04Response:
        ensure_user_dirs(s, user_id)
        result = search_competitors_v0_4(
            analysis_plan=req.analysis_plan,
            analysis_plan_path=req.analysis_plan_path,
            product_profile=req.product_profile,
            product_profile_path=req.product_profile_path,
            provider=req.provider,
            max_queries=req.max_queries,
            per_query_results=req.per_query_results,
            max_candidates_to_check=req.max_candidates_to_check,
            use_llm_feature_enrichment=req.use_llm_feature_enrichment,
            llm_min_relevance_for_enrichment=req.llm_min_relevance_for_enrichment,
            include_page_fetch=req.include_page_fetch,
            page_fetch_timeout_s=req.page_fetch_timeout_s,
            page_fetch_max_chars=req.page_fetch_max_chars,
            verbose_terminal=req.verbose_terminal,
            verbose_search_hits=req.verbose_search_hits,
            user_root=s.user_dir(user_id),
            work_root=s.user_work_dir(user_id),
        )
        return CompetitorSearchV04Response(competitor_search_results=result)

    return router
