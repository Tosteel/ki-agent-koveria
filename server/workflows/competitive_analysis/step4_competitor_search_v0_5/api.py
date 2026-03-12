from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .competitor_search_v0_5 import search_competitors_v0_5
from .models import CompetitorSearchV05Request, CompetitorSearchV05Response


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    #@router.post("/competitive/competitors/search/v0.5", response_model=CompetitorSearchV05Response)
    def competitive_company_search_v0_5(
        req: CompetitorSearchV05Request,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> CompetitorSearchV05Response:
        ensure_user_dirs(s, user_id)
        result = search_competitors_v0_5(
            analysis_plan=req.analysis_plan,
            analysis_plan_path=req.analysis_plan_path,
            provider=req.provider,
            max_queries=req.max_queries,
            per_query_results=req.per_query_results,
            max_candidates_to_check=req.max_candidates_to_check,
            verbose_terminal=req.verbose_terminal,
            verbose_search_hits=req.verbose_search_hits,
            user_root=s.user_dir(user_id),
            work_root=s.user_work_dir(user_id),
        )
        return CompetitorSearchV05Response(competitor_search_results=result)

    return router
