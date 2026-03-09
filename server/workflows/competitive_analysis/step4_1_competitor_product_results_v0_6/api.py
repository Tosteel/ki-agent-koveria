from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .competitor_product_results_v0_6 import build_competitor_product_results_v0_6
from .models import CompetitorProductResultsV06Request, CompetitorProductResultsV06Response


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    @router.post("/competitive/competitors/products/v0.6", response_model=CompetitorProductResultsV06Response)
    def competitive_competitor_product_results_v0_6(
        req: CompetitorProductResultsV06Request,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> CompetitorProductResultsV06Response:
        ensure_user_dirs(s, user_id)
        result = build_competitor_product_results_v0_6(
            competitor_search_results=req.competitor_search_results,
            competitor_search_results_path=req.competitor_search_results_path,
            provider=req.provider,
            top_n=req.top_n,
            verbose_terminal=req.verbose_terminal,
            user_root=s.user_dir(user_id),
            work_root=s.user_work_dir(user_id),
        )
        return CompetitorProductResultsV06Response(competitor_product_results=result)

    return router

