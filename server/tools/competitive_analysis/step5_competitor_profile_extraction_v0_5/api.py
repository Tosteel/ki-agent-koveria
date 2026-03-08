from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .competitor_profile_extraction_v0_5 import extract_competitor_profiles_v0_5
from .models import CompetitorProfileExtractionV05Request, CompetitorProfileExtractionV05Response


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    @router.post('/competitive/competitors/profiles/v0.5', response_model=CompetitorProfileExtractionV05Response)
    def competitive_competitor_profiles_v0_5(
        req: CompetitorProfileExtractionV05Request,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> CompetitorProfileExtractionV05Response:
        ensure_user_dirs(s, user_id)
        result = extract_competitor_profiles_v0_5(
            competitor_search_results=req.competitor_search_results,
            competitor_search_results_path=req.competitor_search_results_path,
            product_profile=req.product_profile,
            product_profile_path=req.product_profile_path,
            provider=req.provider,
            max_competitors=req.max_competitors,
            exclude_same_manufacturer=req.exclude_same_manufacturer,
            top_n_by_relevance=req.top_n_by_relevance,
            include_page_fetch=req.include_page_fetch,
            page_fetch_timeout_s=req.page_fetch_timeout_s,
            page_fetch_max_chars=req.page_fetch_max_chars,
            verbose_terminal=req.verbose_terminal,
            user_root=s.user_dir(user_id),
            work_root=s.user_work_dir(user_id),
        )
        return CompetitorProfileExtractionV05Response(competitor_profile_results=result)

    return router
