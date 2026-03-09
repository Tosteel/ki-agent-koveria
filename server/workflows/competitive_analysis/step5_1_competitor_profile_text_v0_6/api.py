from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .competitor_profile_text_v0_6 import build_competitor_profile_text_v0_6
from .models import CompetitorProfileTextV06Request, CompetitorProfileTextV06Response


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    @router.post("/competitive/competitors/profiles/text/v0.6", response_model=CompetitorProfileTextV06Response)
    def competitive_competitor_profile_text_v0_6(
        req: CompetitorProfileTextV06Request,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> CompetitorProfileTextV06Response:
        ensure_user_dirs(s, user_id)
        result = build_competitor_profile_text_v0_6(
            competitor_product_results=req.competitor_product_results,
            competitor_product_results_path=req.competitor_product_results_path,
            product_profile=req.product_profile,
            product_profile_path=req.product_profile_path,
            provider=req.provider,
            max_competitors=req.max_competitors,
            brave_enable_research=req.brave_enable_research,
            brave_stream=req.brave_stream,
            brave_language=req.brave_language,
            brave_country=req.brave_country,
            verbose_terminal=req.verbose_terminal,
            user_root=s.user_dir(user_id),
            work_root=s.user_work_dir(user_id),
        )
        return CompetitorProfileTextV06Response(competitor_profile_text=result)

    return router
