from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .competitor_profile_extraction_v0_6 import extract_competitor_profiles_v0_6
from .models import CompetitorProfileExtractionV06Request, CompetitorProfileExtractionV06Response


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    #@router.post("/competitive/competitors/profiles/v0.6", response_model=CompetitorProfileExtractionV06Response)
    def competitive_competitor_profiles_v0_6(
        req: CompetitorProfileExtractionV06Request,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> CompetitorProfileExtractionV06Response:
        ensure_user_dirs(s, user_id)
        result = extract_competitor_profiles_v0_6(
            competitor_profile_text=req.competitor_profile_text,
            competitor_profile_text_path=req.competitor_profile_text_path,
            product_profile=req.product_profile,
            product_profile_path=req.product_profile_path,
            provider=req.provider,
            max_competitors=req.max_competitors,
            verbose_terminal=req.verbose_terminal,
            user_root=s.user_dir(user_id),
            work_root=s.user_work_dir(user_id),
        )
        return CompetitorProfileExtractionV06Response(competitor_profile_results=result)

    return router

