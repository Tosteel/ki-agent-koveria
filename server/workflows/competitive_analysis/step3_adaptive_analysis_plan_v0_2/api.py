from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .adaptive_analysis_plan_v0_2 import generate_adaptive_analysis_plan_v0_2
from .models import AdaptiveAnalysisPlanRequest, AdaptiveAnalysisPlanResponse


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    #@router.post('/competitive/analysis/plan/v0.2', response_model=AdaptiveAnalysisPlanResponse)
    def competitive_analysis_plan_v0_2(
        req: AdaptiveAnalysisPlanRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> AdaptiveAnalysisPlanResponse:
        ensure_user_dirs(s, user_id)
        plan = generate_adaptive_analysis_plan_v0_2(
            product_profile=req.product_profile,
            product_profile_path=req.product_profile_path,
            provider=req.provider,
            max_context_chars=req.max_context_chars,
            user_root=s.user_dir(user_id),
            work_root=s.user_work_dir(user_id),
        )
        return AdaptiveAnalysisPlanResponse(analysis_plan=plan)

    return router
