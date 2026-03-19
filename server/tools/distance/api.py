from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .distance import distance_check
from .models import DistanceCheckRequest, DistanceCheckResponse


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    @router.post("/tools/distance/check", response_model=DistanceCheckResponse)
    def check_route(
        req: DistanceCheckRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> DistanceCheckResponse:
        ensure_user_dirs(s, user_id)
        return DistanceCheckResponse(
            **distance_check(
                origin=req.origin,
                destination=req.destination,
                max_distance_km=req.max_distance_km,
            )
        )

    return router
