from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .models import OfferflowStep1Request, OfferflowStep1Response
from .offerflow_step import run_step_1

security = HTTPBearer(auto_error=False)


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    @router.post("/offerflow/step-1/run", response_model=OfferflowStep1Response)
    def run_offerflow_step_1(
        req: OfferflowStep1Request,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> OfferflowStep1Response:
        ensure_user_dirs(s, user_id)
        api_key = credentials.credentials if credentials else ""
        result = run_step_1(s=s, user_id=user_id, api_key=api_key, req=req)
        return OfferflowStep1Response(**result)

    return router
