from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .models import OfferflowStep2Request, OfferflowStep2Response
from .offerflow_step import run_step_2

security = HTTPBearer(auto_error=False)


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    @router.post("/offerflow/step-2/run", response_model=OfferflowStep2Response)
    def run_offerflow_step_2(
        req: OfferflowStep2Request,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> OfferflowStep2Response:
        ensure_user_dirs(s, user_id)
        api_key = credentials.credentials if credentials else ""
        result = run_step_2(s=s, user_id=user_id, api_key=api_key, req=req)
        return OfferflowStep2Response(**result)

    return router
