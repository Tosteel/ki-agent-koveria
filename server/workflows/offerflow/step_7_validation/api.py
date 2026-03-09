from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .models import OfferflowStep7Request, OfferflowStep7Response
from .offerflow_step import run_step_7

security = HTTPBearer(auto_error=False)


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    @router.post("/offerflow/step-7/run", response_model=OfferflowStep7Response)
    def run_offerflow_step_7(
        req: OfferflowStep7Request,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> OfferflowStep7Response:
        ensure_user_dirs(s, user_id)
        api_key = credentials.credentials if credentials else ""
        result = run_step_7(s=s, user_id=user_id, api_key=api_key, req=req)
        return OfferflowStep7Response(**result)

    return router
