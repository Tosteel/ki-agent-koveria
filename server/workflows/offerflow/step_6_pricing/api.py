from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .models import OfferflowStep6Request, OfferflowStep6Response
from .offerflow_step import run_step_6

security = HTTPBearer(auto_error=False)


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    #@router.post("/offerflow/step-6/run", response_model=OfferflowStep6Response)
    def run_offerflow_step_6(
        req: OfferflowStep6Request,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> OfferflowStep6Response:
        ensure_user_dirs(s, user_id)
        api_key = credentials.credentials if credentials else ""
        result = run_step_6(s=s, user_id=user_id, api_key=api_key, req=req)
        return OfferflowStep6Response(**result)

    return router
