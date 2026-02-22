from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .models import SearchGenerateJsonRequest
from .websearch_table import SearchService

security = HTTPBearer(auto_error=False)


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    @router.post('/search/generate_json')
    def search_generate_json(
        req: SearchGenerateJsonRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> Dict[str, Any]:
        ensure_user_dirs(s, user_id)
        service = SearchService(s.search_base_url, credentials.credentials)
        return service.search_generate_json(user_prompt=req.user_prompt)

    return router
