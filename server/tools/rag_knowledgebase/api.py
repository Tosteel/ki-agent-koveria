from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .models import RagQueryRequest
from .rag_knowledgebase import RagService

security = HTTPBearer(auto_error=False)


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    @router.post('/rag/query')
    def rag_query(
        req: RagQueryRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> Dict[str, Any]:
        ensure_user_dirs(s, user_id)

        api_key = credentials.credentials
        service = RagService(s.rag_base_url, api_key)

        data = service.query(query=req.query, top_k=req.top_k, classification=req.classification)
        return data

    return router
