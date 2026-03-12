from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .langsearch import search_langsearch
from .models import LangSearchRequest


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    @router.post('/tools/langsearch/search')
    def langsearch_search(
        req: LangSearchRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> Dict[str, Any]:
        ensure_user_dirs(s, user_id)
        return search_langsearch(
            query=req.query,
            count=req.count,
            summary=req.summary,
            freshness=req.freshness,
        )

    return router
