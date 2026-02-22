from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .models import VideoAnalyzeSyncRequest
from .websearch_videoanalyzer import VideoAnalyzerService

security = HTTPBearer(auto_error=False)


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    @router.post('/video/analyze_json_sync_from_prompt')
    def analyze_json_sync_from_prompt(
        req: VideoAnalyzeSyncRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> Dict[str, Any]:
        ensure_user_dirs(s, user_id)
        service = VideoAnalyzerService(s.video_analyzer_base_url, credentials.credentials)
        return service.analyze_json_sync_from_prompt(prompt=req.prompt)

    return router
