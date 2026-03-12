from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .models import TemplateToolRequest, TemplateToolResponse
from .tool_template import run_tool


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    @router.post("/tools/template_tool/run", response_model=TemplateToolResponse)
    def template_tool_run(
        req: TemplateToolRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> TemplateToolResponse:
        ensure_user_dirs(s, user_id)
        result = run_tool(text=req.text, extra=req.extra)
        return TemplateToolResponse(**result)

    return router
