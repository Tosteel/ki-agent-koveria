from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .models import PptExportRequest, PptExportResponse
from .powerpoint import export_text_pptx


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    @router.post('/ppt/export', response_model=PptExportResponse)
    def ppt_export(
        req: PptExportRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> PptExportResponse:
        ensure_user_dirs(s, user_id)
        out = (s.user_work_dir(user_id) / req.output_path.strip().lstrip('/')).resolve()
        result = export_text_pptx(
            out,
            title=req.title,
            text=req.text,
            use_llm_layout=req.use_llm_layout,
            allow_heuristic_fallback=req.allow_heuristic_fallback,
            goal=req.goal,
            instruction=req.instruction,
            max_slides=req.max_slides,
            max_boxes_per_slide=req.max_boxes_per_slide,
        )
        return PptExportResponse(
            output_path=req.output_path,
            bytes_written=int(result.get('bytes_written') or 0),
            layout_mode=str(result.get('layout_mode') or 'heuristic'),
        )

    return router
