from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .models import PdfExportRequest, PdfExportResponse, PdfReadRequest, PdfReadResponse
from .pdf import export_text_pdf, read_pdf_text


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    @router.post('/pdf/export', response_model=PdfExportResponse)
    def pdf_export(
        req: PdfExportRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> PdfExportResponse:
        ensure_user_dirs(s, user_id)
        out = (s.user_work_dir(user_id) / req.output_path.strip().lstrip('/')).resolve()
        size = export_text_pdf(out, title=req.title, text=req.text)
        return PdfExportResponse(output_path=req.output_path, bytes_written=size)

    @router.post('/pdf/read', response_model=PdfReadResponse)
    def pdf_read(
        req: PdfReadRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> PdfReadResponse:
        ensure_user_dirs(s, user_id)
        user_root = s.user_dir(user_id).resolve()
        work_root = s.user_work_dir(user_id).resolve()
        uploads_root = (user_root / "uploads").resolve()
        uploads_root.mkdir(parents=True, exist_ok=True)

        raw = req.path.strip().lstrip("/")
        p = Path(raw)
        if p.is_absolute() or ".." in p.parts:
            raise HTTPException(status_code=400, detail=f"Invalid path: {req.path}")
        candidates: list[Path] = []
        parts = list(p.parts)
        if parts and parts[0] == "uploads":
            candidates.append((uploads_root / Path(*parts[1:])).resolve())
        elif parts and parts[0] == "work":
            candidates.append((work_root / Path(*parts[1:])).resolve())
        else:
            candidates.append((work_root / p).resolve())
            candidates.append((uploads_root / p).resolve())

        target: Path | None = None
        for c in candidates:
            if c.exists() and c.is_file():
                if user_root not in c.parents and c != user_root:
                    continue
                target = c
                break
        if target is None:
            raise HTTPException(status_code=404, detail="PDF not found")

        text, pages = read_pdf_text(target, max_chars=req.max_chars)
        return PdfReadResponse(path=req.path, pages=pages, chars=len(text), text=text)

    return router
