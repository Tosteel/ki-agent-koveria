from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from fastapi import HTTPException
from server.agent.tool_registry import ToolContext, ToolRegistry

from .models import PdfExportRequest, PdfExportResponse, PdfReadRequest, PdfReadResponse
from .pdf import export_text_pdf, read_pdf_text


def register(registry: ToolRegistry) -> None:
    def tool_pdf_export(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = PdfExportRequest(**args)
        out = (ctx.settings.user_work_dir(ctx.user_id) / req.output_path).resolve()
        size = export_text_pdf(out, title=req.title, text=req.text)
        return PdfExportResponse(output_path=req.output_path, bytes_written=size).model_dump()

    def tool_read_pdf(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
        req = PdfReadRequest(**args)
        raw = req.path.strip().lstrip("/")
        if not raw:
            raise HTTPException(status_code=422, detail="path is required")

        # Allow explicit prefixes and safe defaults for existing user flows.
        # Supported:
        # - uploads/<file>.pdf
        # - work/<file>.pdf
        # - <file>.pdf  (search work first, then uploads)
        user_root = ctx.settings.user_dir(ctx.user_id).resolve()
        work_root = ctx.settings.user_work_dir(ctx.user_id).resolve()
        uploads_root = (user_root / "uploads").resolve()
        uploads_root.mkdir(parents=True, exist_ok=True)

        candidates: list[Path] = []
        raw_path = Path(raw)
        if raw_path.is_absolute() or ".." in raw_path.parts:
            raise HTTPException(status_code=400, detail=f"Invalid path: {req.path}")
        parts = list(raw_path.parts)
        if parts and parts[0] == "uploads":
            candidates.append((uploads_root / Path(*parts[1:])).resolve())
        elif parts and parts[0] == "work":
            candidates.append((work_root / Path(*parts[1:])).resolve())
        else:
            candidates.append((work_root / raw_path).resolve())
            candidates.append((uploads_root / raw_path).resolve())

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
        return PdfReadResponse(
            path=str(req.path),
            pages=pages,
            chars=len(text),
            text=text,
        ).model_dump()

    registry.register("pdf_export", tool_pdf_export, request_model=PdfExportRequest)
    registry.register("read_pdf", tool_read_pdf, request_model=PdfReadRequest)
