from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .filesystem import read_text, write_text
from .models import FileReadRequest, FileReadResponse, FileWriteRequest, FileWriteResponse


def _safe_upload_name(raw_name: str) -> str:
    name = Path(str(raw_name or "").strip()).name
    if not name:
        name = "upload.bin"
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return name or "upload.bin"


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    @router.post('/tools/files/read', response_model=FileReadResponse)
    def files_read(
        req: FileReadRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> FileReadResponse:
        ensure_user_dirs(s, user_id)
        content = read_text(
            s.user_work_dir(user_id),
            req.path,
            encoding=req.encoding,
            uploads_dir=s.user_dir(user_id) / "uploads",
        )
        return FileReadResponse(path=req.path, content=content)

    @router.post('/tools/files/write', response_model=FileWriteResponse)
    def files_write(
        req: FileWriteRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> FileWriteResponse:
        ensure_user_dirs(s, user_id)
        content = req.content
        if isinstance(content, (dict, list)):
            content_text = json.dumps(content, ensure_ascii=False, indent=2)
        elif isinstance(content, str):
            content_text = content
        else:
            content_text = str(content)
        n = write_text(
            s.user_work_dir(user_id),
            req.path,
            content_text,
            encoding=req.encoding,
            overwrite=req.overwrite,
        )
        return FileWriteResponse(path=req.path, bytes_written=n)

    @router.post('/tools/files/upload')
    async def files_upload(
        file: UploadFile = File(...),
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> Dict[str, Any]:
        ensure_user_dirs(s, user_id)
        uploads_dir = (s.user_dir(user_id) / "uploads").resolve()
        uploads_dir.mkdir(parents=True, exist_ok=True)
        work_dir = s.user_work_dir(user_id).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)

        original_name = str(file.filename or "").strip()
        safe_name = _safe_upload_name(original_name)
        if not safe_name:
            raise HTTPException(status_code=422, detail="Invalid filename")

        target = uploads_dir / safe_name
        stem = Path(safe_name).stem
        suffix = Path(safe_name).suffix
        idx = 1
        while target.exists():
            candidate = f"{stem}_{idx}{suffix}"
            target = uploads_dir / candidate
            idx += 1

        size = 0
        try:
            with target.open("wb") as f:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    size += len(chunk)
        finally:
            await file.close()

        work_target = (work_dir / target.name).resolve()
        shutil.copy2(target, work_target)

        return {
            "ok": True,
            "filename": target.name,
            "bytes_written": size,
            "content_type": str(file.content_type or ""),
            "upload_path": f"uploads/{target.name}",
            "storage_path": f"data/users/{user_id}/uploads/{target.name}",
            "work_path": f"data/users/{user_id}/work/{work_target.name}",
        }

    return router
