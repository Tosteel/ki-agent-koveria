from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .models import RagQueryRequest
from .service import RagService

security = HTTPBearer(auto_error=False)


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    @router.post('/tools/rag/query')
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

    @router.post('/tools/rag/upload')
    async def rag_upload(
        request: Request,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> Dict[str, Any]:
        ensure_user_dirs(s, user_id)

        if credentials is None or not credentials.credentials:
            raise HTTPException(status_code=401, detail="Missing bearer token")

        form = await request.form()

        raw_classification = form.get("classification")
        classification = str(raw_classification or "").strip()
        if not classification:
            raise HTTPException(status_code=422, detail="classification is required")

        multipart_data: list[tuple[str, str]] = [("classification", classification)]
        if "local_path" in form:
            multipart_data.append(("local_path", str(form.get("local_path") or "")))
        if "custom_metadata" in form:
            multipart_data.append(("custom_metadata", str(form.get("custom_metadata") or "")))

        multipart_files: list[tuple[str, tuple[str, bytes, str]]] = []
        for item in form.getlist("files"):
            if isinstance(item, UploadFile):
                content = await item.read()
                multipart_files.append(
                    (
                        "files",
                        (
                            str(item.filename or "upload.bin"),
                            content,
                            str(item.content_type or "application/octet-stream"),
                        ),
                    )
                )
                await item.close()
            else:
                multipart_data.append(("files", str(item or "")))

        api_key = credentials.credentials
        service = RagService(s.rag_base_url, api_key)
        data = service.upload(data=multipart_data, files=multipart_files)
        return data

    return router
