from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from server.tools.rag_knowledgebase.models import RagQueryRequest
from server.tools.filesystem.models import FileReadRequest, FileReadResponse, FileWriteRequest, FileWriteResponse
from server.tools.pdf.models import PdfExportRequest, PdfExportResponse
from server.tools.powerpoint.models import PptExportRequest, PptExportResponse
from server.tools.mail.models import MailSendRequest, MailSendResponse
from server.tools.search_multitable.models import SearchGenerateJsonRequest
from server.tools.rag_knowledgebase import RagService
from server.tools.search_multitable import SearchService
from server.tools.filesystem import read_text, write_text
from server.tools.pdf import export_text_pdf
from server.tools.powerpoint import export_text_pptx
from server.tools.mail import send_mail

security = HTTPBearer(auto_error=False)


def create_tool_router(*, ensure_user_dirs):
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

    @router.post('/files/read', response_model=FileReadResponse)
    def files_read(
        req: FileReadRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> FileReadResponse:
        ensure_user_dirs(s, user_id)
        content = read_text(s.user_work_dir(user_id), req.path, encoding=req.encoding)
        return FileReadResponse(path=req.path, content=content)

    @router.post('/files/write', response_model=FileWriteResponse)
    def files_write(
        req: FileWriteRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> FileWriteResponse:
        ensure_user_dirs(s, user_id)
        n = write_text(
            s.user_work_dir(user_id),
            req.path,
            req.content,
            encoding=req.encoding,
            overwrite=req.overwrite,
        )
        return FileWriteResponse(path=req.path, bytes_written=n)

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

    @router.post('/mail/send', response_model=MailSendResponse)
    def mail_send(
        req: MailSendRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> MailSendResponse:
        ensure_user_dirs(s, user_id)
        result = send_mail(
            to=req.to,
            subject=req.subject,
            body=req.body,
            attachment_paths=req.attachment_paths,
            work_dir=s.user_work_dir(user_id),
            cc=req.cc,
            bcc=req.bcc,
            from_email=req.from_email,
            reply_to=req.reply_to,
            is_html=req.is_html,
        )
        return MailSendResponse(**result)

    return router
