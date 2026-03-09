from __future__ import annotations

from fastapi import APIRouter, Depends

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings

from .models import ProfessionalPdfPublisherRequest, ProfessionalPdfPublisherResponse
from .professional_pdf_publisher import publish_competition_pdf


def create_router(*, ensure_user_dirs) -> APIRouter:
    router = APIRouter()

    @router.post('/competitive/report/publish-pdf', response_model=ProfessionalPdfPublisherResponse)
    def publish_pdf(
        req: ProfessionalPdfPublisherRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> ProfessionalPdfPublisherResponse:
        ensure_user_dirs(s, user_id)
        result = publish_competition_pdf(
            final_report=req.final_report,
            final_report_path=req.final_report_path,
            output_path=req.output_path,
            logo_path=req.logo_path,
            report_config_path=req.report_config_path,
            chart_paths=req.chart_paths,
            include_render_log=req.include_render_log,
            render_log_path=req.render_log_path,
            user_root=s.user_dir(user_id),
            work_root=s.user_work_dir(user_id),
        )
        return ProfessionalPdfPublisherResponse(**result)

    return router
