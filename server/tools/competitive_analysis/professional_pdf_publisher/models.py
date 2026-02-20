from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class ProfessionalPdfPublisherRequest(BaseModel):
    final_report: Optional[Dict[str, Any]] = None
    final_report_path: Optional[str] = None
    output_path: str = "competition_analysis_report.pdf"
    logo_path: Optional[str] = None
    report_config_path: Optional[str] = None
    chart_paths: List[str] = Field(default_factory=list)
    include_render_log: bool = True
    render_log_path: str = "render_log.json"

    @model_validator(mode="after")
    def _validate_input(self) -> "ProfessionalPdfPublisherRequest":
        if not self.final_report and not (self.final_report_path or "").strip():
            raise ValueError("Either final_report or final_report_path must be provided.")
        return self


class ProfessionalPdfPublisherResponse(BaseModel):
    ok: bool = True
    output_path: str
    bytes_written: int
    render_log_path: str = ""
    warnings: List[str] = Field(default_factory=list)


__all__ = [
    "ProfessionalPdfPublisherRequest",
    "ProfessionalPdfPublisherResponse",
]
