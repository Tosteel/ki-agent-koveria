from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, model_validator


class Step71PdfExportResult(BaseModel):
    schema_version: str = "1.0"
    provider: str = "ionos"
    output_file: str = "step7.1_report.pdf"
    bytes_written: int = 0
    title: str = "Competitive Intelligence Report"
    extraction_warnings: list[str] = Field(default_factory=list)


class Step71PdfExportRequest(BaseModel):
    final_report: Optional[Dict[str, Any]] = None
    final_report_path: Optional[str] = None
    output_path: str = "step7.1_report.pdf"
    title: str = "Competitive Intelligence Report"
    subtitle: str = "Wettbewerbsanalyse und strategische Handlungsempfehlungen"
    report_year: Optional[int] = Field(default=None, ge=2000, le=2100)
    created_by: str = "Competitive Intelligence Tool"
    company_logo_path: Optional[str] = None
    tool_logo_path: Optional[str] = None
    provider: str = "ionos"

    @model_validator(mode="after")
    def _validate_inputs(self) -> "Step71PdfExportRequest":
        has_inline = isinstance(self.final_report, dict) and bool(self.final_report)
        has_path = bool((self.final_report_path or "").strip())
        if not has_inline and not has_path:
            raise ValueError("Either final_report or final_report_path must be provided.")
        return self


class Step71PdfExportResponse(BaseModel):
    pdf_export: Step71PdfExportResult


__all__ = [
    "Step71PdfExportRequest",
    "Step71PdfExportResponse",
    "Step71PdfExportResult",
]
