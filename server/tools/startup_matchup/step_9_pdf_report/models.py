from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, model_validator


class PdfReportResult(BaseModel):
    output_path: str
    bytes_written: int
    title: str
    text_preview: str = ""


class StartupMatchupStep9Request(BaseModel):
    final_report: Optional[Dict[str, Any]] = None
    final_report_path: Optional[str] = None
    output_path: str = Field(default="startup_matchup_report.pdf", min_length=5)
    title: str = "Startup Matchup Report"

    @model_validator(mode="after")
    def _validate_inputs(self) -> "StartupMatchupStep9Request":
        has_report = isinstance(self.final_report, dict) and bool(self.final_report)
        has_report_path = bool((self.final_report_path or "").strip())
        if not has_report and not has_report_path:
            raise ValueError("final_report or final_report_path is required")
        return self


class StartupMatchupStep9Response(BaseModel):
    pdf_report: PdfReportResult


__all__ = [
    "PdfReportResult",
    "StartupMatchupStep9Request",
    "StartupMatchupStep9Response",
]
