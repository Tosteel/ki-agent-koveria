from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class Step61FinalReportChapters(BaseModel):
    chapter_1_executive_summary: str = "-"
    chapter_2_company_profiles: str = "-"
    chapter_3_company_matrix: str = "-"
    chapter_4_insights: str = "-"
    chapter_5_recommendations: str = "-"
    chapter_6_appendix_trends: str = "-"


class Step61FinalReportResult(BaseModel):
    schema_version: str = "1.0"
    provider: str = "ionos"
    output_file: str = "step6.1_final_report.json"
    chapters: Step61FinalReportChapters = Field(default_factory=Step61FinalReportChapters)
    report_markdown: str = ""
    source_urls: List[str] = Field(default_factory=list)
    extraction_warnings: List[str] = Field(default_factory=list)


class Step61FinalReportRequest(BaseModel):
    market_trends_summary: Optional[Dict[str, Any]] = None
    market_trends_summary_path: Optional[str] = None
    competitor_trends: Optional[Dict[str, Any]] = None
    competitor_trends_path: Optional[str] = None
    matrix: Optional[Dict[str, Any]] = None
    matrix_path: Optional[str] = None
    insights: Optional[Dict[str, Any]] = None
    insights_path: Optional[str] = None
    recommendations: Optional[Dict[str, Any]] = None
    recommendations_path: Optional[str] = None
    provider: str = "ionos"
    max_companies: int = Field(default=120, ge=1, le=1000)

    @model_validator(mode="after")
    def _validate_inputs(self) -> "Step61FinalReportRequest":
        checks = [
            (self.market_trends_summary, self.market_trends_summary_path, "market_trends_summary"),
            (self.competitor_trends, self.competitor_trends_path, "competitor_trends"),
            (self.matrix, self.matrix_path, "matrix"),
            (self.insights, self.insights_path, "insights"),
            (self.recommendations, self.recommendations_path, "recommendations"),
        ]
        for inline_obj, path, name in checks:
            has_inline = isinstance(inline_obj, dict) and bool(inline_obj)
            has_path = bool((path or "").strip())
            if not has_inline and not has_path:
                raise ValueError(f"Either {name} or {name}_path must be provided.")
        return self


class Step61FinalReportResponse(BaseModel):
    final_report: Step61FinalReportResult


__all__ = [
    "Step61FinalReportChapters",
    "Step61FinalReportRequest",
    "Step61FinalReportResponse",
    "Step61FinalReportResult",
]
