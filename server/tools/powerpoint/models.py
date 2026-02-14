from __future__ import annotations

from pydantic import BaseModel, Field


class PptExportRequest(BaseModel):
    output_path: str = Field(..., description="")
    title: str = "Export"
    text: str
    use_llm_layout: bool = True
    allow_heuristic_fallback: bool = False
    goal: str = ""
    instruction: str = ""
    max_slides: int = Field(12, ge=1, le=60)
    max_boxes_per_slide: int = Field(3, ge=1, le=6)


class PptExportResponse(BaseModel):
    output_path: str
    bytes_written: int
    layout_mode: str = "heuristic"


__all__ = ["PptExportRequest", "PptExportResponse"]
