from __future__ import annotations

from pydantic import BaseModel, Field


class PdfExportRequest(BaseModel):
    output_path: str = Field(..., description="")
    title: str = "Export"
    text: str


class PdfExportResponse(BaseModel):
    output_path: str
    bytes_written: int


__all__ = ["PdfExportRequest", "PdfExportResponse"]
