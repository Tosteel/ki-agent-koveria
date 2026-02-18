from __future__ import annotations

from pydantic import BaseModel, Field


class PdfExportRequest(BaseModel):
    output_path: str = Field(..., description="")
    title: str = "Export"
    text: str


class PdfExportResponse(BaseModel):
    output_path: str
    bytes_written: int


class PdfReadRequest(BaseModel):
    path: str = Field(..., description="Pfad zur PDF (z.B. uploads/datei.pdf oder work/datei.pdf)")
    max_chars: int = Field(20000, ge=500, le=200000)


class PdfReadResponse(BaseModel):
    path: str
    pages: int
    chars: int
    text: str


__all__ = ["PdfExportRequest", "PdfExportResponse", "PdfReadRequest", "PdfReadResponse"]
