from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ParsedSection(BaseModel):
    name: str
    text: str = ""
    start_line: Optional[int] = None
    end_line: Optional[int] = None


class ParsedTable(BaseModel):
    title: str = ""
    page: Optional[int] = None
    headers: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)
    source: str = ""


class ParsedMeasurement(BaseModel):
    property_name: str = ""
    value: float | int
    unit: str
    raw: str
    context: str = ""
    page: Optional[int] = None


class ParsedMetadata(BaseModel):
    product_name: Optional[str] = None
    manufacturer: Optional[str] = None
    document_version: Optional[str] = None
    publication_date: Optional[str] = None


class ParsedDocument(BaseModel):
    schema_version: str = "1.0"
    source_path: str
    detected_format: str
    mime_type: str = ""
    ocr_used: bool = False
    language: Optional[str] = None
    metadata: ParsedMetadata = Field(default_factory=ParsedMetadata)
    sections: List[ParsedSection] = Field(default_factory=list)
    tables: List[ParsedTable] = Field(default_factory=list)
    measurements: List[ParsedMeasurement] = Field(default_factory=list)
    raw_text: str = ""
    extraction_warnings: List[str] = Field(default_factory=list)


class CompetitiveDocumentImportRequest(BaseModel):
    path: str = Field(..., min_length=1)
    max_chars: int = Field(default=50000, ge=1000, le=500000)


class CompetitiveDocumentImportResponse(BaseModel):
    parsed_doc: ParsedDocument


__all__ = [
    "CompetitiveDocumentImportRequest",
    "CompetitiveDocumentImportResponse",
    "ParsedDocument",
    "ParsedMetadata",
    "ParsedMeasurement",
    "ParsedSection",
    "ParsedTable",
]
