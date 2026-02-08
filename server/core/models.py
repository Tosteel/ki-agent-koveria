from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class RagQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(3, ge=1, le=50)
    classification: Optional[str] = None

class RagHit(BaseModel):
    source: str
    score: float
    text: str


class RagQueryResponse(BaseModel):
    query: str
    hits: List[RagHit]


class FileReadRequest(BaseModel):
    path: str = Field(..., description="Relativer Pfad innerhalb user/work")
    encoding: str = Field("utf-8")


class FileReadResponse(BaseModel):
    path: str
    content: str


class FileWriteRequest(BaseModel):
    path: str = Field(..., description="Relativer Pfad innerhalb user/work")
    content: str
    encoding: str = Field("utf-8")
    overwrite: bool = True


class FileWriteResponse(BaseModel):
    path: str
    bytes_written: int


class PdfExportRequest(BaseModel):
    output_path: str = Field(..., description="Relativer Pfad innerhalb user/work")
    title: str = "Export"
    text: str


class PdfExportResponse(BaseModel):
    output_path: str
    bytes_written: int


class ToolStep(BaseModel):
    tool: str
    args: Dict[str, Any] = Field(default_factory=dict)


class AgentRunRequest(BaseModel):
    steps: List[ToolStep] = Field(default_factory=list, description="Explizite Tool-Schritte. Wenn leer: default flow.")
    rag_query: Optional[str] = None
    top_k: int = 3
    write_path: str = "result.txt"
    pdf_path: str = "result.pdf"
    pdf_title: str = "Ergebnis"


class AgentRunResponse(BaseModel):
    ok: bool
    outputs: List[Dict[str, Any]] = Field(default_factory=list)
