from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field



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
    path: str = Field(..., description="") #Relativer Pfad innerhalb user/work"
    encoding: str = Field("utf-8")

class FileReadResponse(BaseModel):
    path: str
    content: str

class FileWriteRequest(BaseModel):
    path: str = Field(..., description="") #Relativer Pfad innerhalb user/work
    content: str
    encoding: str = Field("utf-8")
    overwrite: bool = True

class FileWriteResponse(BaseModel):
    path: str
    bytes_written: int

class PdfExportRequest(BaseModel):
    output_path: str = Field(..., description="") #Relativer Pfad innerhalb user/work
    title: str = "Export"
    text: str

class PdfExportResponse(BaseModel):
    output_path: str
    bytes_written: int

class PptExportRequest(BaseModel):
    output_path: str = Field(..., description="")  # Relativer Pfad innerhalb user/work
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

class SearchTableColumn(BaseModel):
    name: str
    description: str

class SearchTableSpec(BaseModel):
    delimiter: str = ";"
    columns: List[SearchTableColumn] = Field(default_factory=list)

class GenerateJsonRequestBodyRequest(BaseModel):
    user_prompt: str

class SearchGenerateJsonRequest(BaseModel):
    user_prompt: str

class GenerateJsonRequest(BaseModel):
    prompt: str
    table: SearchTableSpec
    batches: int = Field(1, ge=1, le=100)

class GenerateJsonToolRequest(BaseModel):
    prompt: str
    table: Dict[str, Any]
    batches: int = Field(1, ge=1, le=100)

class LlmSummaryRequest(BaseModel):
    text: str
    instruction: str = ""
    goal: str = ""
    max_chars: int = Field(1200, ge=200, le=10000)

class LlmSummaryResponse(BaseModel):
    summary: str
    text: str
    fallback_used: bool = False
    model: str = ""
    usage: Optional[Dict[str, Any]] = None

class LlmComposeRequest(BaseModel):
    text: str
    instruction: str = ""
    goal: str = ""
    max_chars: int = Field(3000, ge=200, le=20000)

class LlmComposeResponse(BaseModel):
    composed_text: str
    text: str
    fallback_used: bool = False
    model: str = ""
    usage: Optional[Dict[str, Any]] = None

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

class AgentAskRequest(BaseModel):
    goal: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=50)
    classification: Optional[str] = None  # optional

class AgentAskResponse(BaseModel):
    ok: bool
    goal: str
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    tool_outputs: List[Dict[str, Any]] = Field(default_factory=list)
    answer: str
