from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SearchTableColumn(BaseModel):
    name: str
    description: str


class SearchTableSpec(BaseModel):
    delimiter: str = ";"
    columns: List[SearchTableColumn] = Field(default_factory=list)


class GenerateJsonRequestBodyRequest(BaseModel):
    user_prompt: str


class GenerateJsonRequest(BaseModel):
    prompt: str
    table: SearchTableSpec
    batches: int = Field(1, ge=1, le=100)


class GenerateJsonToolRequest(BaseModel):
    prompt: str
    table: Dict[str, Any]
    batches: int = Field(1, ge=1, le=100)


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


class AgentAskResponse(BaseModel):
    ok: bool
    goal: str
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    tool_outputs: List[Dict[str, Any]] = Field(default_factory=list)
    answer: str
    requires_user_input: bool = False
    missing_fields: List[str] = Field(default_factory=list)
    questions: List[str] = Field(default_factory=list)
