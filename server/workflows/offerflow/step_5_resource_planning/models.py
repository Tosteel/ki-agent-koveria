from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from server.workflows.offerflow.models_common import OfferflowBaseResponse, OfferflowMetadata


class OfferflowStep5Request(BaseModel):
    offer_id: str = Field(..., min_length=1)
    boq_draft: Dict[str, Any] = Field(default_factory=dict)
    planning_rules: Dict[str, Any] = Field(default_factory=dict)
    metadata: OfferflowMetadata = Field(default_factory=OfferflowMetadata)
    top_k: int = Field(3, ge=1, le=20)


class OfferflowStep5Response(OfferflowBaseResponse):
    materials_plan: Dict[str, Any] = Field(default_factory=dict)
    work_plan: Dict[str, Any] = Field(default_factory=dict)
    output_files: List[str] = Field(default_factory=list)
    rag_query: Dict[str, Any] = Field(default_factory=dict)
    rag_upload: Dict[str, Any] = Field(default_factory=dict)
