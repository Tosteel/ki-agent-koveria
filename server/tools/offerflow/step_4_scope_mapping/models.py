from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from server.tools.offerflow.models_common import OfferflowBaseResponse, OfferflowMetadata


class OfferflowStep4Request(BaseModel):
    offer_id: str = Field(..., min_length=1)
    clarified_requirements: Dict[str, Any] = Field(default_factory=dict)
    service_catalog: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: OfferflowMetadata = Field(default_factory=OfferflowMetadata)
    top_k: int = Field(3, ge=1, le=20)


class OfferflowStep4Response(OfferflowBaseResponse):
    boq_draft: Dict[str, Any] = Field(default_factory=dict)
    output_files: List[str] = Field(default_factory=list)
    rag_query: Dict[str, Any] = Field(default_factory=dict)
    rag_upload: Dict[str, Any] = Field(default_factory=dict)
