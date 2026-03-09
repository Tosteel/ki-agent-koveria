from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from server.workflows.offerflow.models_common import OfferflowBaseResponse, OfferflowMetadata


class OfferflowStep1Request(BaseModel):
    offer_id: str = Field(..., min_length=1)
    raw_request: str = Field(..., min_length=1)
    customer_data: Dict[str, Any] = Field(default_factory=dict)
    attachments: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: OfferflowMetadata = Field(default_factory=OfferflowMetadata)
    top_k: int = Field(3, ge=1, le=20)


class OfferflowStep1Response(OfferflowBaseResponse):
    lead_profile: Dict[str, Any] = Field(default_factory=dict)
    attachments_index: Dict[str, Any] = Field(default_factory=dict)
    output_files: List[str] = Field(default_factory=list)
    rag_query: Dict[str, Any] = Field(default_factory=dict)
    rag_upload: Dict[str, Any] = Field(default_factory=dict)
