from __future__ import annotations

from typing import Any, Dict

from server.core.settings import Settings
from server.tools.offerflow.common import (
    metadata_payload,
    offer_dir,
    read_json,
    safe_rag_query,
    safe_rag_upload,
    write_json,
)
from server.tools.rag_knowledgebase.service import RagService

from .models import OfferflowStep4Request


def run_step_4(*, s: Settings, user_id: str, api_key: str, req: OfferflowStep4Request) -> Dict[str, Any]:
    root = offer_dir(s, user_id, req.offer_id)
    service = RagService(s.rag_base_url, api_key)

    clarified = req.clarified_requirements or read_json(root / "3_clarified_requirements.json")
    query_seed = str(clarified.get("requirements") or req.offer_id)
    query_result = safe_rag_query(
        service,
        query=query_seed,
        top_k=req.top_k,
        classification="offer.step_4_scope_mapping.reference",
    )

    base_positions = req.service_catalog or [
        {"code": "P001", "text": "Baustelleneinrichtung", "unit": "pauschal", "quantity": 1},
        {"code": "P010", "text": "Leistung laut geklärtem Umfang", "unit": "m2", "quantity": 1},
    ]

    boq_draft = {
        "offer_id": req.offer_id,
        "positions": base_positions,
        "variants": {
            "good": {"modifier": -0.05, "description": "Basisvariante"},
            "better": {"modifier": 0.0, "description": "Empfohlene Variante"},
            "best": {"modifier": 0.12, "description": "Premiumvariante"},
        },
        "source": "rule_based_scope_mapping",
    }

    p = write_json(root / "4_boq_draft.json", boq_draft)

    meta = metadata_payload(
        step=4,
        trade=req.metadata.trade,
        region=req.metadata.region,
        project_type=req.metadata.project_type,
        scope_tags=req.metadata.scope_tags,
        size=req.metadata.size,
        outcome=req.metadata.outcome,
    )
    upload_result = safe_rag_upload(
        service,
        classification="offer.step_4_scope_mapping.case",
        local_path=str(root),
        custom_metadata=meta,
        extra_fields={"files": boq_draft},
    )

    return {
        "ok": True,
        "offer_id": req.offer_id,
        "step": 4,
        "message": "Step 4 abgeschlossen.",
        "boq_draft": boq_draft,
        "output_files": [str(p)],
        "rag_query": query_result,
        "rag_upload": upload_result,
    }
