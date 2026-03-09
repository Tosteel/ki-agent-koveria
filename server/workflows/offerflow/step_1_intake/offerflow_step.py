from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from server.core.settings import Settings
from server.workflows.offerflow.common import (
    metadata_payload,
    offer_dir,
    safe_rag_query,
    safe_rag_upload,
    write_json,
)
from server.tools.rag_knowledgebase.service import RagService

from .models import OfferflowStep1Request


def _infer_trade_and_project(raw_request: str) -> tuple[str, str]:
    t = raw_request.lower()
    trade_map = {
        "maler": "Maler",
        "heizung": "Heizung",
        "pv": "PV",
        "solar": "PV",
        "bad": "Sanitär",
        "elektro": "Elektro",
    }
    trade = ""
    for key, value in trade_map.items():
        if key in t:
            trade = value
            break

    project_type = "Sanierung" if any(x in t for x in ["sanierung", "bestand", "renovierung"]) else "Neubau"
    return trade, project_type


def run_step_1(*, s: Settings, user_id: str, api_key: str, req: OfferflowStep1Request) -> Dict[str, Any]:
    root = offer_dir(s, user_id, req.offer_id)
    service = RagService(s.rag_base_url, api_key)

    query_result = safe_rag_query(
        service,
        query=req.raw_request,
        top_k=req.top_k,
        classification="offer.step_1_intake.reference",
    )

    inferred_trade, inferred_project_type = _infer_trade_and_project(req.raw_request)
    lead_profile = {
        "offer_id": req.offer_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "customer": req.customer_data,
        "raw_request": req.raw_request,
        "scope": {
            "pre_classification_trade": req.metadata.trade or inferred_trade,
            "pre_classification_project_type": req.metadata.project_type or inferred_project_type,
        },
        "attachments_count": len(req.attachments),
        "reference_hits_count": len((query_result.get("data") or {}).get("hits", [])),
    }

    attachments_index = {
        "offer_id": req.offer_id,
        "items": [
            {
                "index": i + 1,
                "name": str(a.get("name") or a.get("filename") or f"attachment_{i+1}"),
                "content_type": str(a.get("content_type") or ""),
                "note": str(a.get("note") or ""),
            }
            for i, a in enumerate(req.attachments)
        ],
    }

    p_lead = write_json(root / "1_lead_profile.json", lead_profile)
    p_att = write_json(root / "1_attachments_index.json", attachments_index)

    meta = metadata_payload(
        step=1,
        trade=req.metadata.trade or inferred_trade,
        region=req.metadata.region,
        project_type=req.metadata.project_type or inferred_project_type,
        scope_tags=req.metadata.scope_tags,
        size=req.metadata.size,
        outcome=req.metadata.outcome,
    )

    upload_result = safe_rag_upload(
        service,
        classification="offer.step_1_intake.case",
        local_path=str(root),
        custom_metadata=meta,
        extra_fields={
            "files": {
                "lead_profile": lead_profile,
                "attachments_index": attachments_index,
            }
        },
    )

    return {
        "ok": True,
        "offer_id": req.offer_id,
        "step": 1,
        "message": "Step 1 abgeschlossen.",
        "lead_profile": lead_profile,
        "attachments_index": attachments_index,
        "output_files": [str(p_lead), str(p_att)],
        "rag_query": query_result,
        "rag_upload": upload_result,
    }
