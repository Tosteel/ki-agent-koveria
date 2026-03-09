from __future__ import annotations

from typing import Any, Dict

from server.core.settings import Settings
from server.workflows.offerflow.common import (
    metadata_payload,
    offer_dir,
    read_json,
    safe_rag_query,
    safe_rag_upload,
    write_json,
)
from server.tools.rag_knowledgebase.service import RagService

from .models import OfferflowStep2Request


def run_step_2(*, s: Settings, user_id: str, api_key: str, req: OfferflowStep2Request) -> Dict[str, Any]:
    root = offer_dir(s, user_id, req.offer_id)
    service = RagService(s.rag_base_url, api_key)

    lead_profile = req.lead_profile or read_json(root / "1_lead_profile.json")
    query_seed = str(lead_profile.get("raw_request") or lead_profile.get("scope") or req.offer_id)
    query_result = safe_rag_query(
        service,
        query=query_seed,
        top_k=req.top_k,
        classification="offer.step_2_qualification.reference",
    )

    flags: list[str] = []
    rules = req.rules or {}
    min_hits = int(rules.get("min_reference_hits", 0) or 0)
    hit_count = len((query_result.get("data") or {}).get("hits", []))
    if hit_count < min_hits:
        flags.append("low_reference_coverage")

    raw_text = str(lead_profile.get("raw_request") or "").lower()
    if "dringend" in raw_text or "sofort" in raw_text:
        flags.append("tight_timeline")

    decision = "go" if "high_risk_forbidden" not in flags else "no-go"
    if rules.get("force_no_go") is True:
        decision = "no-go"
        flags.append("rule_force_no_go")

    qualification_decision = {
        "offer_id": req.offer_id,
        "decision": decision,
        "reason": "Regelbasierte Qualifizierung auf Basis Profil + Referenzfälle.",
        "flags": flags,
        "commercial_fit": "ok" if decision == "go" else "kritisch",
        "timing_fit": "ok" if "tight_timeline" not in flags else "kritisch",
    }

    p = write_json(root / "2_qualification_decision.json", qualification_decision)

    meta = metadata_payload(
        step=2,
        trade=req.metadata.trade,
        region=req.metadata.region,
        project_type=req.metadata.project_type,
        scope_tags=req.metadata.scope_tags,
        size=req.metadata.size,
        outcome=req.metadata.outcome,
    )
    upload_result = safe_rag_upload(
        service,
        classification="offer.step_2_qualification.case",
        local_path=str(root),
        custom_metadata=meta,
        extra_fields={"files": qualification_decision},
    )

    return {
        "ok": True,
        "offer_id": req.offer_id,
        "step": 2,
        "message": "Step 2 abgeschlossen.",
        "qualification_decision": qualification_decision,
        "output_files": [str(p)],
        "rag_query": query_result,
        "rag_upload": upload_result,
    }
