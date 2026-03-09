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

from .models import OfferflowStep7Request


def run_step_7(*, s: Settings, user_id: str, api_key: str, req: OfferflowStep7Request) -> Dict[str, Any]:
    root = offer_dir(s, user_id, req.offer_id)
    service = RagService(s.rag_base_url, api_key)

    boq_draft = req.boq_draft or read_json(root / "4_boq_draft.json")
    pricing_sheet = req.pricing_sheet or read_json(root / "6_pricing_sheet.json")

    query_seed = str({"boq": boq_draft.get("positions", []), "pricing": pricing_sheet.get("totals", {})})
    query_result = safe_rag_query(
        service,
        query=query_seed,
        top_k=req.top_k,
        classification="offer.step_7_validation.reference",
    )

    warnings = []
    fixes = []
    score = 0.1

    offer_price = float((pricing_sheet.get("totals") or {}).get("offer_price") or 0)
    if offer_price <= 0:
        warnings.append("offer_price_missing_or_zero")
        fixes.append("pricing_sheet prüfen und neu kalkulieren")
        score += 0.35

    if not boq_draft.get("positions"):
        warnings.append("empty_boq")
        fixes.append("Step 4 Scope Mapping erneut ausführen")
        score += 0.35

    if req.constraints.get("deadline"):
        score += 0.05

    validation_report = {
        "offer_id": req.offer_id,
        "warnings": warnings,
        "fixes": fixes,
        "risk_score": round(min(score, 1.0), 2),
    }

    final_snapshot = {
        "offer_id": req.offer_id,
        "boq_draft": boq_draft,
        "pricing_sheet": pricing_sheet,
        "constraints": req.constraints,
        "validation": validation_report,
    }

    p_v = write_json(root / "7_validation_report.json", validation_report)
    p_f = write_json(root / "7_final_inputs_snapshot.json", final_snapshot)

    meta = metadata_payload(
        step=7,
        trade=req.metadata.trade,
        region=req.metadata.region,
        project_type=req.metadata.project_type,
        scope_tags=req.metadata.scope_tags,
        size=req.metadata.size,
        outcome=req.metadata.outcome,
    )
    upload_result = safe_rag_upload(
        service,
        classification="offer.step_7_validation.case",
        local_path=str(root),
        custom_metadata=meta,
        extra_fields={"files": {"validation_report": validation_report, "final_inputs_snapshot": final_snapshot}},
    )

    return {
        "ok": True,
        "offer_id": req.offer_id,
        "step": 7,
        "message": "Step 7 abgeschlossen.",
        "validation_report": validation_report,
        "final_inputs_snapshot": final_snapshot,
        "output_files": [str(p_v), str(p_f)],
        "rag_query": query_result,
        "rag_upload": upload_result,
    }
