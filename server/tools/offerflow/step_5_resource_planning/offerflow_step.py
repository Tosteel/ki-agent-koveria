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

from .models import OfferflowStep5Request


def run_step_5(*, s: Settings, user_id: str, api_key: str, req: OfferflowStep5Request) -> Dict[str, Any]:
    root = offer_dir(s, user_id, req.offer_id)
    service = RagService(s.rag_base_url, api_key)

    boq_draft = req.boq_draft or read_json(root / "4_boq_draft.json")
    query_seed = str(boq_draft.get("positions") or req.offer_id)
    query_result = safe_rag_query(
        service,
        query=query_seed,
        top_k=req.top_k,
        classification="offer.step_5_resource_planning.reference",
    )

    positions = boq_draft.get("positions") or []
    materials = []
    work_items = []
    for i, pos in enumerate(positions, start=1):
        qty = float(pos.get("quantity") or 1)
        materials.append({
            "position_code": pos.get("code") or f"P{i:03d}",
            "material": f"Material_{i}",
            "quantity": round(qty * 1.1, 2),
            "unit": "Stk",
            "waste_factor": 0.1,
        })
        work_items.append({
            "position_code": pos.get("code") or f"P{i:03d}",
            "role": "Monteur",
            "hours": round(max(qty, 1.0) * 0.75, 2),
            "setup_hours": 0.5,
            "travel_hours": 0.5,
        })

    materials_plan = {"offer_id": req.offer_id, "items": materials}
    work_plan = {"offer_id": req.offer_id, "items": work_items, "dependencies": []}

    p_m = write_json(root / "5_materials_plan.json", materials_plan)
    p_w = write_json(root / "5_work_plan.json", work_plan)

    meta = metadata_payload(
        step=5,
        trade=req.metadata.trade,
        region=req.metadata.region,
        project_type=req.metadata.project_type,
        scope_tags=req.metadata.scope_tags,
        size=req.metadata.size,
        outcome=req.metadata.outcome,
    )
    upload_result = safe_rag_upload(
        service,
        classification="offer.step_5_resource_planning.case",
        local_path=str(root),
        custom_metadata=meta,
        extra_fields={"files": {"materials_plan": materials_plan, "work_plan": work_plan}},
    )

    return {
        "ok": True,
        "offer_id": req.offer_id,
        "step": 5,
        "message": "Step 5 abgeschlossen.",
        "materials_plan": materials_plan,
        "work_plan": work_plan,
        "output_files": [str(p_m), str(p_w)],
        "rag_query": query_result,
        "rag_upload": upload_result,
    }
