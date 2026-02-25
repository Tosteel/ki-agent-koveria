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

from .models import OfferflowStep6Request


def run_step_6(*, s: Settings, user_id: str, api_key: str, req: OfferflowStep6Request) -> Dict[str, Any]:
    root = offer_dir(s, user_id, req.offer_id)
    service = RagService(s.rag_base_url, api_key)

    materials = req.materials_plan or read_json(root / "5_materials_plan.json")
    work = req.work_plan or read_json(root / "5_work_plan.json")

    query_seed = str({"materials": materials.get("items", []), "work": work.get("items", [])})
    query_result = safe_rag_query(
        service,
        query=query_seed,
        top_k=req.top_k,
        classification="offer.step_6_pricing.reference",
    )

    rules = req.pricing_rules or {}
    material_factor = float(rules.get("material_factor", 22.0))
    labor_rate = float(rules.get("labor_rate", 62.0))
    overhead = float(rules.get("overhead_rate", 0.12))
    margin = float(rules.get("margin_rate", 0.18))

    material_cost = sum(float(i.get("quantity") or 0) * material_factor for i in (materials.get("items") or []))
    labor_cost = sum(float(i.get("hours") or 0) * labor_rate for i in (work.get("items") or []))
    net = material_cost + labor_cost
    net_plus_overhead = net * (1.0 + overhead)
    total = net_plus_overhead * (1.0 + margin)

    pricing_sheet = {
        "offer_id": req.offer_id,
        "positions": [
            {"name": "Material", "amount": round(material_cost, 2)},
            {"name": "Labor", "amount": round(labor_cost, 2)},
            {"name": "Overhead", "amount": round(net_plus_overhead - net, 2)},
        ],
        "totals": {
            "net": round(net, 2),
            "net_plus_overhead": round(net_plus_overhead, 2),
            "offer_price": round(total, 2),
        },
        "parameters": {
            "material_factor": material_factor,
            "labor_rate": labor_rate,
            "overhead_rate": overhead,
            "margin_rate": margin,
        },
    }

    p_sheet = write_json(root / "6_pricing_sheet.json", pricing_sheet)
    output_files = [str(p_sheet)]

    rationale = ""
    if req.include_rationale:
        rationale = (
            "# Pricing Rationale\n"
            f"- Material factor: {material_factor}\n"
            f"- Labor rate: {labor_rate}\n"
            f"- Overhead: {overhead}\n"
            f"- Margin: {margin}\n"
            f"- Offer price: {round(total, 2)}\n"
        )
        p_rat = root / "6_pricing_rationale.md"
        p_rat.write_text(rationale, encoding="utf-8")
        output_files.append(str(p_rat))

    meta = metadata_payload(
        step=6,
        trade=req.metadata.trade,
        region=req.metadata.region,
        project_type=req.metadata.project_type,
        scope_tags=req.metadata.scope_tags,
        size=req.metadata.size,
        outcome=req.metadata.outcome,
    )
    upload_result = safe_rag_upload(
        service,
        classification="offer.step_6_pricing.case",
        local_path=str(root),
        custom_metadata=meta,
        extra_fields={"files": pricing_sheet},
    )

    return {
        "ok": True,
        "offer_id": req.offer_id,
        "step": 6,
        "message": "Step 6 abgeschlossen.",
        "pricing_sheet": pricing_sheet,
        "pricing_rationale": rationale,
        "output_files": output_files,
        "rag_query": query_result,
        "rag_upload": upload_result,
    }
