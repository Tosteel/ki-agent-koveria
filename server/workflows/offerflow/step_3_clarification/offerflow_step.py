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

from .models import OfferflowStep3Request


def run_step_3(*, s: Settings, user_id: str, api_key: str, req: OfferflowStep3Request) -> Dict[str, Any]:
    root = offer_dir(s, user_id, req.offer_id)
    service = RagService(s.rag_base_url, api_key)

    qual = req.qualification_decision or read_json(root / "2_qualification_decision.json")
    lead_profile = req.lead_profile or read_json(root / "1_lead_profile.json")

    query_seed = str(lead_profile.get("raw_request") or req.offer_id)
    query_result = safe_rag_query(
        service,
        query=query_seed,
        top_k=req.top_k,
        classification="offer.step_3_clarification.reference",
    )

    decision = str(qual.get("decision") or "go").lower()
    if decision == "no-go":
        question_set = {"questions": [], "note": "No-Go: keine Klärungsfragen erzeugt."}
        clarified = {
            "offer_id": req.offer_id,
            "status": "blocked_no_go",
            "requirements": {},
            "answers": req.answers,
        }
    else:
        default_questions = [
            "Welche exakten Maße/Mengen liegen vor?",
            "Gibt es Randbedingungen zu Zugang/Arbeitszeiten?",
            "Welche Termine sind fix und welche flexibel?",
        ]
        question_set = {
            "offer_id": req.offer_id,
            "questions": [{"priority": i + 1, "text": q} for i, q in enumerate(default_questions)],
        }
        clarified = {
            "offer_id": req.offer_id,
            "status": "clarified",
            "requirements": {
                "scope": lead_profile.get("scope", {}),
                "constraints": req.answers,
            },
            "answers": req.answers,
        }

    p_q = write_json(root / "3_question_set.json", question_set)
    p_c = write_json(root / "3_clarified_requirements.json", clarified)

    meta = metadata_payload(
        step=3,
        trade=req.metadata.trade,
        region=req.metadata.region,
        project_type=req.metadata.project_type,
        scope_tags=req.metadata.scope_tags,
        size=req.metadata.size,
        outcome=req.metadata.outcome,
    )
    upload_result = safe_rag_upload(
        service,
        classification="offer.step_3_clarification.case",
        local_path=str(root),
        custom_metadata=meta,
        extra_fields={"files": {"question_set": question_set, "clarified_requirements": clarified}},
    )

    return {
        "ok": True,
        "offer_id": req.offer_id,
        "step": 3,
        "message": "Step 3 abgeschlossen.",
        "question_set": question_set,
        "clarified_requirements": clarified,
        "output_files": [str(p_q), str(p_c)],
        "rag_query": query_result,
        "rag_upload": upload_result,
    }
