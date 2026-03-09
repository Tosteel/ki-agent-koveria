from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from server.tools.startup_matchup.common import (
    clean_text,
    clamp_score,
    compact_context,
    llm_json,
    load_json_obj,
    overlap_score,
    safe_list_str,
)

from .models import GapAnalysis, StartupMatchupStep3Request


_STEP3_SCHEMA: Dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "identified_gaps": {"type": "array", "items": {"type": "string"}},
        "innovation_opportunities": {"type": "array", "items": {"type": "string"}},
        "startup_search_fields": {"type": "array", "items": {"type": "string"}},
        "startup_search_queries": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "identified_gaps",
        "innovation_opportunities",
        "startup_search_fields",
        "startup_search_queries",
    ],
}


def _build_queries(company_name: str, fields: List[str], max_queries: int) -> List[str]:
    company = clean_text(company_name) or "enterprise"
    out: List[str] = []
    seen: set[str] = set()
    for field in fields:
        f = clean_text(field)
        if not f:
            continue
        candidates = [
            f"startup {f} {company} partnership",
            f"{f} startup companies B2B collaboration",
            f"emerging startups in {f}",
        ]
        for q in candidates:
            key = q.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(q)
            if len(out) >= max_queries:
                return out
    if not out:
        out.append(f"startup innovation partners for {company}")
    return out[:max_queries]


def run_step_3(*, req: StartupMatchupStep3Request, user_root: Path, work_root: Path) -> GapAnalysis:
    warnings: List[str] = []

    workshop = load_json_obj(
        inline_obj=req.workshop_analysis,
        path=req.workshop_analysis_path,
        root_key="workshop_analysis",
        user_root=user_root,
        work_root=work_root,
    )
    company = load_json_obj(
        inline_obj=req.company_profile,
        path=req.company_profile_path,
        root_key="company_profile",
        user_root=user_root,
        work_root=work_root,
    )

    workshop_terms = safe_list_str(
        safe_list_str(workshop.get("innovation_goals"))
        + safe_list_str(workshop.get("strategic_fields"))
        + safe_list_str(workshop.get("technology_interests"))
        + safe_list_str(workshop.get("target_use_cases"))
    )
    company_terms = safe_list_str(
        safe_list_str(company.get("technology_domains"))
        + safe_list_str(company.get("innovation_focus"))
        + safe_list_str(company.get("strategic_objectives"))
        + [clean_text(company.get("industry")), clean_text(company.get("core_business"))]
    )

    company_blob = "\n".join(company_terms)
    identified_gaps: List[str] = []
    for term in workshop_terms:
        score = overlap_score(term, company_blob)
        if score < 0.14:
            identified_gaps.append(term)

    identified_gaps = safe_list_str(identified_gaps, limit=20)
    if not identified_gaps:
        identified_gaps = workshop_terms[:8]
        warnings.append("No clear lexical gaps found; used workshop priorities as search baseline.")

    opportunities = [f"Collaboration opportunity in: {gap}" for gap in identified_gaps[:12]]
    startup_search_fields = safe_list_str(
        identified_gaps
        + safe_list_str(workshop.get("problem_statements"))
        + safe_list_str(workshop.get("technology_interests"))
    )[:12]

    company_name = clean_text(company.get("company_name"))
    startup_search_queries = _build_queries(company_name, startup_search_fields, req.max_queries)

    system_prompt = (
        "Du fuehrst eine Gap-Analyse fuer Startup-Kooperationen durch. "
        "Vergleiche Zielbild aus Workshop mit Unternehmensprofil. "
        "Liefere nur JSON gemaess Schema."
    )
    user_prompt = (
        f"Workshop:\n{compact_context([workshop], max_chars=req.max_context_chars // 2)}\n\n"
        f"Company profile:\n{compact_context([company], max_chars=req.max_context_chars // 2)}"
    )
    parsed = llm_json(
        provider=req.provider,
        schema_name="startup_matchup_step3",
        schema=_STEP3_SCHEMA,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        warnings=warnings,
    )

    llm_gaps = safe_list_str(parsed.get("identified_gaps")) if isinstance(parsed, dict) else []
    llm_opp = safe_list_str(parsed.get("innovation_opportunities")) if isinstance(parsed, dict) else []
    llm_fields = safe_list_str(parsed.get("startup_search_fields")) if isinstance(parsed, dict) else []
    llm_queries = safe_list_str(parsed.get("startup_search_queries")) if isinstance(parsed, dict) else []

    final_gaps = llm_gaps[:20] if llm_gaps else identified_gaps
    final_opportunities = llm_opp[:20] if llm_opp else opportunities
    final_fields = llm_fields[:15] if llm_fields else startup_search_fields
    final_queries = llm_queries[: req.max_queries] if llm_queries else startup_search_queries

    # Guardrails: ensure queries are still useful and not empty.
    final_queries = [q for q in final_queries if clean_text(q)]
    if not final_queries:
        final_queries = _build_queries(company_name, final_fields, req.max_queries)

    # deterministic score sanity check for internal debug signal
    _ = [clamp_score(overlap_score(g, company_blob)) for g in final_gaps]

    return GapAnalysis(
        identified_gaps=final_gaps,
        innovation_opportunities=final_opportunities,
        startup_search_fields=final_fields,
        startup_search_queries=final_queries,
        extraction_warnings=warnings,
    )
