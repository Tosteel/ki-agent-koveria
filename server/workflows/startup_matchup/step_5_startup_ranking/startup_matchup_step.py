from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from server.workflows.startup_matchup.common import (
    clean_text,
    clamp_score,
    domain_of,
    load_json_obj,
    normalize_website,
    overlap_score,
    safe_list_str,
)

from .models import RankedStartup, StartupMatchupStep5Request, StartupRankedList


def _collect_target_terms(gap: Dict[str, Any], company: Dict[str, Any]) -> Tuple[List[str], str]:
    terms = safe_list_str(
        safe_list_str(gap.get("startup_search_fields"))
        + safe_list_str(gap.get("identified_gaps"))
        + safe_list_str(company.get("innovation_focus"))
        + safe_list_str(company.get("technology_domains"))
        + safe_list_str(company.get("strategic_objectives"))
    )
    return terms, "\n".join(terms)


def run_step_5(*, req: StartupMatchupStep5Request, user_root: Path, work_root: Path) -> StartupRankedList:
    warnings: List[str] = []

    structured = load_json_obj(
        inline_obj=req.startup_structured_list,
        path=req.startup_structured_list_path,
        root_key="startup_structured_list",
        user_root=user_root,
        work_root=work_root,
    )
    gap = load_json_obj(
        inline_obj=req.gap_analysis,
        path=req.gap_analysis_path,
        root_key="gap_analysis",
        user_root=user_root,
        work_root=work_root,
    ) if (req.gap_analysis or req.gap_analysis_path) else {}
    company = load_json_obj(
        inline_obj=req.company_profile,
        path=req.company_profile_path,
        root_key="company_profile",
        user_root=user_root,
        work_root=work_root,
    ) if (req.company_profile or req.company_profile_path) else {}

    results = structured.get("startups") if isinstance(structured.get("startups"), list) else []
    if not results:
        warnings.append("No structured startup candidates to rank.")

    terms, terms_blob = _collect_target_terms(gap, company)

    ranked: List[RankedStartup] = []
    seen: set[Tuple[str, str]] = set()
    for item in results:
        if not isinstance(item, dict):
            continue
        name = clean_text(item.get("name") or "")
        description = clean_text(item.get("snippet") or item.get("description") or "")
        website = normalize_website(item.get("url") or item.get("website") or "")
        domain = domain_of(website)
        if not name and not domain:
            continue

        key = (name.lower(), domain)
        if key in seen:
            continue
        seen.add(key)

        candidate_blob = f"{name}\n{description}\n{domain}"
        fit_score = overlap_score(candidate_blob, terms_blob)
        field_score = 0.0
        for term in terms:
            field_score = max(field_score, overlap_score(candidate_blob, term))

        score = clamp_score(0.6 * fit_score + 0.4 * field_score)

        ranked.append(
            RankedStartup(
                name=name or clean_text(domain),
                description=description,
                website=website,
                relevance_score=score,
                score_breakdown={
                    "fit_score": round(fit_score, 4),
                    "field_score": round(field_score, 4),
                },
            )
        )

    ranked.sort(key=lambda x: (x.relevance_score, x.name), reverse=True)
    ranked = ranked[: req.top_k]

    if not ranked:
        warnings.append("Ranking produced an empty list.")

    return StartupRankedList(
        startups=ranked,
        scoring_formula="relevance_score = clamp(0.6*fit_score + 0.4*field_score)",
        extraction_warnings=warnings,
    )
