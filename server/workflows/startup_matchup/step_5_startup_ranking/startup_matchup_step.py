from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from server.workflows.startup_matchup.common import (
    clean_text,
    clamp_score,
    compact_context,
    domain_of,
    llm_json,
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


_STEP5_SCORE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "fit_score": {"type": "number"},
        "field_score": {"type": "number"},
    },
    "required": ["fit_score", "field_score"],
}


def _quantize_score(value: float, step: float = 0.05) -> float:
    s = max(0.0, min(1.0, float(value or 0.0)))
    if step <= 0:
        return clamp_score(s)
    return clamp_score(round(s / step) * step)


def _llm_scores_for_startup(
    *,
    provider: str,
    startup_name: str,
    startup_description: str,
    startup_website: str,
    terms: List[str],
    max_context_chars: int,
    warnings: List[str],
) -> Tuple[float, float]:
    candidate_blob = f"{startup_name}\n{startup_description}\n{domain_of(startup_website)}"
    terms_blob = "\n".join(terms)
    lexical_fit = overlap_score(candidate_blob, terms_blob)
    lexical_field = 0.0
    for term in terms:
        lexical_field = max(lexical_field, overlap_score(candidate_blob, term))

    target_context = compact_context([{"target_terms": terms[:50]}], max_chars=max_context_chars // 2)
    startup_context = compact_context(
        [
            {
                "startup_name": startup_name,
                "startup_description": startup_description,
                "startup_website": startup_website,
            }
        ],
        max_chars=max_context_chars // 2,
    )

    parsed = llm_json(
        provider=provider,
        schema_name="startup_matchup_step5_scores",
        schema=_STEP5_SCORE_SCHEMA,
        system_prompt=(
            "Bewerte die strategische Relevanz eines Startups fuer ein Unternehmen. "
            "Gib nur JSON zurueck. "
            "fit_score und field_score muessen jeweils zwischen 0.0 und 1.0 liegen."
        ),
        user_prompt=(
            "Bewerte dieses Startup anhand der Zielterme.\n\n"
            f"Startup:\n{startup_context}\n\n"
            f"Zielterme:\n{target_context}\n\n"
            "Definitionen:\n"
            "- fit_score: Gesamtpassung des Startups zu den Zieltermen.\n"
            "- field_score: Passung zu den wichtigsten Suchfeldern/Technologiefeldern.\n"
        ),
        warnings=warnings,
    )

    if not isinstance(parsed, dict) or ("fit_score" not in parsed and "field_score" not in parsed):
        warnings.append(f"step5 llm scoring unavailable for startup '{startup_name}'; lexical fallback used")
        return _quantize_score(lexical_fit), _quantize_score(lexical_field)

    fit = _quantize_score(parsed.get("fit_score"))
    field = _quantize_score(parsed.get("field_score"))
    if fit == 0.0 and field == 0.0 and (lexical_fit > 0.0 or lexical_field > 0.0):
        warnings.append(f"step5 llm returned zero scores for '{startup_name}'; lexical fallback applied")
        fit = _quantize_score(lexical_fit)
        field = _quantize_score(lexical_field)
    return fit, field


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
    if not terms:
        warnings.append("No target terms from gap/company profile; LLM scoring may be weak.")

    ranked: List[RankedStartup] = []
    seen: set[Tuple[str, str]] = set()
    llm_scored = 0
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
        # LLM scoring per startup until top_k is reached (user requirement).
        if llm_scored < req.top_k:
            fit_score, field_score = _llm_scores_for_startup(
                provider=req.provider,
                startup_name=name,
                startup_description=description,
                startup_website=website,
                terms=terms,
                max_context_chars=req.max_context_chars,
                warnings=warnings,
            )
            llm_scored += 1
        else:
            # Fallback for remaining candidates if present.
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
