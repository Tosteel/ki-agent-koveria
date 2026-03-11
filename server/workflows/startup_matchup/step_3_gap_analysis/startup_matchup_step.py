from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, List

from server.workflows.startup_matchup.common import (
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


def _tokens(text: str) -> List[str]:
    return [t for t in clean_text(text).lower().split() if len(t) >= 3]


def _matches_any_term(query: str, terms: List[str]) -> bool:
    q_tokens = set(_tokens(query))
    if not q_tokens:
        return False
    for term in terms:
        t_tokens = set(_tokens(term))
        if not t_tokens:
            continue
        if q_tokens.intersection(t_tokens):
            return True
    return False


def _contains_required_intent(query: str, tech_terms: List[str], focus_terms: List[str], objective_terms: List[str]) -> bool:
    return (
        _matches_any_term(query, tech_terms)
        and _matches_any_term(query, focus_terms)
        and _matches_any_term(query, objective_terms)
    )


def _enrich_query_with_intent(
    query: str,
    tech_terms: List[str],
    focus_terms: List[str],
    objective_terms: List[str],
) -> str:
    q = clean_text(query)
    if not q:
        return q

    additions: List[str] = []
    if tech_terms and not _matches_any_term(q, tech_terms):
        additions.append(tech_terms[0])
    if focus_terms and not _matches_any_term(q, focus_terms):
        additions.append(focus_terms[0])
    if objective_terms and not _matches_any_term(q, objective_terms):
        additions.append(objective_terms[0])

    if additions:
        q = f"{q} {' '.join(clean_text(x) for x in additions if clean_text(x))}"
    return clean_text(q)


def _generate_profile_queries(
    *,
    tech_terms: List[str],
    focus_terms: List[str],
    objective_terms: List[str],
    max_queries: int,
) -> List[str]:
    if not tech_terms or not focus_terms or not objective_terms:
        return []

    out: List[str] = []
    seen: set[str] = set()
    for i in range(max_queries * 3):
        t = tech_terms[i % len(tech_terms)]
        f = focus_terms[(i // max(1, len(tech_terms))) % len(focus_terms)]
        o = objective_terms[(i // max(1, len(tech_terms) * len(focus_terms))) % len(objective_terms)]
        q = clean_text(
            f"startup scaleup collaboration {t} {f} {o} enterprise pilot partnership"
        )
        key = q.lower()
        if not q or key in seen:
            continue
        seen.add(key)
        out.append(q)
        if len(out) >= max_queries:
            break
    return out[:max_queries]


def _build_final_queries(
    *,
    profile_queries: List[str],
    llm_queries: List[str],
    fallback_queries: List[str],
    tech_terms: List[str],
    focus_terms: List[str],
    objective_terms: List[str],
    max_queries: int,
    warnings: List[str],
) -> List[str]:
    candidates = profile_queries + llm_queries + fallback_queries
    out: List[str] = []
    seen: set[str] = set()

    for raw in candidates:
        q = _enrich_query_with_intent(raw, tech_terms, focus_terms, objective_terms)
        key = q.lower()
        if not q or key in seen:
            continue
        if not _contains_required_intent(q, tech_terms, focus_terms, objective_terms):
            continue
        seen.add(key)
        out.append(q)
        if len(out) >= max_queries:
            break

    if len(out) < max_queries:
        generated = _generate_profile_queries(
            tech_terms=tech_terms,
            focus_terms=focus_terms,
            objective_terms=objective_terms,
            max_queries=max_queries,
        )
        for q in generated:
            key = q.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(q)
            if len(out) >= max_queries:
                break

    if not out:
        warnings.append("Step3 could not build intent-complete queries from company profile.")
    elif len(out) < max_queries:
        warnings.append(f"Step3 generated only {len(out)} intent-complete queries (requested {max_queries}).")

    return out[:max_queries]


def _normalize_gap_statement(text: str) -> str:
    t = clean_text(text)
    if not t:
        return ""
    low = t.lower()

    # Convert weak "no explicit statement" phrasing into clear capability gaps.
    weak_markers = (
        "keine expliziten aussagen",
        "keine aussagen",
        "nicht explizit",
        "keine informationen",
    )
    if any(m in low for m in weak_markers):
        # Remove generic lead-in and keep the topic phrase.
        t = re.sub(
            r"(?i)^unternehmen\s+hat\s+(?:bereits\s+)?(?:keine|nicht)\s+expliziten?\s+aussagen\s+zu\s+",
            "",
            t,
        )
        t = re.sub(r"(?i)^unternehmen\s+hat\s+(?:keine|nicht)\s+aussagen\s+zu\s+", "", t)
        t = re.sub(r"(?i)^unternehmen\s+hat\s+keine\s+informationen\s+zu\s+", "", t)
        t = re.sub(r"(?i)\s+gemacht\.?$", "", t).strip(" .")
        if t:
            t = f"Fehlende Fähigkeit in: {t}"

    # Keep gaps concise and actionable.
    t = re.sub(r"\s+", " ", t).strip()
    if len(t.split()) > 18:
        t = " ".join(t.split()[:18]).rstrip(".,;: ")
    return clean_text(t)


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
        "Formuliere identified_gaps als konkrete Faehigkeitsluecken (kurz, praezise, umsetzungsnah). "
        "Vermeide Formulierungen wie 'keine expliziten Aussagen' oder 'es gibt keine Informationen'. "
        "Liefere nur JSON gemaess Schema."
    )
    user_prompt = (
        f"Workshop:\n{compact_context([workshop], max_chars=req.max_context_chars // 2)}\n\n"
        f"Company profile:\n{compact_context([company], max_chars=req.max_context_chars // 2)}\n\n"
        "Regeln fuer identified_gaps:\n"
        "- Schreibe pro Gap eine klare Luecke/Faehigkeit, die fehlt oder unzureichend ist.\n"
        "- Keine Meta-Formulierungen ueber fehlende Aussagen/Dokumentation.\n"
        "- 6-16 Woerter pro Gap, direkt such-/matchbar fuer Startup-Scouting."
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

    final_gaps_raw = llm_gaps[:20] if llm_gaps else identified_gaps
    final_gaps = safe_list_str([_normalize_gap_statement(g) for g in final_gaps_raw], limit=20)
    if not final_gaps:
        final_gaps = identified_gaps
    final_opportunities = llm_opp[:20] if llm_opp else opportunities

    # User requirement:
    # - startup_search_fields = research_queries from Step2
    # - startup_search_queries = prefixed version of startup_search_fields
    profile_queries = safe_list_str(company.get("research_queries"))
    if profile_queries:
        final_fields = profile_queries
    else:
        final_fields = llm_fields[:15] if llm_fields else startup_search_fields
        warnings.append(
            "Step3 could not compose startup_search_fields from Step2 research_queries; fallback fields used."
        )

    prefix = "find startups, collaborations or scaleup in germany and europe for"
    final_queries: List[str] = []
    seen_queries: set[str] = set()
    for field in final_fields:
        f = clean_text(field)
        if not f:
            continue
        q = f if f.lower().startswith(prefix) else f"{prefix} {f}"
        q = clean_text(q)
        key = q.lower()
        if key in seen_queries:
            continue
        seen_queries.add(key)
        final_queries.append(q)
        if len(final_queries) >= req.max_queries:
            break

    # deterministic score sanity check for internal debug signal
    _ = [clamp_score(overlap_score(g, company_blob)) for g in final_gaps]

    return GapAnalysis(
        identified_gaps=final_gaps,
        innovation_opportunities=final_opportunities,
        startup_search_fields=final_fields,
        startup_search_queries=final_queries,
        extraction_warnings=warnings,
    )
