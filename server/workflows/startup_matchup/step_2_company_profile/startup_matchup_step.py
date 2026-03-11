from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from server.workflows.startup_matchup.common import (
    brave_answers_text,
    clean_text,
    compact_context,
    llm_json,
    load_json_obj,
    safe_list_str,
)

from .models import CompanyProfile, StartupMatchupStep2Request


_STEP2_SCHEMA: Dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "company_name": {"type": "string"},
        "industry": {"type": "string"},
        "core_business": {"type": "string"},
        "technology_domains": {"type": "array", "items": {"type": "string"}},
        "innovation_focus": {"type": "array", "items": {"type": "string"}},
        "strategic_objectives": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "company_name",
        "industry",
        "core_business",
        "technology_domains",
        "innovation_focus",
        "strategic_objectives",
    ],
}


def _infer_industry(text: str) -> str:
    t = clean_text(text).lower()
    mapping = [
        ("automotive", "Automotive"),
        ("mobility", "Mobility"),
        ("retail", "Retail"),
        ("bank", "Financial Services"),
        ("insurance", "Insurance"),
        ("health", "Healthcare"),
        ("pharma", "Healthcare"),
        ("manufacturing", "Manufacturing"),
        ("industrial", "Industrial"),
        ("energy", "Energy"),
        ("software", "Software"),
        ("saas", "Software"),
        ("logistic", "Logistics"),
    ]
    for key, value in mapping:
        if key in t:
            return value
    return ""


def _derive_profile_queries(company_name: str, workshop: Dict[str, Any], max_queries: int) -> List[str]:
    queries: List[str] = []
    c = clean_text(company_name) or "company"
    queries.append(f"{c} company profile industry core business")
    queries.append(f"{c} innovation strategy technology initiatives")

    strategic_fields = safe_list_str(workshop.get("strategic_fields"))
    for field in strategic_fields[:3]:
        queries.append(f"{c} {field} innovation partnership strategy")

    out: List[str] = []
    seen: set[str] = set()
    for q in queries:
        q_clean = clean_text(q)
        if not q_clean:
            continue
        key = q_clean.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(q_clean)
        if len(out) >= max_queries:
            break
    return out


def _derive_startup_intent_queries(
    *,
    target_use_cases: List[str],
    technology_domains: List[str],
    innovation_focus: List[str],
    strategic_objectives: List[str],
    max_queries: int,
) -> List[str]:
    # User requirement: Step2 research_queries are a 1:1 carry-over from Step1 target_use_cases.
    _ = (technology_domains, innovation_focus, strategic_objectives)  # kept for stable signature
    return safe_list_str(target_use_cases)[:max_queries]


def run_step_2(*, req: StartupMatchupStep2Request, user_root: Path, work_root: Path) -> CompanyProfile:
    warnings: List[str] = []
    workshop = load_json_obj(
        inline_obj=req.workshop_analysis,
        path=req.workshop_analysis_path,
        root_key="workshop_analysis",
        user_root=user_root,
        work_root=work_root,
    ) if (req.workshop_analysis or req.workshop_analysis_path) else {}

    company_name = clean_text(req.company_name)
    if not company_name:
        company_name = clean_text(workshop.get("company_name"))
    if not company_name:
        company_name = "Unknown Company"
        warnings.append("company_name missing; using fallback value")

    profile_queries = _derive_profile_queries(company_name, workshop, min(req.max_research_queries, 6))

    snippets: List[str] = []
    for query in profile_queries:
        text = brave_answers_text(
            query=query,
            enable_research=req.brave_enable_research,
            stream=req.brave_stream,
            language=req.brave_language,
            country=req.brave_country,
            warnings=warnings,
        )
        if text:
            snippets.append(text[:2500])

    research_context = "\n\n".join([f"Query: {q}\nResult: {s}" for q, s in zip(profile_queries, snippets)])

    system_prompt = (
        "Du erstellst ein strukturiertes Unternehmensprofil fuer Startup-Kooperationen. "
        "Nutze Workshop-Input und externe Recherche. Antworte nur als JSON gemaess Schema."
    )
    user_prompt = (
        f"Company: {company_name}\n"
        f"Workshop:\n{compact_context([workshop], max_chars=req.max_context_chars // 2)}\n\n"
        f"Research:\n{research_context[: req.max_context_chars]}"
    )
    parsed = llm_json(
        provider=req.provider,
        schema_name="startup_matchup_step2",
        schema=_STEP2_SCHEMA,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        warnings=warnings,
    )

    llm_company_name = clean_text(parsed.get("company_name")) if isinstance(parsed, dict) else ""
    llm_industry = clean_text(parsed.get("industry")) if isinstance(parsed, dict) else ""
    llm_core = clean_text(parsed.get("core_business")) if isinstance(parsed, dict) else ""

    technology_domains = safe_list_str(parsed.get("technology_domains")) if isinstance(parsed, dict) else []
    innovation_focus = safe_list_str(parsed.get("innovation_focus")) if isinstance(parsed, dict) else []
    strategic_objectives = safe_list_str(parsed.get("strategic_objectives")) if isinstance(parsed, dict) else []

    if not technology_domains:
        technology_domains = safe_list_str(workshop.get("technology_interests"))[:8]
    if not innovation_focus:
        innovation_focus = safe_list_str(workshop.get("strategic_fields"))[:8]
    if not strategic_objectives:
        strategic_objectives = safe_list_str(workshop.get("innovation_goals"))[:8]

    startup_intent_queries = _derive_startup_intent_queries(
        target_use_cases=safe_list_str(workshop.get("target_use_cases"))[:20],
        technology_domains=technology_domains,
        innovation_focus=innovation_focus,
        strategic_objectives=strategic_objectives,
        max_queries=req.max_research_queries,
    )
    if not startup_intent_queries:
        warnings.append("Could not derive startup-intent queries from profile fields.")

    if not llm_industry:
        llm_industry = _infer_industry("\n".join(snippets)) or _infer_industry(research_context)
    if not llm_core:
        llm_core = clean_text(snippets[0])[:180] if snippets else ""

    return CompanyProfile(
        company_name=llm_company_name or company_name,
        industry=llm_industry,
        core_business=llm_core,
        technology_domains=technology_domains,
        innovation_focus=innovation_focus,
        strategic_objectives=strategic_objectives,
        research_queries=startup_intent_queries,
        research_snippets=[clean_text(s)[:600] for s in snippets[:8]],
        extraction_warnings=warnings,
    )
