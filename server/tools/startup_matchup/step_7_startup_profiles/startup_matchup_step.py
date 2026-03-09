from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from server.tools.startup_matchup.common import (
    clean_text,
    clamp_score,
    compact_context,
    domain_of,
    infer_company_name_from_domain,
    llm_json,
    load_json_obj,
    location_from_text,
    normalize_website,
    overlap_score,
    pick_year_from_text,
    safe_list_str,
)

from .models import StartupMatchupStep7Request, StartupProfiles, StructuredStartupProfile


_STEP7_SCHEMA: Dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string"},
        "founding_year": {"type": "string"},
        "location": {"type": "string"},
        "technology_focus": {"type": "array", "items": {"type": "string"}},
        "description": {"type": "string"},
        "why_relevant": {"type": "string"},
        "relevance_score": {"type": "number"},
    },
    "required": [
        "name",
        "founding_year",
        "location",
        "technology_focus",
        "description",
        "why_relevant",
        "relevance_score",
    ],
}


_TECH_HINTS = [
    "ai",
    "machine learning",
    "automation",
    "robotics",
    "cloud",
    "saas",
    "data platform",
    "computer vision",
    "cybersecurity",
    "iot",
    "digital twin",
    "analytics",
    "fintech",
    "healthtech",
    "climate tech",
]


def _build_ranked_lookup(ranked: Dict[str, Any]) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    items = ranked.get("startups") if isinstance(ranked.get("startups"), list) else []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = clean_text(item.get("name") or item.get("startup_name") or "").lower()
        website = normalize_website(item.get("website") or item.get("url") or "")
        domain = domain_of(website)
        score = clamp_score(item.get("relevance_score") or 0.0)
        out[(name, domain)] = score
    return out


def _heuristic_tech_focus(text: str) -> List[str]:
    low = clean_text(text).lower()
    out: List[str] = []
    for hint in _TECH_HINTS:
        if hint in low:
            out.append(hint)
    return safe_list_str(out, limit=8)


def run_step_7(*, req: StartupMatchupStep7Request, user_root: Path, work_root: Path) -> StartupProfiles:
    warnings: List[str] = []

    raw = load_json_obj(
        inline_obj=req.startup_deep_profiles_raw,
        path=req.startup_deep_profiles_raw_path,
        root_key="startup_deep_profiles_raw",
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
    ranked = load_json_obj(
        inline_obj=req.startup_ranked_list,
        path=req.startup_ranked_list_path,
        root_key="startup_ranked_list",
        user_root=user_root,
        work_root=work_root,
    ) if (req.startup_ranked_list or req.startup_ranked_list_path) else {}

    ranking_lookup = _build_ranked_lookup(ranked)
    search_fields = safe_list_str(gap.get("startup_search_fields"))
    target_blob = "\n".join(search_fields)

    research_items = raw.get("startup_research") if isinstance(raw.get("startup_research"), list) else []
    profiles: List[StructuredStartupProfile] = []

    for item in research_items:
        if not isinstance(item, dict):
            continue
        name = clean_text(item.get("name") or "")
        website = normalize_website(item.get("website") or "")
        domain = clean_text(item.get("domain") or domain_of(website))
        raw_text = clean_text(item.get("raw_text") or "")
        if not name:
            name = infer_company_name_from_domain(website) or "Unknown Startup"

        fallback_year = pick_year_from_text(raw_text)
        fallback_location = location_from_text(raw_text)
        fallback_focus = _heuristic_tech_focus(raw_text)
        overlap = overlap_score(raw_text, target_blob)

        rank_score = ranking_lookup.get((name.lower(), domain), 0.0)
        fallback_score = clamp_score(0.6 * overlap + 0.4 * rank_score)

        fallback_profile = {
            "name": name,
            "founding_year": fallback_year,
            "location": fallback_location,
            "technology_focus": fallback_focus,
            "description": raw_text[:400],
            "why_relevant": (
                f"Matches search fields with overlap score {round(overlap, 3)} and ranked score {round(rank_score, 3)}."
            ),
            "relevance_score": fallback_score,
        }

        system_prompt = (
            "Du strukturierst einen Startup-Rohtext in ein standardisiertes Profil fuer Kooperationsentscheidungen. "
            "Liefere nur JSON gemaess Schema und nutze nur nachweisbare Informationen aus dem Text."
        )
        user_prompt = (
            f"Startup: {name}\nWebsite: {website}\n"
            f"Search fields:\n{compact_context(search_fields, max_chars=3000)}\n\n"
            f"Raw text:\n{raw_text[: req.max_context_chars]}"
        )
        parsed = llm_json(
            provider=req.provider,
            schema_name="startup_matchup_step7_profile",
            schema=_STEP7_SCHEMA,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            warnings=warnings,
        )

        source = parsed if isinstance(parsed, dict) and parsed else fallback_profile

        profiles.append(
            StructuredStartupProfile(
                name=clean_text(source.get("name")) or name,
                founding_year=clean_text(source.get("founding_year")) or fallback_year,
                location=clean_text(source.get("location")) or fallback_location,
                technology_focus=safe_list_str(source.get("technology_focus")) or fallback_focus,
                description=clean_text(source.get("description")) or fallback_profile["description"],
                why_relevant=clean_text(source.get("why_relevant")) or fallback_profile["why_relevant"],
                relevance_score=clamp_score(source.get("relevance_score") if isinstance(source, dict) else fallback_score),
                website=website,
                domain=domain,
            )
        )

    profiles.sort(key=lambda x: (x.relevance_score, x.name), reverse=True)
    if not profiles:
        warnings.append("No structured startup profiles generated.")

    return StartupProfiles(startups=profiles, extraction_warnings=warnings)
