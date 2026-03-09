from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from server.workflows.startup_matchup.common import (
    brave_answers_text,
    clean_text,
    domain_of,
    load_json_obj,
    normalize_website,
)

from .models import DeepResearchItem, StartupDeepProfilesRaw, StartupMatchupStep6Request


def _build_query(name: str, website: str) -> str:
    n = clean_text(name)
    return (
        f"Research startup {n}. "
        "Provide concise facts: founding year, HQ location, key technologies, focus areas, products, "
        "and potential enterprise cooperation fields."
    )


def run_step_6(*, req: StartupMatchupStep6Request, user_root: Path, work_root: Path) -> StartupDeepProfilesRaw:
    warnings: List[str] = []

    ranked = load_json_obj(
        inline_obj=req.startup_ranked_list,
        path=req.startup_ranked_list_path,
        root_key="startup_ranked_list",
        user_root=user_root,
        work_root=work_root,
    )

    startups = ranked.get("startups") if isinstance(ranked.get("startups"), list) else []
    selected = [s for s in startups if isinstance(s, dict)][: req.top_n]

    research_items: List[DeepResearchItem] = []
    selected_names: List[str] = []

    for row in selected:
        name = clean_text(row.get("name") or row.get("startup_name") or "")
        website = normalize_website(row.get("website") or row.get("url") or "")
        query = _build_query(name or domain_of(website), website)
        raw_text = brave_answers_text(
            query=query,
            enable_research=req.brave_enable_research,
            stream=req.brave_stream,
            language=req.brave_language,
            country=req.brave_country,
            warnings=warnings,
        )
        if not raw_text:
            warnings.append(f"No deep research text for startup: {name or website}")

        selected_names.append(name or domain_of(website) or "Unknown Startup")
        research_items.append(
            DeepResearchItem(
                name=name or domain_of(website) or "Unknown Startup",
                website=website,
                domain=domain_of(website),
                query=query,
                raw_text=clean_text(raw_text),
                source="brave_answers",
            )
        )

    if not research_items:
        warnings.append("No startup deep research items generated.")

    return StartupDeepProfilesRaw(
        selected_startups=selected_names,
        startup_research=research_items,
        extraction_warnings=warnings,
    )
