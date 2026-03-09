from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from server.workflows.startup_matchup.common import (
    clean_text,
    infer_company_name_from_domain,
    llm_json,
    load_json_obj,
    normalize_website,
)

from .models import StartupMatchupStep41Request, StartupStructuredList, StructuredStartupCandidate


_STEP41_ITEM_SCHEMA: Dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "url": {"type": "string"},
    },
    "required": ["name", "description", "url"],
}

_STEP41_SNIPPET_SCHEMA: Dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "startups": {
            "type": "array",
            "items": _STEP41_ITEM_SCHEMA,
        },
    },
    "required": ["startups"],
}


def _fallback_name(description: str, url: str) -> str:
    inferred = infer_company_name_from_domain(url)
    if inferred:
        return inferred
    text = clean_text(description)
    return text[:60] if text else ""


def run_step_41(*, req: StartupMatchupStep41Request, user_root: Path, work_root: Path) -> StartupStructuredList:
    warnings: List[str] = []

    raw = load_json_obj(
        inline_obj=req.startup_candidates_raw,
        path=req.startup_candidates_raw_path,
        root_key="startup_candidates_raw",
        user_root=user_root,
        work_root=work_root,
    )

    search_results = raw.get("search_results") if isinstance(raw.get("search_results"), list) else []
    startups: List[StructuredStartupCandidate] = []
    seen: set[tuple[str, str]] = set()

    for idx, item in enumerate(search_results, start=1):
        if not isinstance(item, dict):
            continue

        raw_description = clean_text(item.get("snippet") or item.get("description") or "")
        raw_url = normalize_website(item.get("url") or item.get("website") or "")
        raw_source = clean_text(item.get("source") or "")

        if not raw_description and not raw_url:
            continue

        system_prompt = (
            "Du strukturierst Startup-Suchergebnisse in ein festes JSON-Format. "
            "Nutze nur Fakten aus dem Input. "
            "Wichtig: Das Feld description kann ein eingebettetes JSON mit search_results enthalten. "
            "Extrahiere dann alle enthaltenen Startups. "
            "Wenn die description kein solches JSON enthaelt, erzeuge genau einen Startup-Eintrag aus der description. "
            "Wenn Name oder URL unklar sind, leer lassen statt raten."
        )
        user_prompt = (
            f"search_result_index: {idx}\n"
            f"description: {raw_description[: req.max_context_chars]}\n"
            f"url: {raw_url}\n"
            f"source: {raw_source}"
        )

        parsed = llm_json(
            provider=req.provider,
            schema_name="startup_matchup_step41_snippet",
            schema=_STEP41_SNIPPET_SCHEMA,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            warnings=warnings,
        )

        llm_rows = parsed.get("startups") if isinstance(parsed, dict) and isinstance(parsed.get("startups"), list) else []
        if not llm_rows:
            warnings.append(f"step4.1 llm parsing empty for search_result #{idx}; fallback used")
            llm_rows = [{"name": "", "description": raw_description, "url": raw_url}]

        for row in llm_rows:
            if not isinstance(row, dict):
                continue
            llm_name = clean_text(row.get("name"))
            llm_description = clean_text(row.get("description"))
            llm_url = normalize_website(row.get("url"))

            name = llm_name or _fallback_name(llm_description or raw_description, llm_url or raw_url)
            description = llm_description or raw_description
            url = llm_url or raw_url

            if not description and not url:
                continue

            key = (name.lower(), url.lower())
            if key in seen:
                continue
            seen.add(key)

            startups.append(
                StructuredStartupCandidate(
                    name=name,
                    description=description,
                    url=url,
                )
            )

    if not startups:
        warnings.append("No structured startup entries created from step4 search_results.")

    return StartupStructuredList(startups=startups, extraction_warnings=warnings)
