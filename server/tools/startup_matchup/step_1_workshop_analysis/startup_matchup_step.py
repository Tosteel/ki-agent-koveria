from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from fastapi import HTTPException

from server.tools.startup_matchup.common import (
    classify_bullets_heuristic,
    clean_text,
    compact_context,
    extract_bullets_from_text,
    llm_json,
    read_document_text,
    resolve_input_path,
    safe_list_str,
)

from .models import StartupMatchupStep1Request, WorkshopAnalysis


_STEP1_SCHEMA: Dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "innovation_goals": {"type": "array", "items": {"type": "string"}},
        "strategic_fields": {"type": "array", "items": {"type": "string"}},
        "problem_statements": {"type": "array", "items": {"type": "string"}},
        "technology_interests": {"type": "array", "items": {"type": "string"}},
        "target_use_cases": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "innovation_goals",
        "strategic_fields",
        "problem_statements",
        "technology_interests",
        "target_use_cases",
    ],
}


def run_step_1(*, req: StartupMatchupStep1Request, user_root: Path, work_root: Path) -> WorkshopAnalysis:
    warnings: List[str] = []
    source_path = ""

    if (req.workshop_text or "").strip():
        raw_text = str(req.workshop_text or "")
    else:
        resolved = resolve_input_path(
            req.workshop_document_path or "",
            user_root=user_root,
            work_root=work_root,
        )
        source_path = str(req.workshop_document_path or "")
        raw_text = read_document_text(resolved, max_chars=req.max_chars)

    cleaned_text = clean_text(raw_text)
    if not cleaned_text:
        raise HTTPException(status_code=400, detail="Workshop input is empty.")

    bullets = extract_bullets_from_text(raw_text, limit=250)
    if not bullets:
        bullets = [cleaned_text[:500]]
        warnings.append("No explicit bullet points detected; used plain text fallback.")

    heuristic = classify_bullets_heuristic(bullets)

    system_prompt = (
        "Du strukturierst Workshop-Stichpunkte fuer die Startup-Kooperationssuche. "
        "Liefere nur JSON gemaess Schema. "
        "Keine Halluzinationen, nur Informationen aus dem Kontext."
    )
    user_prompt = (
        "Extrahiere und clustere die Inhalte in die Zielkategorien.\n"
        f"Kontext:\n{compact_context(bullets, max_chars=req.max_context_chars)}"
    )
    parsed = llm_json(
        provider=req.provider,
        schema_name="startup_matchup_step1",
        schema=_STEP1_SCHEMA,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        warnings=warnings,
    )

    def _pick(key: str) -> List[str]:
        llm_vals = safe_list_str(parsed.get(key)) if isinstance(parsed, dict) else []
        if llm_vals:
            return llm_vals[:20]
        return heuristic.get(key, [])[:20]

    innovation_goals = _pick("innovation_goals")
    strategic_fields = _pick("strategic_fields")
    problem_statements = _pick("problem_statements")
    technology_interests = _pick("technology_interests")
    target_use_cases = _pick("target_use_cases")

    merged_topics = []
    for entry in innovation_goals + strategic_fields + problem_statements + technology_interests + target_use_cases:
        if entry not in merged_topics:
            merged_topics.append(entry)

    return WorkshopAnalysis(
        innovation_goals=innovation_goals,
        strategic_fields=strategic_fields,
        problem_statements=problem_statements,
        technology_interests=technology_interests,
        target_use_cases=target_use_cases,
        extracted_topics=merged_topics[:30],
        source_path=source_path,
        extraction_warnings=warnings,
    )
