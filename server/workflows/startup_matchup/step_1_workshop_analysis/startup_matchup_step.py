from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, List

from fastapi import HTTPException

from server.workflows.startup_matchup.common import (
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

_STEP1_USE_CASE_SCHEMA: Dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "target_use_cases": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["target_use_cases"],
}


def _short_use_case_bullets(values: List[str], *, max_items: int = 20, max_words: int = 12) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for raw in values:
        t = clean_text(raw)
        if not t:
            continue
        t = re.split(r"[,;]", t, maxsplit=1)[0]
        words = t.split()
        if len(words) > max_words:
            t = " ".join(words[:max_words]).rstrip(".,;: ")
        t = clean_text(t)
        key = t.lower()
        if not t or key in seen:
            continue
        seen.add(key)
        out.append(t)
        if len(out) >= max_items:
            break
    return out


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

    use_case_seed_context = {
        "target_use_cases_seed": target_use_cases,
        "technology_interests": technology_interests,
        "strategic_fields": strategic_fields,
        "innovation_goals": innovation_goals,
        "problem_statements": problem_statements,
    }
    use_case_parsed = llm_json(
        provider=req.provider,
        schema_name="startup_matchup_step1_use_cases",
        schema=_STEP1_USE_CASE_SCHEMA,
        system_prompt=(
            "Du formulierst konkrete target_use_cases fuer Startup-Matching. "
            "Liefere nur JSON gemaess Schema. "
            "Jeder Use Case soll moeglichst spezifisch sein und, wo passend, strategisches Feld und Technologiebezug enthalten. "
            "Formatiere als kurze Stichpunkte: bevorzugt 6-12 Woerter, keine langen Nebensaetze, keine Erklaertexte. "
            "Nutze ausschliesslich den bereitgestellten Kontext."
        ),
        user_prompt=(
            "Erzeuge aus den Input-Listen konkrete, nicht-generische target_use_cases als kurze Stichpunkte.\n"
            "Regeln:\n"
            "- Jeder Use Case soll klaren Anwendungsbezug haben.\n"
            "- Wenn sinnvoll, verbinde Use Case mit passender Technologie und strategischem Feld.\n"
            "- Pro Eintrag genau ein kurzer Use Case (keine Saetze mit mehreren Aussagen).\n"
            "- Keine neuen Themen ausserhalb des Kontexts.\n"
            f"Kontext:\n{compact_context([use_case_seed_context], max_chars=req.max_context_chars // 2)}"
        ),
        warnings=warnings,
    )
    llm_use_cases = safe_list_str(use_case_parsed.get("target_use_cases")) if isinstance(use_case_parsed, dict) else []
    if llm_use_cases:
        target_use_cases = _short_use_case_bullets(llm_use_cases, max_items=20)
    else:
        warnings.append("Step1 use-case refinement via LLM unavailable; using extracted target_use_cases.")
        target_use_cases = _short_use_case_bullets(target_use_cases, max_items=20)

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
