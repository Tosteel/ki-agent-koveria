from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from server.workflows.startup_matchup.common import (
    clean_text,
    clamp_score,
    compact_context,
    llm_json,
    load_json_obj,
    safe_list_str,
)

from .models import FinalReport, FinalReportNarrative, RecommendedStartup, StartupMatchupStep8Request


def _maybe_load(
    *,
    inline_obj: Dict[str, Any] | None,
    path: str | None,
    root_key: str,
    user_root: Path,
    work_root: Path,
) -> Dict[str, Any]:
    if isinstance(inline_obj, dict) and inline_obj:
        return load_json_obj(
            inline_obj=inline_obj,
            path=None,
            root_key=root_key,
            user_root=user_root,
            work_root=work_root,
        )
    if (path or "").strip():
        return load_json_obj(
            inline_obj=None,
            path=path,
            root_key=root_key,
            user_root=user_root,
            work_root=work_root,
        )
    return {}


def _section_report_text(
    *,
    provider: str,
    section_key: str,
    section_title: str,
    context: str,
    fallback_text: str,
    warnings: List[str],
) -> str:
    schema: Dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"report": {"type": "string"}},
        "required": ["report"],
    }
    def _is_bad_narrative(text: str) -> bool:
        t = clean_text(text)
        if len(t) < 120:
            return True
        lower = t.lower()
        banned = (
            "company_profile:",
            "recommended_startups:",
            "innovation_goals:",
            "identified_gaps:",
            "startup_search_fields:",
        )
        if any(b in lower for b in banned):
            return True
        if any(ch in t for ch in "{}[]"):
            return True
        # Heuristic for key-value dumps.
        if t.count(":") >= 6 and t.count(".") <= 2:
            return True
        return False

    prompts = [
        (
            "Du schreibst professionelle Fliesstext-Zusammenfassungen fuer Managementberichte. "
            "Nutze nur den gegebenen Kontext, keine Halluzinationen. "
            "Stil: praezise, sachlich, klar.",
            f"Sektion: {section_title}\n"
            "Erstelle einen kompakten Fliesstext (max. 1200 Zeichen), der die wichtigsten Aussagen zusammenfasst.\n"
            "Keine Listen, kein JSON, keine Feldnamen wie key:value.\n\n"
            f"Kontext:\n{context}",
        ),
        (
            "Du bist ein professioneller Berichtsautor. "
            "Verwandle strukturierte Rohdaten in einen gut lesbaren Absatz fuer Entscheider.",
            f"Sektion: {section_title}\n"
            "Schreibe genau einen Fliesstext-Absatz. "
            "Verboten: JSON-Syntax, geschweifte Klammern, eckige Klammern, Feldnamen mit Doppelpunkt.\n\n"
            f"Kontext:\n{context}",
        ),
    ]

    for idx, (system_prompt, user_prompt) in enumerate(prompts, start=1):
        parsed = llm_json(
            provider=provider,
            schema_name=f"startup_matchup_step8_report_{section_key}_{idx}",
            schema=schema,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            warnings=warnings,
        )
        txt = clean_text(parsed.get("report")) if isinstance(parsed, dict) else ""
        if txt and not _is_bad_narrative(txt):
            return txt

    fallback = clean_text(fallback_text)
    if len(fallback) > 1200:
        fallback = fallback[:1200].rsplit(" ", 1)[0] + "."
    if not fallback:
        fallback = f"Fuer den Abschnitt '{section_title}' liegen derzeit keine ausreichend belastbaren Inhalte vor."
    warnings.append(f"step8 narrative fallback used for section: {section_key}")
    return fallback


def run_step_8(*, req: StartupMatchupStep8Request, user_root: Path, work_root: Path) -> FinalReport:
    warnings: List[str] = []

    workshop = _maybe_load(
        inline_obj=req.workshop_analysis,
        path=req.workshop_analysis_path,
        root_key="workshop_analysis",
        user_root=user_root,
        work_root=work_root,
    )
    company = _maybe_load(
        inline_obj=req.company_profile,
        path=req.company_profile_path,
        root_key="company_profile",
        user_root=user_root,
        work_root=work_root,
    )
    gap = _maybe_load(
        inline_obj=req.gap_analysis,
        path=req.gap_analysis_path,
        root_key="gap_analysis",
        user_root=user_root,
        work_root=work_root,
    )
    ranked = _maybe_load(
        inline_obj=req.startup_ranked_list,
        path=req.startup_ranked_list_path,
        root_key="startup_ranked_list",
        user_root=user_root,
        work_root=work_root,
    )
    profiles = _maybe_load(
        inline_obj=req.startup_profiles,
        path=req.startup_profiles_path,
        root_key="startup_profiles",
        user_root=user_root,
        work_root=work_root,
    )

    for payload in (workshop, company, gap, ranked, profiles):
        warnings.extend(safe_list_str(payload.get("extraction_warnings"), limit=100))

    recommended: List[RecommendedStartup] = []

    profile_items = profiles.get("startups") if isinstance(profiles.get("startups"), list) else []
    if profile_items:
        sorted_profiles = sorted(
            [p for p in profile_items if isinstance(p, dict)],
            key=lambda x: float(x.get("relevance_score") or 0.0),
            reverse=True,
        )
        for idx, item in enumerate(sorted_profiles[: req.top_k], start=1):
            recommended.append(
                RecommendedStartup(
                    rank=idx,
                    startup_name=clean_text(item.get("name")),
                    relevance_score=clamp_score(item.get("relevance_score") or 0.0),
                    short_description=clean_text(item.get("description"))[:280],
                    profile=item,
                )
            )

    if not recommended:
        ranked_items = ranked.get("startups") if isinstance(ranked.get("startups"), list) else []
        sorted_ranked = sorted(
            [r for r in ranked_items if isinstance(r, dict)],
            key=lambda x: float(x.get("relevance_score") or 0.0),
            reverse=True,
        )
        for idx, item in enumerate(sorted_ranked[: req.top_k], start=1):
            recommended.append(
                RecommendedStartup(
                    rank=idx,
                    startup_name=clean_text(item.get("name") or item.get("startup_name")),
                    relevance_score=clamp_score(item.get("relevance_score") or 0.0),
                    short_description=clean_text(item.get("description") or item.get("snippet"))[:280],
                    profile=item,
                )
            )

    if not recommended:
        warnings.append("No recommended startups available in step8 input artifacts.")

    report = FinalReport(
        report=FinalReportNarrative(),
        company_profile=company,
        innovation_goals=safe_list_str(workshop.get("innovation_goals"))[:20],
        identified_gaps=safe_list_str(gap.get("identified_gaps"))[:20],
        startup_search_fields=safe_list_str(gap.get("startup_search_fields"))[:20],
        recommended_startups=recommended,
        extraction_warnings=[],
    )

    # Build narrative texts at the very end from the final assembled report payload.
    company_ctx = compact_context([report.company_profile], max_chars=req.max_context_chars)
    goals_ctx = compact_context(report.innovation_goals, max_chars=req.max_context_chars)
    gap_ctx = compact_context(
        [{"identified_gaps": report.identified_gaps}],
        max_chars=req.max_context_chars,
    )
    fields_ctx = compact_context(
        [{"startup_search_fields": report.startup_search_fields}],
        max_chars=req.max_context_chars,
    )
    startups_ctx = compact_context(
        [
            {
                "recommended_startups": [
                    {
                        "rank": r.rank,
                        "name": r.startup_name,
                        "relevance_score": r.relevance_score,
                        "short_description": r.short_description,
                    }
                    for r in report.recommended_startups
                ]
            }
        ],
        max_chars=req.max_context_chars,
    )

    fallback_company = (
        f"{clean_text(report.company_profile.get('company_name')) or 'Das Unternehmen'} ist in der Branche "
        f"{clean_text(report.company_profile.get('industry')) or 'n/a'} taetig und fokussiert sich auf "
        f"{clean_text(report.company_profile.get('core_business')) or 'sein Kerngeschaeft'}."
    )
    fallback_goals = (
        "Die priorisierten Innovationsziele sind: "
        + ", ".join(report.innovation_goals[:6])
        + "."
        if report.innovation_goals
        else "Aktuell liegen keine priorisierten Innovationsziele vor."
    )
    fallback_gaps = (
        "Die Analyse zeigt folgende zentrale Luecken: "
        + ", ".join(report.identified_gaps[:6])
        + "."
        if report.identified_gaps
        else "Aktuell wurden keine relevanten Luecken identifiziert."
    )
    fallback_fields = (
        "Die Startup-Suche fokussiert sich auf: "
        + ", ".join(report.startup_search_fields[:6])
        + "."
        if report.startup_search_fields
        else "Es wurden derzeit keine belastbaren Suchfelder definiert."
    )
    top_names = [r.startup_name for r in report.recommended_startups[:5] if clean_text(r.startup_name)]
    fallback_recommended = (
        "Als priorisierte Kooperationskandidaten wurden insbesondere "
        + ", ".join(top_names)
        + " identifiziert."
        if top_names
        else "Derzeit liegen keine priorisierten Startup-Empfehlungen vor."
    )
    fallback_conclusion = (
        "Als naechste Schritte werden eine Kurzpruefung der Top-Startups, "
        "eine priorisierte Kontaktaufnahme sowie die Vorbereitung von Pilotprojekten empfohlen."
    )

    report.report = FinalReportNarrative(
        company_profile=_section_report_text(
            provider=req.provider,
            section_key="company_profile",
            section_title="Unternehmensprofil",
            context=company_ctx,
            fallback_text=fallback_company,
            warnings=warnings,
        ),
        innovation_goals=_section_report_text(
            provider=req.provider,
            section_key="innovation_goals",
            section_title="Innovationsziele",
            context=goals_ctx,
            fallback_text=fallback_goals,
            warnings=warnings,
        ),
        gap_analysis=_section_report_text(
            provider=req.provider,
            section_key="gap_analysis",
            section_title="Gap-Analyse",
            context=gap_ctx,
            fallback_text=fallback_gaps,
            warnings=warnings,
        ),
        startup_search_fields=_section_report_text(
            provider=req.provider,
            section_key="startup_search_fields",
            section_title="Suchfelder fuer Startup-Kooperationen",
            context=fields_ctx,
            fallback_text=fallback_fields,
            warnings=warnings,
        ),
        recommended_startups=_section_report_text(
            provider=req.provider,
            section_key="recommended_startups",
            section_title="Empfohlene Startups",
            context=startups_ctx,
            fallback_text=fallback_recommended,
            warnings=warnings,
        ),
        conclusion_next_steps=_section_report_text(
            provider=req.provider,
            section_key="conclusion_next_steps",
            section_title="Fazit und naechste Schritte",
            context=compact_context(
                [
                    {"identified_gaps": report.identified_gaps[:10]},
                    {"startup_search_fields": report.startup_search_fields[:10]},
                    {
                        "recommended_startups": [
                            {
                                "rank": r.rank,
                                "name": r.startup_name,
                                "relevance_score": r.relevance_score,
                            }
                            for r in report.recommended_startups[:8]
                        ]
                    },
                ],
                max_chars=req.max_context_chars,
            ),
            fallback_text=fallback_conclusion,
            warnings=warnings,
        ),
    )
    report.report.executive_summary = _section_report_text(
        provider=req.provider,
        section_key="executive_summary",
        section_title="Executive Summary",
        context=compact_context(
            [
                {"company_profile": report.company_profile},
                {"innovation_goals": report.innovation_goals[:10]},
                {"identified_gaps": report.identified_gaps[:10]},
                {
                    "recommended_startups": [
                        {
                            "rank": r.rank,
                            "name": r.startup_name,
                            "relevance_score": r.relevance_score,
                        }
                        for r in report.recommended_startups[:8]
                    ]
                },
            ],
            max_chars=req.max_context_chars,
        ),
        fallback_text=(
            f"{fallback_company} {fallback_goals} {fallback_gaps} {fallback_recommended}"
        ),
        warnings=warnings,
    )
    report.extraction_warnings = safe_list_str(warnings, limit=200)
    return report
