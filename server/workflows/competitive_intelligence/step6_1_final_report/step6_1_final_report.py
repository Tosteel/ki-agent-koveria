from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from fastapi import HTTPException

from server.services.llm_ionos import IonosLLM
from server.services.llm_openai import LlmOpenai

from .models import Step61FinalReportChapters, Step61FinalReportRequest, Step61FinalReportResult


def _clean_text(value: str) -> str:
    return " ".join(str(value or "").split()).strip()


def _normalize_url(value: str) -> str:
    raw = _clean_text(str(value or "")).rstrip(".,;:")
    if not raw:
        return ""
    if any(ch in raw for ch in ("\n", "\r", "\t", " ", "\\", "\u2026")):
        return ""
    if not (raw.startswith("http://") or raw.startswith("https://")):
        return ""
    return raw


def _resolve_input_path(path: str, *, user_root: Path, work_root: Path) -> Path:
    raw = str(path or "").strip().lstrip("/")
    p = Path(raw)
    if not raw or p.is_absolute() or ".." in p.parts:
        raise HTTPException(status_code=400, detail=f"Invalid path: {path}")
    user_root = user_root.resolve()
    work_root = work_root.resolve()
    uploads_root = (user_root / "uploads").resolve()
    uploads_root.mkdir(parents=True, exist_ok=True)

    candidates: List[Path] = []
    parts = list(p.parts)
    if parts and parts[0] == "uploads":
        candidates.append((uploads_root / Path(*parts[1:])).resolve())
    elif parts and parts[0] == "work":
        candidates.append((work_root / Path(*parts[1:])).resolve())
    else:
        candidates.append((work_root / p).resolve())
        candidates.append((uploads_root / p).resolve())

    for c in candidates:
        if c.exists() and c.is_file() and (user_root in c.parents or c == user_root):
            return c
    raise HTTPException(status_code=404, detail=f"File not found: {path}")


def _load_payload(
    *,
    inline_obj: Dict[str, Any] | None,
    path: str | None,
    wrapper_key: str,
    user_root: Path,
    work_root: Path,
) -> Dict[str, Any]:
    if isinstance(inline_obj, dict) and inline_obj:
        payload = inline_obj
    else:
        resolved = _resolve_input_path(str(path or ""), user_root=user_root, work_root=work_root)
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON in path: {path}") from exc
    if isinstance(payload.get(wrapper_key), dict):
        payload = payload[wrapper_key]
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail=f"Invalid payload for {wrapper_key}: expected object.")
    return payload


def _extract_openai_output_text(resp: Dict[str, Any]) -> str:
    out = ""
    for item in resp.get("output", []):
        for c in item.get("content", []):
            if c.get("type") == "output_text":
                out += str(c.get("text") or "")
    return _clean_text(out)


def _llm_text(*, provider: str, system_prompt: str, user_prompt: str, warnings: List[str], warning_key: str) -> str:
    p = (provider or "ionos").strip().lower()
    if p not in {"ionos", "openai"}:
        p = "ionos"
    try:
        if p == "openai":
            client = LlmOpenai()
            if not client.enabled():
                warnings.append(f"{warning_key}:openai_not_configured")
                return ""
            resp = client._call(
                input_messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            return _extract_openai_output_text(resp)

        client_i = IonosLLM()
        if not client_i.enabled():
            warnings.append(f"{warning_key}:ionos_not_configured")
            return ""
        comp = client_i.chat_completions(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        return _clean_text(client_i.extract_text(comp))
    except Exception as exc:
        warnings.append(f"{warning_key}:llm_failed:{exc}")
        return ""


def _chapter_2_company_profiles(*, provider: str, competitor_trends: Dict[str, Any], max_companies: int, warnings: List[str]) -> str:
    profiles = competitor_trends.get("profiles") if isinstance(competitor_trends.get("profiles"), list) else []
    rows = [r for r in profiles if isinstance(r, dict)][:max_companies]
    if not rows:
        return "-"
    lines: List[str] = []
    for r in rows:
        company = _clean_text(str(r.get("company") or ""))
        region = _clean_text(str(r.get("region") or ""))
        prof = r.get("company_profile_target_audience") if isinstance(r.get("company_profile_target_audience"), dict) else {}
        actions = r.get("offers_actions") if isinstance(r.get("offers_actions"), dict) else {}
        ratings = r.get("ratings_reach") if isinstance(r.get("ratings_reach"), dict) else {}
        lines.append(
            f"{company} ({region}) | Profil: {_clean_text(str(prof.get('summary') or '-'))} | "
            f"Aktionen: {_clean_text(str(actions.get('summary') or '-'))} | "
            f"Ratings: {_clean_text(str(ratings.get('summary') or '-'))}"
        )
    context = "\n".join(lines)
    out = _llm_text(
        provider=provider,
        system_prompt="Erstelle Kapiteltext fuer Unternehmensprofile aus strukturierten Daten.",
        user_prompt=(
            "Erstelle Kapitel 2 'Unternehmensprofile' als klaren Fliesstext (3-8 Saetze).\n"
            "Nutze nur den Kontext.\n"
            f"Kontext:\n{context}"
        ),
        warnings=warnings,
        warning_key="chapter2_profiles",
    )
    return out or context


def _chapter_3_matrix(matrix: Dict[str, Any], max_companies: int) -> str:
    profiles = matrix.get("profiles") if isinstance(matrix.get("profiles"), list) else []
    rows = [r for r in profiles if isinstance(r, dict)][:max_companies]
    if not rows:
        return "-"

    companies = [_clean_text(str(r.get("company") or "")) or f"Company_{i+1}" for i, r in enumerate(rows)]
    variables: List[tuple[str, str]] = [
        ("customer_segment_bullets", "Kundensegment"),
        ("actions_bullets", "Aktionen"),
        ("ratings_bullets", "Ratings"),
        ("press_coverage_bullets", "Presse"),
        ("google_rating", "Google Bewertung"),
        ("google_review_count", "Anzahl Bewertungen"),
        ("social_media_reach_bullets", "Social Media Reichweite"),
    ]
    trend_names: List[str] = []
    for r in rows:
        trend_items = r.get("trend_items") if isinstance(r.get("trend_items"), list) else []
        for t in trend_items:
            if not isinstance(t, dict):
                continue
            tn = _clean_text(str(t.get("trend_name") or ""))
            if tn and tn not in trend_names:
                trend_names.append(tn)

    header = "| Variable | " + " | ".join(companies) + " |"
    sep = "|---|" + "|".join(["---"] * len(companies)) + "|"
    lines = [header, sep]

    for key, label in variables:
        vals: List[str] = []
        for r in rows:
            v = r.get(key)
            vals.append(_clean_text(str(v if v is not None else "-")) or "-")
        lines.append("| " + label + " | " + " | ".join(vals) + " |")

    for tn in trend_names:
        vals: List[str] = []
        for r in rows:
            trend_items = r.get("trend_items") if isinstance(r.get("trend_items"), list) else []
            found = "-"
            for t in trend_items:
                if not isinstance(t, dict):
                    continue
                if _clean_text(str(t.get("trend_name") or "")) == tn:
                    score = t.get("match_score")
                    kws = t.get("trend_keywords") if isinstance(t.get("trend_keywords"), list) else []
                    kw = ", ".join([_clean_text(str(x)) for x in kws if _clean_text(str(x))][:5])
                    found = f"score={score}; keywords={kw or '-'}"
                    break
            vals.append(found)
        lines.append("| Trend: " + tn + " | " + " | ".join(vals) + " |")

    return "\n".join(lines)


def _chapter_4_insights(insights: Dict[str, Any]) -> str:
    parts = [
        f"Kundensegmente: {_clean_text(str(insights.get('customer_segment_insights') or '-'))}",
        f"Aktionen: {_clean_text(str(insights.get('actions_insights') or '-'))}",
        f"Ratings: {_clean_text(str(insights.get('ratings_insights') or '-'))}",
        f"Trends: {_clean_text(str(insights.get('trend_items_insights') or '-'))}",
        f"Gegenueberstellung: {_clean_text(str(insights.get('competitor_comparison_insights') or '-'))}",
    ]
    return "\n".join(parts)


def _chapter_5_recommendations(recommendations: Dict[str, Any]) -> str:
    parts = [
        f"Kundensegmente: {_clean_text(str(recommendations.get('customer_segements_recommendations') or '-'))}",
        f"Aktionen: {_clean_text(str(recommendations.get('actions_recommendations') or '-'))}",
        f"Ratings: {_clean_text(str(recommendations.get('ratings_recommendations') or '-'))}",
        f"Trends: {_clean_text(str(recommendations.get('trend_items_recommendations') or '-'))}",
        f"Gesamtvergleich: {_clean_text(str(recommendations.get('competitor_comparison_recommendations') or '-'))}",
    ]
    return "\n".join(parts)


def _chapter_6_appendix_trends(step1_3: Dict[str, Any]) -> str:
    summaries = step1_3.get("summaries") if isinstance(step1_3.get("summaries"), list) else []
    rows = [r for r in summaries if isinstance(r, dict)]
    if not rows:
        return "-"

    lines: List[str] = []
    for idx, row in enumerate(rows, start=1):
        summary = _clean_text(str(row.get("summary") or ""))
        if not summary or summary == "-":
            continue
        lines.append(f"Trend {idx}: {summary}")

        evidence = row.get("evidence_points") if isinstance(row.get("evidence_points"), list) else []
        evidence_clean = [_clean_text(str(x)) for x in evidence if _clean_text(str(x))]
        for ev in evidence_clean[:5]:
            lines.append(f"- Evidenz: {ev}")

        urls = row.get("source_urls") if isinstance(row.get("source_urls"), list) else []
        urls_clean = [_normalize_url(str(u)) for u in urls]
        urls_clean = [u for u in urls_clean if u]
        for u in urls_clean[:8]:
            lines.append(f"- Quelle: {u}")

        source_count = row.get("source_count")
        if isinstance(source_count, int) and source_count > 0:
            lines.append(f"- Anzahl Quellen: {source_count}")

        lines.append("")

    return "\n".join(lines).strip() or "-"


def _chapter_1_executive_summary(*, provider: str, ch2: str, ch3: str, ch4: str, ch5: str, warnings: List[str]) -> str:
    def _strip_exec_label(text: str) -> str:
        t = str(text or "").strip()
        t = t.replace("**Executive Summary**", "").replace("Executive Summary", "").strip()
        t = t.replace("Zusammenfassung:", "").strip()
        return _clean_text(t) or "-"

    out = _llm_text(
        provider=provider,
        system_prompt="Erstelle eine praegnante Zusammenfassung auf Basis mehrerer Report-Kapitel.",
        user_prompt=(
            "Erstelle Kapitel 1 'Zusammenfassung' (4-8 Saetze), als Zusammenfassung der Kapitel 2-5.\n"
            "Kein Titel oder Label im Text. Nicht 'Executive Summary' schreiben.\n"
            "Hebe wichtigste Befunde und Prioritaeten hervor.\n"
            f"Kapitel 2:\n{ch2}\n\nKapitel 3:\n{ch3}\n\nKapitel 4:\n{ch4}\n\nKapitel 5:\n{ch5}"
        ),
        warnings=warnings,
        warning_key="chapter1_executive_summary",
    )
    return _strip_exec_label(out)


def _build_report_markdown(chapters: Step61FinalReportChapters) -> str:
    return (
        "# Competitive Intelligence Report\n\n"
        "## Kapitel 1: Zusammenfassung\n"
        f"{chapters.chapter_1_executive_summary}\n\n"
        "## Kapitel 2: Unternehmensprofile\n"
        f"{chapters.chapter_2_company_profiles}\n\n"
        "## Kapitel 3: Wettbewerbsvergleich\n"
        f"{chapters.chapter_3_company_matrix}\n\n"
        "## Kapitel 4: Insights\n"
        f"{chapters.chapter_4_insights}\n\n"
        "## Kapitel 5: Handlungsempfehlungen\n"
        f"{chapters.chapter_5_recommendations}\n\n"
        "## Kapitel 6: Anhang: Quellen zur Trenderfassung\n"
        f"{chapters.chapter_6_appendix_trends}\n"
    )


def run_step_6_1_final_report(
    *,
    req: Step61FinalReportRequest,
    user_root: Path,
    work_root: Path,
) -> Step61FinalReportResult:
    warnings: List[str] = []
    provider = _clean_text(str(req.provider or "ionos")).lower() or "ionos"
    if provider not in {"ionos", "openai"}:
        warnings.append(f"unsupported_provider:{provider};fallback_to_ionos")
        provider = "ionos"

    step1_3 = _load_payload(
        inline_obj=req.market_trends_summary,
        path=req.market_trends_summary_path,
        wrapper_key="market_trends_summary",
        user_root=user_root,
        work_root=work_root,
    )
    step2_4 = _load_payload(
        inline_obj=req.competitor_trends,
        path=req.competitor_trends_path,
        wrapper_key="competitor_trends",
        user_root=user_root,
        work_root=work_root,
    )
    step3_1 = _load_payload(
        inline_obj=req.matrix,
        path=req.matrix_path,
        wrapper_key="matrix",
        user_root=user_root,
        work_root=work_root,
    )
    step4_1 = _load_payload(
        inline_obj=req.insights,
        path=req.insights_path,
        wrapper_key="insights",
        user_root=user_root,
        work_root=work_root,
    )
    step5_1 = _load_payload(
        inline_obj=req.recommendations,
        path=req.recommendations_path,
        wrapper_key="recommendations",
        user_root=user_root,
        work_root=work_root,
    )

    chapter_2 = _chapter_2_company_profiles(
        provider=provider,
        competitor_trends=step2_4,
        max_companies=req.max_companies,
        warnings=warnings,
    )
    chapter_3 = _chapter_3_matrix(step3_1, req.max_companies)
    chapter_4 = _chapter_4_insights(step4_1)
    chapter_5 = _chapter_5_recommendations(step5_1)
    chapter_6 = _chapter_6_appendix_trends(step1_3)
    chapter_1 = _chapter_1_executive_summary(
        provider=provider,
        ch2=chapter_2,
        ch3=chapter_3,
        ch4=chapter_4,
        ch5=chapter_5,
        warnings=warnings,
    )

    chapters = Step61FinalReportChapters(
        chapter_1_executive_summary=chapter_1,
        chapter_2_company_profiles=chapter_2,
        chapter_3_company_matrix=chapter_3,
        chapter_4_insights=chapter_4,
        chapter_5_recommendations=chapter_5,
        chapter_6_appendix_trends=chapter_6,
    )
    report_markdown = _build_report_markdown(chapters)

    source_urls: List[str] = []
    for payload in (step1_3, step2_4, step3_1, step4_1, step5_1):
        urls = payload.get("source_urls") if isinstance(payload.get("source_urls"), list) else []
        for u in urls:
            nu = _normalize_url(str(u))
            if nu and nu not in source_urls:
                source_urls.append(nu)
        profiles = payload.get("profiles") if isinstance(payload.get("profiles"), list) else []
        for p in profiles:
            if not isinstance(p, dict):
                continue
            p_urls = p.get("source_urls") if isinstance(p.get("source_urls"), list) else []
            for u in p_urls:
                nu = _normalize_url(str(u))
                if nu and nu not in source_urls:
                    source_urls.append(nu)

    return Step61FinalReportResult(
        provider=provider,
        chapters=chapters,
        report_markdown=report_markdown,
        source_urls=source_urls,
        extraction_warnings=list(dict.fromkeys([_clean_text(w) for w in warnings if _clean_text(w)])),
    )
