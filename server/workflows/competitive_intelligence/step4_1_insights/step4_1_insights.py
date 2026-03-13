from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from fastapi import HTTPException

from server.services.llm_ionos import IonosLLM
from server.services.llm_openai import LlmOpenai

from .models import Step41InsightsRequest, Step41InsightsResult


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

    for candidate in candidates:
        if candidate.exists() and candidate.is_file() and (user_root in candidate.parents or candidate == user_root):
            return candidate
    raise HTTPException(status_code=404, detail=f"File not found: {path}")


def _load_payload(
    *,
    inline_obj: Dict[str, Any] | None,
    path: str | None,
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
    if isinstance(payload.get("matrix"), dict):
        payload = payload["matrix"]
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid payload: expected object.")
    return payload


def _extract_profiles(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = payload.get("profiles") if isinstance(payload.get("profiles"), list) else []
    return [r for r in rows if isinstance(r, dict)]


def _extract_openai_output_text(resp: Dict[str, Any]) -> str:
    out = ""
    for item in resp.get("output", []):
        for c in item.get("content", []):
            if c.get("type") == "output_text":
                out += str(c.get("text") or "")
    return _clean_text(out)


def _parse_json_strictish(text: str) -> Dict[str, Any]:
    t = _clean_text(text)
    if not t:
        return {}
    if t.startswith("```"):
        t = t.strip("`").strip()
        if "\n" in t:
            first, rest = t.split("\n", 1)
            if first.strip().lower() in {"json", "javascript"}:
                t = rest.strip()
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    start, end = t.find("{"), t.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(t[start : end + 1])
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _llm_json(
    *,
    provider: str,
    system_prompt: str,
    user_prompt: str,
    schema: Dict[str, Any],
    warnings: List[str],
    warning_key: str,
) -> Dict[str, Any]:
    p = (provider or "ionos").strip().lower()
    if p not in {"ionos", "openai"}:
        p = "ionos"
    try:
        if p == "openai":
            client = LlmOpenai()
            if not client.enabled():
                warnings.append(f"{warning_key}:openai_not_configured")
                return {}
            resp = client._call(
                input_messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                text_format={
                    "type": "json_schema",
                    "name": "step4_1_insights_schema",
                    "schema": schema,
                    "strict": False,
                },
            )
            return _parse_json_strictish(_extract_openai_output_text(resp))

        client_i = IonosLLM()
        if not client_i.enabled():
            warnings.append(f"{warning_key}:ionos_not_configured")
            return {}
        comp = client_i.chat_completions(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "step4_1_insights_schema",
                    "schema": schema,
                    "strict": False,
                },
            },
        )
        return _parse_json_strictish(client_i.extract_text(comp))
    except Exception as exc:
        warnings.append(f"{warning_key}:llm_failed:{exc}")
        return {}


def _collect_contexts(rows: List[Dict[str, Any]]) -> Dict[str, str]:
    customer_lines: List[str] = []
    actions_lines: List[str] = []
    ratings_lines: List[str] = []
    trend_lines: List[str] = []
    comparison_lines: List[str] = []

    for row in rows:
        company = _clean_text(str(row.get("company") or ""))
        seg = _clean_text(str(row.get("customer_segment_bullets") or ""))
        act = _clean_text(str(row.get("actions_bullets") or ""))
        rat = _clean_text(str(row.get("ratings_bullets") or ""))
        press = _clean_text(str(row.get("press_coverage_bullets") or ""))
        social = _clean_text(str(row.get("social_media_reach_bullets") or ""))
        google_rating = row.get("google_rating")
        google_reviews = row.get("google_review_count")

        if seg and seg != "-":
            customer_lines.append(f"{company}: {seg}")
        if act and act != "-":
            actions_lines.append(f"{company}: {act}")
        rating_parts: List[str] = []
        if rat and rat != "-":
            rating_parts.append(rat)
        if google_rating is not None:
            rating_parts.append(f"google_rating={google_rating}")
        if google_reviews is not None:
            rating_parts.append(f"google_review_count={google_reviews}")
        if rating_parts:
            ratings_lines.append(f"{company}: {' | '.join(rating_parts)}")

        trend_items = row.get("trend_items") if isinstance(row.get("trend_items"), list) else []
        for t in trend_items:
            if not isinstance(t, dict):
                continue
            trend_name = _clean_text(str(t.get("trend_name") or ""))
            trend_summary = _clean_text(str(t.get("summary") or ""))
            kws = t.get("trend_keywords") if isinstance(t.get("trend_keywords"), list) else []
            kws_clean = [str(k).strip() for k in kws if _clean_text(str(k))]
            score = t.get("match_score")
            trend_lines.append(
                f"{company}: trend={trend_name} | score={score} | keywords={kws_clean} | summary={trend_summary}"
            )

        comparison_parts: List[str] = []
        if seg:
            comparison_parts.append(f"segment={seg}")
        if act:
            comparison_parts.append(f"actions={act}")
        if rat:
            comparison_parts.append(f"ratings={rat}")
        if press:
            comparison_parts.append(f"press={press}")
        if social:
            comparison_parts.append(f"social={social}")
        if google_rating is not None:
            comparison_parts.append(f"google_rating={google_rating}")
        if google_reviews is not None:
            comparison_parts.append(f"google_review_count={google_reviews}")
        if comparison_parts:
            comparison_lines.append(f"{company}: {' | '.join(comparison_parts)}")

    return {
        "customer_segment_insights": "\n".join(customer_lines),
        "actions_insights": "\n".join(actions_lines),
        "ratings_insights": "\n".join(ratings_lines),
        "trend_items_insights": "\n".join(trend_lines),
        "competitor_comparison_insights": "\n".join(comparison_lines),
    }


def _llm_insight(
    *,
    provider: str,
    field_name: str,
    question: str,
    context: str,
    force_dash_if_empty: bool,
    warnings: List[str],
) -> str:
    clean_context = _clean_text(context)
    if not clean_context and force_dash_if_empty:
        return "-"
    if not clean_context:
        return "-"

    field_focus = {
        "customer_segment_insights": (
            "Fokussiere auf Segment-Abdeckung, Ueberschneidungen und erkennbare Zielgruppen-Luecken."
        ),
        "actions_insights": (
            "Fokussiere auf wiederkehrende Aktionsmuster vs. ungewoehnliche Aktionen und deren Differenzierungspotenzial."
        ),
        "ratings_insights": (
            "Fokussiere auf Gewinner bei Rating/Review-Volumen und den Abstand zu den anderen Unternehmen."
        ),
        "trend_items_insights": (
            "Fokussiere auf stark/ schwach bediente Trends, fehlende Trendbegriffe und klare Fuehrerschaft pro Trend. "
            "Benenne explizit, welches Unternehmen welchen Trend am staerksten bedient."
        ),
        "competitor_comparison_insights": (
            "Fokussiere auf die wichtigsten Auffaelligkeiten im direkten Vergleich (Staerken, Schwaechen, White Spots)."
        ),
    }.get(field_name, "Formuliere die wichtigsten Vergleichserkenntnisse.")

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "text": {"type": "string"},
        },
        "required": ["text"],
    }
    system_prompt = (
        "Du erzeugst belastbare Wettbewerbs-Insights auf Basis der gelieferten Daten. "
        "Keine erfundenen Inhalte."
    )
    user_prompt = (
        f"Feld: {field_name}\n"
        f"Frage: {question}\n"
        f"Kontext:\n{clean_context}\n\n"
        "Regeln:\n"
        "- Antworte als kurzer Fliesstext (1-4 Saetze), keine Stichpunkte.\n"
        "- Schreibe klar und praezise auf Deutsch.\n"
        "- Nutze nur Informationen aus dem Kontext.\n"
        "- Gib keine reine Aufzaehlung wieder, sondern leite eine Erkenntnis ab (Muster, Unterschiede, Luecken, Fuehrung).\n"
        "- Nutze Formulierungen wie 'haeufig', 'auffaellig', 'im Vergleich', 'waehrend', 'fuehrend', wenn durch Kontext gedeckt.\n"
        f"- Fokus: {field_focus}\n"
        "- Wenn die Frage mit dem Kontext nicht beantwortbar ist, antworte exakt mit '-'.\n"
        "- Keine Einleitung."
    )
    parsed = _llm_json(
        provider=provider,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=schema,
        warnings=warnings,
        warning_key=f"llm_{field_name}",
    )
    text = _clean_text(str(parsed.get("text") or ""))
    if not text:
        return "-"
    return text


def run_step_4_1_insights(
    *,
    req: Step41InsightsRequest,
    user_root: Path,
    work_root: Path,
) -> Step41InsightsResult:
    warnings: List[str] = []
    provider = _clean_text(str(req.provider or "ionos")).lower() or "ionos"
    if provider not in {"ionos", "openai"}:
        warnings.append(f"unsupported_provider:{provider};fallback_to_ionos")
        provider = "ionos"

    payload = _load_payload(
        inline_obj=req.matrix,
        path=req.matrix_path,
        user_root=user_root,
        work_root=work_root,
    )
    rows = _extract_profiles(payload)[: req.max_companies]
    if not rows:
        warnings.append("profiles_empty")
        return Step41InsightsResult(
            provider=provider,
            customer_segment_insights="-",
            actions_insights="-",
            ratings_insights="-",
            trend_items_insights="-",
            competitor_comparison_insights="-",
            extraction_warnings=warnings,
        )

    contexts = _collect_contexts(rows)

    customer_segment_insights = _llm_insight(
        provider=provider,
        field_name="customer_segment_insights",
        question="Welche Kundensegmente bedienen die Unternehmen? Gibt es Luecken?",
        context=contexts["customer_segment_insights"],
        force_dash_if_empty=True,
        warnings=warnings,
    )
    actions_insights = _llm_insight(
        provider=provider,
        field_name="actions_insights",
        question="Welche Rabatte und Aktionen kommen haeufiger vor oder sind unueblich?",
        context=contexts["actions_insights"],
        force_dash_if_empty=True,
        warnings=warnings,
    )
    ratings_insights = _llm_insight(
        provider=provider,
        field_name="ratings_insights",
        question="Welches Unternehmen hat das beste Rating? Wer die meisten Bewertungen?",
        context=contexts["ratings_insights"],
        force_dash_if_empty=False,
        warnings=warnings,
    )
    trend_items_insights = _llm_insight(
        provider=provider,
        field_name="trend_items_insights",
        question="Welche Trendbegriffe werden aufgegriffen, welche fehlen, und welcher Trend wird von welchem Unternehmen am staerksten bedient?",
        context=contexts["trend_items_insights"],
        force_dash_if_empty=False,
        warnings=warnings,
    )
    competitor_comparison_insights = _llm_insight(
        provider=provider,
        field_name="competitor_comparison_insights",
        question="Welche Aspekte fallen bei der Gegenueberstellung der Wettbewerber auf?",
        context=contexts["competitor_comparison_insights"],
        force_dash_if_empty=False,
        warnings=warnings,
    )

    source_urls: List[str] = []
    for row in rows:
        urls = row.get("source_urls") if isinstance(row.get("source_urls"), list) else []
        for u in urls:
            nu = _normalize_url(str(u))
            if nu and nu not in source_urls:
                source_urls.append(nu)

    return Step41InsightsResult(
        provider=provider,
        customer_segment_insights=customer_segment_insights or "-",
        actions_insights=actions_insights or "-",
        ratings_insights=ratings_insights or "-",
        trend_items_insights=trend_items_insights or "-",
        competitor_comparison_insights=competitor_comparison_insights or "-",
        source_urls=source_urls,
        extraction_warnings=list(dict.fromkeys([_clean_text(w) for w in warnings if _clean_text(w)])),
    )
