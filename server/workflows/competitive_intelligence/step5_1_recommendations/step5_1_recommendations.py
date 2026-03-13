from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from fastapi import HTTPException

from server.services.llm_ionos import IonosLLM
from server.services.llm_openai import LlmOpenai

from .models import Step51RecommendationsRequest, Step51RecommendationsResult


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
                    "name": "step5_1_recommendations_schema",
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
                    "name": "step5_1_recommendations_schema",
                    "schema": schema,
                    "strict": False,
                },
            },
        )
        return _parse_json_strictish(client_i.extract_text(comp))
    except Exception as exc:
        warnings.append(f"{warning_key}:llm_failed:{exc}")
        return {}


def _section_value(row: Dict[str, Any], key: str) -> str:
    return _clean_text(str(row.get(key) or ""))


def _collect_context(target: Dict[str, Any], peers: List[Dict[str, Any]], key: str) -> str:
    lines: List[str] = []
    tgt_name = _clean_text(str(target.get("company") or "Target"))
    tgt_val = _section_value(target, key)
    if tgt_val and tgt_val != "-":
        lines.append(f"TARGET {tgt_name}: {tgt_val}")
    for p in peers:
        name = _clean_text(str(p.get("company") or "Peer"))
        val = _section_value(p, key)
        if val and val != "-":
            lines.append(f"PEER {name}: {val}")
    return "\n".join(lines)


def _ratings_context(target: Dict[str, Any], peers: List[Dict[str, Any]]) -> str:
    lines: List[str] = []

    def _line(prefix: str, row: Dict[str, Any]) -> str:
        name = _clean_text(str(row.get("company") or ""))
        parts: List[str] = []
        rb = _section_value(row, "ratings_bullets")
        if rb and rb != "-":
            parts.append(f"ratings={rb}")
        if row.get("google_rating") is not None:
            parts.append(f"google_rating={row.get('google_rating')}")
        if row.get("google_review_count") is not None:
            parts.append(f"google_review_count={row.get('google_review_count')}")
        return f"{prefix} {name}: {' | '.join(parts)}" if parts else ""

    t = _line("TARGET", target)
    if t:
        lines.append(t)
    for p in peers:
        x = _line("PEER", p)
        if x:
            lines.append(x)
    return "\n".join(lines)


def _trend_context(target: Dict[str, Any], peers: List[Dict[str, Any]]) -> str:
    lines: List[str] = []

    def _extract(row: Dict[str, Any], prefix: str) -> None:
        name = _clean_text(str(row.get("company") or ""))
        items = row.get("trend_items") if isinstance(row.get("trend_items"), list) else []
        for item in items:
            if not isinstance(item, dict):
                continue
            trend_name = _clean_text(str(item.get("trend_name") or ""))
            score = item.get("match_score")
            kws = item.get("trend_keywords") if isinstance(item.get("trend_keywords"), list) else []
            kws_clean = [str(k).strip() for k in kws if _clean_text(str(k))]
            summary = _clean_text(str(item.get("summary") or ""))
            lines.append(
                f"{prefix} {name}: trend={trend_name} | score={score} | keywords={kws_clean} | summary={summary}"
            )

    _extract(target, "TARGET")
    for p in peers:
        _extract(p, "PEER")
    return "\n".join(lines)


def _comparison_context(target: Dict[str, Any], peers: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    keys = [
        "customer_segment_bullets",
        "actions_bullets",
        "ratings_bullets",
        "press_coverage_bullets",
        "social_media_reach_bullets",
    ]
    target_name = _clean_text(str(target.get("company") or "Target"))
    t_parts: List[str] = []
    for k in keys:
        v = _section_value(target, k)
        if v and v != "-":
            t_parts.append(f"{k}={v}")
    if target.get("google_rating") is not None:
        t_parts.append(f"google_rating={target.get('google_rating')}")
    if target.get("google_review_count") is not None:
        t_parts.append(f"google_review_count={target.get('google_review_count')}")
    if t_parts:
        lines.append(f"TARGET {target_name}: {' | '.join(t_parts)}")

    for p in peers:
        name = _clean_text(str(p.get("company") or "Peer"))
        parts: List[str] = []
        for k in keys:
            v = _section_value(p, k)
            if v and v != "-":
                parts.append(f"{k}={v}")
        if p.get("google_rating") is not None:
            parts.append(f"google_rating={p.get('google_rating')}")
        if p.get("google_review_count") is not None:
            parts.append(f"google_review_count={p.get('google_review_count')}")
        if parts:
            lines.append(f"PEER {name}: {' | '.join(parts)}")
    return "\n".join(lines)


def _llm_recommendation(
    *,
    provider: str,
    field_name: str,
    question: str,
    context: str,
    warnings: List[str],
) -> str:
    clean_context = _clean_text(context)
    if not clean_context:
        return "-"

    focus = {
        "customer_segements_recommendations": "Bewerte Segmentabdeckung des TARGET gegen PEERs und nenne ggf. Handlungsbedarf.",
        "actions_recommendations": "Bewerte Wettbewerbsfaehigkeit der Aktionen des TARGET und nenne konkrete Verbesserungen.",
        "ratings_recommendations": "Bewerte Rating-/Review-Position des TARGET und nenne konkrete Hebel bei Nachteil.",
        "trend_items_recommendations": (
            "Bewerte primaer die Website-Sichtbarkeit des TARGET anhand Trend-Suchbegriffen vs. PEERs. "
            "Leite daraus Website-/SEO-Handlungsbedarf ab (Begriffe gezielt aufnehmen, Content gleichziehen oder differenzieren)."
        ),
        "competitor_comparison_recommendations": "Bewerte die Gesamtposition des TARGET und nenne die wichtigsten naechsten Schritte.",
    }.get(field_name, "Bewerte TARGET gegen PEERs und gib Handlungsbedarf an.")

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }
    system_prompt = (
        "Du bist ein Competitive-Strategy-Analyst. "
        "Du gibst kurze, umsetzbare Empfehlungen basierend auf Vergleichsdaten."
    )
    user_prompt = (
        f"Feld: {field_name}\n"
        f"Frage: {question}\n"
        f"Fokus: {focus}\n"
        f"Kontext:\n{clean_context}\n\n"
        "Regeln:\n"
        "- Antworte als kurzer Fliesstext (1-4 Saetze).\n"
        "- Beziehe dich auf TARGET im Vergleich zu PEERs.\n"
        "- Wenn TARGET klar schwaecher ist: expliziter Handlungsbedarf mit 1-2 konkreten Massnahmen.\n"
        "- Bei trend_items_recommendations den Schwerpunkt auf Website/SEO legen, nicht auf Produktentwicklung.\n"
        "- Wenn keine belastbare Empfehlung moeglich ist: antworte exakt mit '-'.\n"
        "- Keine Erfindungen."
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
    return text or "-"


def run_step_5_1_recommendations(
    *,
    req: Step51RecommendationsRequest,
    user_root: Path,
    work_root: Path,
) -> Step51RecommendationsResult:
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
        return Step51RecommendationsResult(provider=provider, extraction_warnings=warnings)

    target = rows[0]
    peers = rows[1:]

    customer_ctx = _collect_context(target, peers, "customer_segment_bullets")
    actions_ctx = _collect_context(target, peers, "actions_bullets")
    ratings_ctx = _ratings_context(target, peers)
    trend_ctx = _trend_context(target, peers)
    comparison_ctx = _comparison_context(target, peers)

    customer_segements_recommendations = _llm_recommendation(
        provider=provider,
        field_name="customer_segements_recommendations",
        question="Wie schneidet das evaluierte Unternehmen bei Kundensegmenten im Vergleich zu den Wettbewerbern ab und welcher Handlungsbedarf ergibt sich?",
        context=customer_ctx,
        warnings=warnings,
    )
    actions_recommendations = _llm_recommendation(
        provider=provider,
        field_name="actions_recommendations",
        question="Wie wettbewerbsfaehig sind die Aktionen des evaluierten Unternehmens und welche Massnahmen sind sinnvoll?",
        context=actions_ctx,
        warnings=warnings,
    )
    ratings_recommendations = _llm_recommendation(
        provider=provider,
        field_name="ratings_recommendations",
        question="Wie steht das evaluierte Unternehmen bei Rating und Bewertungsanzahl da und welche konkreten Schritte sind sinnvoll?",
        context=ratings_ctx,
        warnings=warnings,
    )
    trend_items_recommendations = _llm_recommendation(
        provider=provider,
        field_name="trend_items_recommendations",
        question="Wie gut bedient das evaluierte Unternehmen die Trends im Vergleich und wo sollte priorisiert nachgeschärft werden?",
        context=trend_ctx,
        warnings=warnings,
    )
    competitor_comparison_recommendations = _llm_recommendation(
        provider=provider,
        field_name="competitor_comparison_recommendations",
        question="Welche uebergreifenden Empfehlungen ergeben sich aus der Gegenueberstellung des evaluierten Unternehmens mit den Wettbewerbern?",
        context=comparison_ctx,
        warnings=warnings,
    )

    source_urls: List[str] = []
    for row in rows:
        urls = row.get("source_urls") if isinstance(row.get("source_urls"), list) else []
        for u in urls:
            nu = _normalize_url(str(u))
            if nu and nu not in source_urls:
                source_urls.append(nu)

    return Step51RecommendationsResult(
        provider=provider,
        customer_segements_recommendations=customer_segements_recommendations,
        actions_recommendations=actions_recommendations,
        ratings_recommendations=ratings_recommendations,
        trend_items_recommendations=trend_items_recommendations,
        competitor_comparison_recommendations=competitor_comparison_recommendations,
        source_urls=source_urls,
        extraction_warnings=list(dict.fromkeys([_clean_text(w) for w in warnings if _clean_text(w)])),
    )
