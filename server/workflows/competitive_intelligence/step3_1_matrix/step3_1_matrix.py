from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from fastapi import HTTPException

from server.services.llm_ionos import IonosLLM
from server.services.llm_openai import LlmOpenai

from .models import (
    Step31CompanyMatrixProfile,
    Step31MatrixRequest,
    Step31MatrixResult,
    Step31TrendMatrixItem,
)


_SPLIT_SENTENCES_RE = re.compile(r"[.!?]\s+|\n+")


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
    if isinstance(payload.get("competitor_trends"), dict):
        payload = payload["competitor_trends"]
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
                    "name": "step3_1_matrix_schema",
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
                    "name": "step3_1_matrix_schema",
                    "schema": schema,
                    "strict": False,
                },
            },
        )
        return _parse_json_strictish(client_i.extract_text(comp))
    except Exception as exc:
        warnings.append(f"{warning_key}:llm_failed:{exc}")
        return {}


def _compact_bullet_text(value: str, *, max_words: int = 12, max_chars: int = 100) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    text = re.sub(r"^[\-•\*\d\.\)\s]+", "", text)
    text = text.strip(" .;,:")
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words]).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip(" ,;:")
    return text


def _sanitize_summary_text(value: Any) -> str:
    raw = _clean_text(str(value or ""))
    if raw == "-":
        return "-"
    return _compact_bullet_text(raw, max_words=28, max_chars=280)


def _fallback_trend_items(trend_rows: List[Dict[str, Any]], *, max_trend_bullets: int) -> List[Step31TrendMatrixItem]:
    out: List[Step31TrendMatrixItem] = []
    for t in trend_rows:
        trend_name = _clean_text(str(t.get("trend_summary") or ""))
        keywords = [str(k).strip() for k in (t.get("keywords") if isinstance(t.get("keywords"), list) else []) if _clean_text(str(k))]
        try:
            score = float(t.get("match_score") or 0.0)
        except Exception:
            score = 0.0
        summary_parts = []
        if keywords:
            summary_parts.append(f"Keywords: {', '.join(keywords[:6])}")
        summary_parts.append(f"Match-Score: {round(max(0.0, min(1.0, score)), 4)}")
        out.append(
            Step31TrendMatrixItem(
                trend_name=trend_name,
                match_score=round(max(0.0, min(1.0, score)), 4),
                trend_keywords=keywords,
                summary=_clean_text(" | ".join(summary_parts)),
            )
        )
    return out


def _llm_section_text(
    *,
    provider: str,
    company: str,
    section_label: str,
    context_text: str,
    max_items: int,
    warnings: List[str],
) -> str:
    context = _clean_text(context_text)
    if not context:
        return "-"

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "text": {"type": "string"},
        },
        "required": ["text"],
    }

    system_prompt = (
        "Du extrahierst kurze Stichpunkte aus einem gegebenen Kontext. "
        "Nur Fakten aus dem Kontext. Keine Erfindungen."
    )
    user_prompt = (
        f"Unternehmen: {company}\n"
        f"Feld: {section_label}\n"
        f"Kontext: {context}\n\n"
        "Regeln:\n"
        f"- Gib einen kompakten String zur Zusammenfassung aus (kein Array, keine Liste).\n"
        f"- Decke die wichtigsten Aspekte ab, Anzahl Aspekte frei waehlbar.\n"
        "- Form: kurze, verdichtete Phrase mit Trennzeichen ';'.\n"
        "- Keine erfundenen Inhalte.\n"
        "- Wenn kein verwertbarer Inhalt da ist: antworte exakt mit '-'.\n"
        "- Verboten sind Negativsaetze wie 'Keine ... gefunden', 'Keine Uebereinstimmungen gefunden', "
        "'Nicht verfuegbar'. In diesen Faellen nur '-'."
    )
    if section_label == "ratings_bullets":
        user_prompt += (
            "\n- Keine numerischen Bewertungen oder Anzahlen ausgeben "
            "(keine Sternwerte, keine Review-Counts), nur qualitative Aussagen."
        )
    if section_label == "press_coverage_bullets":
        user_prompt += (
            "\n- Nur Presseinhalte erlauben: eigene Pressemitteilungen oder externe Medienberichterstattung."
            "\n- Verboten: Adresse, Kontakt, Oeffnungszeiten, Bewertungszitate, allgemeine Firmenbeschreibung."
            "\n- Wenn keine klaren Presse-/Medieninhalte vorhanden sind: antworte exakt mit '-'."
        )

    parsed = _llm_json(
        provider=provider,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=schema,
        warnings=warnings,
        warning_key=f"llm_matrix_{section_label}",
    )
    text = _sanitize_summary_text(parsed.get("text"))
    return text or "-"


def _llm_trend_text(
    *,
    provider: str,
    company: str,
    trend_name: str,
    match_score: float,
    trend_keywords: List[str],
    matched_keywords: List[str],
    evidence_snippets: List[str],
    max_items: int,
    warnings: List[str],
) -> str:
    if not trend_name:
        return "-"

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "text": {"type": "string"},
        },
        "required": ["text"],
    }
    system_prompt = (
        "Du verdichtest Trend-Match-Informationen in kurze Stichpunkte. "
        "Nur Fakten aus dem Input."
    )
    user_prompt = (
        f"Unternehmen: {company}\n"
        f"Trend: {trend_name}\n"
        f"Match-Score: {round(max(0.0, min(1.0, match_score)), 4)}\n"
        f"Trend-Suchbegriffe: {trend_keywords}\n"
        f"Matched Keywords: {matched_keywords}\n"
        f"Evidence Snippets: {evidence_snippets[:3]}\n\n"
        "Regeln:\n"
        "- Gib genau einen kompakten String aus.\n"
        "- Mehrere Aspekte mit ';' trennen.\n"
        "- Keine Erfindungen.\n"
        "- Wenn nichts verwertbar: antworte exakt mit '-'.\n"
        "- Verboten sind Negativsaetze wie 'Keine ... gefunden' oder 'Keine Uebereinstimmungen gefunden'. "
        "Dann nur '-'."
    )
    parsed = _llm_json(
        provider=provider,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=schema,
        warnings=warnings,
        warning_key="llm_matrix_trend",
    )
    text = _sanitize_summary_text(parsed.get("text"))
    return text or "-"


def run_step_3_1_matrix(
    *,
    req: Step31MatrixRequest,
    user_root: Path,
    work_root: Path,
) -> Step31MatrixResult:
    warnings: List[str] = []
    provider = _clean_text(str(req.provider or "ionos")).lower() or "ionos"
    if provider not in {"ionos", "openai"}:
        warnings.append(f"unsupported_provider:{provider};fallback_to_ionos")
        provider = "ionos"

    payload = _load_payload(
        inline_obj=req.competitor_trends,
        path=req.competitor_trends_path,
        user_root=user_root,
        work_root=work_root,
    )
    rows = _extract_profiles(payload)
    if not rows:
        warnings.append("profiles_empty")
        return Step31MatrixResult(provider=provider, profiles=[], extraction_warnings=warnings)

    out_profiles: List[Step31CompanyMatrixProfile] = []
    for row in rows[: req.max_companies]:
        company = _clean_text(str(row.get("company") or ""))
        website = _normalize_url(str(row.get("website") or ""))
        region = _clean_text(str(row.get("region") or ""))
        local_warnings: List[str] = []

        cpta = row.get("company_profile_target_audience") if isinstance(row.get("company_profile_target_audience"), dict) else {}
        offers = row.get("offers_actions") if isinstance(row.get("offers_actions"), dict) else {}
        ratings = row.get("ratings_reach") if isinstance(row.get("ratings_reach"), dict) else {}
        press = row.get("press_coverage") if isinstance(row.get("press_coverage"), dict) else {}
        trend_rows = row.get("trend_matches") if isinstance(row.get("trend_matches"), list) else []
        trend_rows = [t for t in trend_rows if isinstance(t, dict)]

        profile_summary = _clean_text(str(cpta.get("summary") or ""))
        actions_summary = _clean_text(str(offers.get("summary") or ""))
        ratings_summary = _clean_text(str(ratings.get("summary") or ""))
        press_summary = _clean_text(str(press.get("summary") or ""))
        social_reach = _clean_text(str(ratings.get("social_reach") or ""))

        customer_segment_bullets = _llm_section_text(
            provider=provider,
            company=company,
            section_label="customer_segment_bullets",
            context_text=profile_summary,
            max_items=req.max_bullets_per_section,
            warnings=local_warnings,
        )
        actions_bullets = _llm_section_text(
            provider=provider,
            company=company,
            section_label="actions_bullets",
            context_text=actions_summary,
            max_items=req.max_bullets_per_section,
            warnings=local_warnings,
        )
        ratings_bullets = _llm_section_text(
            provider=provider,
            company=company,
            section_label="ratings_bullets",
            context_text=ratings_summary,
            max_items=req.max_bullets_per_section,
            warnings=local_warnings,
        )
        press_coverage_bullets = _llm_section_text(
            provider=provider,
            company=company,
            section_label="press_coverage_bullets",
            context_text=press_summary,
            max_items=req.max_bullets_per_section,
            warnings=local_warnings,
        )
        social_media_reach_bullets = _llm_section_text(
            provider=provider,
            company=company,
            section_label="social_media_reach_bullets",
            context_text=social_reach,
            max_items=req.max_bullets_per_section,
            warnings=local_warnings,
        )

        trend_items: List[Step31TrendMatrixItem] = []

        for base in trend_rows:
            trend_name = _clean_text(str(base.get("trend_summary") or ""))
            keywords = [str(k).strip() for k in (base.get("keywords") if isinstance(base.get("keywords"), list) else []) if _clean_text(str(k))]
            matched_keywords = [str(k).strip() for k in (base.get("matched_keywords") if isinstance(base.get("matched_keywords"), list) else []) if _clean_text(str(k))]
            evidence_snippets = [str(s).strip() for s in (base.get("evidence_snippets") if isinstance(base.get("evidence_snippets"), list) else []) if _clean_text(str(s))]
            try:
                match_score = float(base.get("match_score") or 0.0)
            except Exception:
                match_score = 0.0

            trend_summary_text = _llm_trend_text(
                provider=provider,
                company=company,
                trend_name=trend_name,
                match_score=match_score,
                trend_keywords=keywords,
                matched_keywords=matched_keywords,
                evidence_snippets=evidence_snippets,
                max_items=req.max_trend_bullets,
                warnings=local_warnings,
            )
            if not trend_summary_text:
                trend_summary_text = _fallback_trend_items([base], max_trend_bullets=req.max_trend_bullets)[0].summary

            trend_items.append(
                Step31TrendMatrixItem(
                    trend_name=trend_name,
                    match_score=round(max(0.0, min(1.0, match_score)), 4),
                    trend_keywords=keywords,
                    summary=trend_summary_text,
                )
            )

        try:
            google_rating = float(ratings.get("google_rating")) if ratings.get("google_rating") is not None else None
        except Exception:
            google_rating = None
        try:
            google_review_count = int(ratings.get("google_review_count")) if ratings.get("google_review_count") is not None else None
        except Exception:
            google_review_count = None

        source_urls = [u for u in (_normalize_url(str(x)) for x in row.get("source_urls", [])) if u]
        profile = Step31CompanyMatrixProfile(
            company=company,
            website=website,
            region=region,
            customer_segment_bullets=customer_segment_bullets,
            actions_bullets=actions_bullets,
            ratings_bullets=ratings_bullets,
            press_coverage_bullets=press_coverage_bullets,
            google_rating=google_rating,
            google_review_count=google_review_count,
            social_media_reach_bullets=social_media_reach_bullets,
            trend_items=trend_items,
            source_urls=source_urls,
            extraction_warnings=[_clean_text(w) for w in local_warnings if _clean_text(w)],
        )
        out_profiles.append(profile)
        warnings.extend([f"{company}:{w}" for w in local_warnings if _clean_text(w)])

    return Step31MatrixResult(
        provider=provider,
        profiles=out_profiles,
        extraction_warnings=list(dict.fromkeys([_clean_text(w) for w in warnings if _clean_text(w)])),
    )
