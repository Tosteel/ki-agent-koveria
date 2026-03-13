from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

from fastapi import HTTPException

from .models import (
    Step23CompetitorProfile,
    Step23CompetitorProfileStructuredRequest,
    Step23CompetitorProfileStructuredResult,
)


_CITATION_RE = re.compile(r"<citation>\s*(\{.*?\})\s*</citation>", flags=re.DOTALL | re.IGNORECASE)
_URL_RE = re.compile(r"https?://[^\s\]\)>,\"]+", flags=re.IGNORECASE)
_USAGE_RE = re.compile(r"<usage>\s*(\{.*?\})\s*</usage>", flags=re.DOTALL | re.IGNORECASE)
_ALLOWED_SUMMARY_CHARS_RE = re.compile(r"[^0-9A-Za-zÄÖÜäöüß\s\.,;:!\?\(\)\-/]")
_RATING_RE = re.compile(r"\b([1-5](?:[.,]\d)?)\b")
_REVIEW_COUNT_RE = re.compile(r"\b(\d{1,5})\s*(bewertungen|rezensionen)\b", flags=re.IGNORECASE)
_BLOCKED_URL_HOSTS = {
    "imgs.search.brave.com",
    "search.brave.com",
    "favicon.search.brave.com",
    "brave.com",
}


def _clean_text(value: str) -> str:
    return " ".join(str(value or "").split()).strip()


def _normalize_real_url(value: str) -> str:
    raw = _clean_text(str(value or "")).rstrip(".,;:")
    if not raw:
        return ""
    if any(ch in raw for ch in ("\n", "\r", "\t", " ")):
        return ""
    try:
        parsed = urlparse(raw)
    except Exception:
        return ""
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if host in _BLOCKED_URL_HOSTS or host.endswith(".search.brave.com"):
        return ""
    return raw


def _clean_text_for_llm(value: str, *, max_chars: int = 5000) -> str:
    text = str(value or "")
    # Remove verbose tool artifacts from Brave responses before sending context to LLM.
    text = _CITATION_RE.sub(" ", text)
    text = _USAGE_RE.sub(" ", text)
    text = text.replace("\\u2026", " ").replace("\\n", " ")
    text = _clean_text(text)
    if max_chars > 0:
        text = text[:max_chars]
    return text


def _sanitize_summary_text(value: str, *, max_chars: int = 0) -> str:
    text = _clean_text_for_llm(value, max_chars=0)
    text = _URL_RE.sub(" ", text)
    text = _ALLOWED_SUMMARY_CHARS_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if max_chars > 0:
        text = text[:max_chars]
    return text


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
    if isinstance(payload.get("competitor_profile_raw"), dict):
        payload = payload["competitor_profile_raw"]
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid payload: expected object.")
    return payload


def _extract_urls_from_text(raw_text: str) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()

    for blob in _CITATION_RE.findall(str(raw_text or "")):
        try:
            obj = json.loads(blob)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        url = _normalize_real_url(str(obj.get("url") or ""))
        if url and url not in seen:
            seen.add(url)
            out.append(url)

    for match in _URL_RE.findall(str(raw_text or "")):
        url = _normalize_real_url(str(match))
        if url and url not in seen:
            seen.add(url)
            out.append(url)

    return out


def _topic_context(company_raw: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    ctx: Dict[str, Dict[str, Any]] = {}
    rows = company_raw.get("raw_searches") if isinstance(company_raw.get("raw_searches"), list) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        topic = _clean_text(str(row.get("topic") or ""))
        query = _clean_text(str(row.get("query") or ""))
        raw_text = _clean_text(str(row.get("raw_text") or ""))
        raw_text_llm = _clean_text_for_llm(raw_text)
        urls = _extract_urls_from_text(raw_text)
        if not topic:
            continue
        ctx[topic] = {
            "query": query,
            "raw_text": raw_text,
            "raw_text_llm": raw_text_llm,
            "source_urls": urls,
        }
    return ctx


def _section_source_hints(topic_ctx: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    by_topic = {
        "company_profile_target_audience": topic_ctx.get("unternehmensprofil_zielgruppe", {}),
        "offers_actions": topic_ctx.get("angebote_aktionen", {}),
        "ratings_reach": topic_ctx.get("bewertungen_reichweite_social", {}),
        "press_coverage": topic_ctx.get("presse_berichterstattung", {}),
    }
    out: Dict[str, List[str]] = {}
    all_urls: List[str] = []
    for section, info in by_topic.items():
        urls = info.get("source_urls") if isinstance(info, dict) and isinstance(info.get("source_urls"), list) else []
        clean_urls = []
        seen: set[str] = set()
        for u in urls:
            url = _normalize_real_url(str(u or ""))
            if url and url not in seen:
                seen.add(url)
                clean_urls.append(url)
                if url not in all_urls:
                    all_urls.append(url)
        out[section] = clean_urls
    out["all"] = all_urls
    return out


def _as_list_str(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_text(str(value or ""))
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _filter_allowed_urls(urls: List[str], allowed: List[str]) -> List[str]:
    allowed_set = {
        normalized
        for normalized in (_normalize_real_url(value) for value in allowed)
        if normalized
    }
    out: List[str] = []
    for url in _as_list_str(urls):
        normalized = _normalize_real_url(url)
        if normalized and normalized in allowed_set and normalized not in out:
            out.append(normalized)
    return out


def _topic_summary(topic_ctx: Dict[str, Dict[str, Any]], topic: str) -> str:
    raw = str((topic_ctx.get(topic) or {}).get("raw_text_llm") or "")
    return _sanitize_summary_text(raw, max_chars=0)


def _extract_google_metrics(topic_ctx: Dict[str, Dict[str, Any]]) -> tuple[float | None, int | None]:
    text = _clean_text_for_llm(str((topic_ctx.get("bewertungen_reichweite_social") or {}).get("raw_text") or ""))

    rating: float | None = None
    for match in _RATING_RE.findall(text):
        try:
            value = float(match.replace(",", "."))
        except Exception:
            continue
        if 0.0 <= value <= 5.0:
            rating = value
            break

    review_count: int | None = None
    counts = [int(m[0]) for m in _REVIEW_COUNT_RE.findall(text)]
    if counts:
        review_count = max(counts)

    return rating, review_count


def _extract_social_reach(topic_ctx: Dict[str, Dict[str, Any]]) -> str:
    text = _clean_text_for_llm(str((topic_ctx.get("bewertungen_reichweite_social") or {}).get("raw_text") or ""))
    if not text:
        return ""
    lowered = text.lower()
    if "keine konkreten daten" in lowered or "keine konkreten informationen" in lowered:
        return "Keine belastbaren Social-Media-Reichweitenzahlen im Rohmaterial gefunden."
    if "follower" in lowered or "likes" in lowered or "instagram" in lowered or "facebook" in lowered:
        return _sanitize_summary_text(text, max_chars=260)
    return ""


def run_step_2_3_competitor_profile_structured(
    *,
    req: Step23CompetitorProfileStructuredRequest,
    user_root: Path,
    work_root: Path,
) -> Step23CompetitorProfileStructuredResult:
    warnings: List[str] = []
    payload = _load_payload(
        inline_obj=req.competitor_profile_raw,
        path=req.competitor_profile_raw_path,
        user_root=user_root,
        work_root=work_root,
    )

    companies_raw = payload.get("companies") if isinstance(payload.get("companies"), list) else []
    if not companies_raw:
        warnings.append("companies_empty")
        return Step23CompetitorProfileStructuredResult(
            provider=str(req.provider or "ionos").strip().lower() or "ionos",
            profiles=[],
            extraction_warnings=warnings,
        )

    provider = "heuristic"

    out: List[Step23CompetitorProfile] = []
    for company_raw in companies_raw[: req.max_companies]:
        if not isinstance(company_raw, dict):
            continue
        company = _clean_text(str(company_raw.get("company") or ""))
        website = _clean_text(str(company_raw.get("website") or ""))
        region = _clean_text(str(company_raw.get("region") or ""))
        if not company:
            continue

        topic_ctx = _topic_context(company_raw)
        source_hints = _section_source_hints(topic_ctx)
        allowed = source_hints.get("all", [])
        cp_summary = _topic_summary(topic_ctx, "unternehmensprofil_zielgruppe")
        oa_summary = _topic_summary(topic_ctx, "angebote_aktionen")
        rr_summary = _topic_summary(topic_ctx, "bewertungen_reichweite_social")
        pc_summary = _topic_summary(topic_ctx, "presse_berichterstattung")
        google_rating, google_review_count = _extract_google_metrics(topic_ctx)
        social_reach = _extract_social_reach(topic_ctx)
        local_warnings: List[str] = []
        if not cp_summary:
            local_warnings.append("missing_company_profile_target_audience")
        if not oa_summary:
            local_warnings.append("missing_offers_actions")
        if not rr_summary:
            local_warnings.append("missing_ratings_reach")
        if not pc_summary:
            local_warnings.append("missing_press_coverage")

        result = Step23CompetitorProfile(
            company=company,
            website=website,
            region=region,
            company_profile_target_audience={
                "summary": cp_summary,
                "source_urls": _filter_allowed_urls(source_hints.get("company_profile_target_audience", []), allowed),
            },
            offers_actions={
                "summary": oa_summary,
                "source_urls": _filter_allowed_urls(source_hints.get("offers_actions", []), allowed),
            },
            ratings_reach={
                "summary": rr_summary,
                "google_rating": _to_float(google_rating),
                "google_review_count": _to_int(google_review_count),
                "social_reach": social_reach,
                "source_urls": _filter_allowed_urls(source_hints.get("ratings_reach", []), allowed),
            },
            press_coverage={
                "summary": pc_summary,
                "source_urls": _filter_allowed_urls(source_hints.get("press_coverage", []), allowed),
            },
            source_urls=_filter_allowed_urls(allowed, allowed),
            extraction_warnings=_as_list_str(local_warnings),
        )
        out.append(result)
        warnings.extend([f"{company}:{w}" for w in local_warnings if _clean_text(w)])

    return Step23CompetitorProfileStructuredResult(
        provider=provider,
        profiles=out,
        extraction_warnings=list(dict.fromkeys([w for w in warnings if _clean_text(w)])),
    )
