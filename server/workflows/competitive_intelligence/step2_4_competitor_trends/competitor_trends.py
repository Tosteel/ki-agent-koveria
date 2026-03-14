from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

from fastapi import HTTPException

from server.services.llm_ionos import IonosLLM
from server.services.llm_openai import LlmOpenai
from server.tools.web.browser import web_crawl_site

from .models import (
    Step24CompetitorTrendProfile,
    Step24CompetitorTrendsRequest,
    Step24CompetitorTrendsResult,
    TrendMatchItem,
)


_WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß][A-Za-zÄÖÜäöüß0-9\-]{2,}")
_SPACE_RE = re.compile(r"\s+")
_ALLOWED_TEXT_RE = re.compile(r"[^0-9A-Za-zÄÖÜäöüß\s\.,;:!\?\(\)\-/]")
_STOPWORDS_DE = {
    "und", "oder", "aber", "auch", "eine", "einer", "einem", "einen", "eines", "der", "die", "das", "den", "dem", "des",
    "mit", "ohne", "fuer", "für", "von", "auf", "bei", "im", "in", "am", "an", "aus", "zu", "zum", "zur", "als", "ist",
    "sind", "war", "wird", "werden", "dass", "dies", "diese", "dieser", "dieses", "kuechen", "küchen", "trend", "trends",
    "2026",
}
_GENERIC_TREND_KEYWORDS = {
    "trend",
    "trends",
    "küchentrend",
    "kuechentrend",
    "küchentrends",
    "kuechentrends",
    "kuechen trend",
    "kuechen trends",
    "küchen trend",
    "küchen trends",
}
_BLOCKED_URL_HOSTS = {
    "imgs.search.brave.com",
    "search.brave.com",
    "favicon.search.brave.com",
    "brave.com",
}


def _clean_text(value: str) -> str:
    return _SPACE_RE.sub(" ", str(value or "")).strip()


def _sanitize_text(value: str, *, max_chars: int = 2000) -> str:
    text = _clean_text(str(value or ""))
    text = _ALLOWED_TEXT_RE.sub(" ", text)
    text = _clean_text(text)
    if max_chars > 0:
        text = text[:max_chars]
    return text


def _normalize_url(value: str) -> str:
    raw = _clean_text(str(value or "")).rstrip(".,;:")
    if not raw:
        return ""
    if any(ch in raw for ch in ("\n", "\r", "\t", " ", "\\", "\u2026")):
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


def _is_generic_keyword(value: str) -> bool:
    token = _clean_text(str(value or "")).lower()
    if not token:
        return True
    if token in _GENERIC_TREND_KEYWORDS:
        return True
    if token in {"2026", "2027", "2028"}:
        return True
    compact = token.replace("-", " ")
    if compact in _GENERIC_TREND_KEYWORDS:
        return True
    if compact.startswith("küchentrend") or compact.startswith("kuechentrend"):
        return True
    if compact.startswith("trend ") or compact == "trend":
        return True
    return False


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
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid payload: expected object.")
    return payload


def _extract_profiles(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(payload.get("competitor_profile_structured"), dict):
        payload = payload["competitor_profile_structured"]
    rows = payload.get("profiles") if isinstance(payload.get("profiles"), list) else []
    return [r for r in rows if isinstance(r, dict)]


def _extract_trends(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(payload.get("market_trends_summary"), dict):
        payload = payload["market_trends_summary"]
    rows = payload.get("summaries") if isinstance(payload.get("summaries"), list) else []
    return [r for r in rows if isinstance(r, dict)]


def _extract_keywords(summary: str, *, min_len: int, max_keywords: int) -> List[str]:
    words = [w.lower() for w in _WORD_RE.findall(summary or "")]
    out: List[str] = []
    seen: set[str] = set()
    for w in words:
        if len(w) < min_len or w in _STOPWORDS_DE or _is_generic_keyword(w):
            continue
        if w in seen:
            continue
        seen.add(w)
        out.append(w)
        if len(out) >= max_keywords:
            break
    return out


def _extract_openai_output_text(resp: Dict[str, Any]) -> str:
    out = ""
    for item in resp.get("output", []):
        for c in item.get("content", []):
            if c.get("type") == "output_text":
                out += str(c.get("text") or "")
    return out


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
        parsed = json.loads(t)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    start, end = t.find("{"), t.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(t[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
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
                    "name": "step2_4_schema",
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
                    "name": "step2_4_schema",
                    "schema": schema,
                    "strict": False,
                },
            },
        )
        return _parse_json_strictish(client_i.extract_text(comp))
    except Exception as exc:
        warnings.append(f"{warning_key}:llm_failed:{exc}")
        return {}


def _llm_keywords_for_trend(
    *,
    provider: str,
    trend_summary: str,
    max_keywords: int,
    min_keyword_len: int,
    warnings: List[str],
) -> List[str]:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "keywords": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["keywords"],
    }
    system_prompt = (
        "Extrahiere praezise Such-Keywords fuer Website-Matching aus einem Trend-Satz. "
        "Nur konkrete Begriffe, keine allgemeinen Fuellwoerter. "
        "VERBOTEN sind generische Begriffe wie: trend, trends, kuechentrends, kuechentrend, 2026."
    )
    user_prompt = (
        f"Trend: {trend_summary}\n"
        f"Gib maximal {max_keywords} Keywords als JSON aus. "
        f"Jedes Keyword mindestens {min_keyword_len} Zeichen.\n"
        "Regeln:\n"
        "- Keine generischen Begriffe (z.B. kuechentrends, trends, trend, 2026).\n"
        "- Bevorzuge konkrete Merkmale, Materialien, Farben, Technologien, Stilrichtungen.\n"
        "- Keywords sollen suchbar auf Unternehmensseiten sein."
    )
    parsed = _llm_json(
        provider=provider,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=schema,
        warnings=warnings,
        warning_key="llm_keywords",
    )
    keywords = parsed.get("keywords") if isinstance(parsed.get("keywords"), list) else []
    cleaned: List[str] = []
    seen: set[str] = set()
    for kw in keywords:
        token = _clean_text(str(kw or "")).lower()
        if len(token) < min_keyword_len or token in _STOPWORDS_DE or _is_generic_keyword(token):
            continue
        if token in seen:
            continue
        seen.add(token)
        cleaned.append(token)
        if len(cleaned) >= max_keywords:
            break
    return cleaned


def _llm_evaluate_snippet(
    *,
    provider: str,
    company: str,
    trend_summary: str,
    keywords: List[str],
    snippet: str,
    matched_keywords: List[str],
    warnings: List[str],
) -> Dict[str, Any]:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "matched": {"type": "boolean"},
            "match_score": {"type": "number"},
            "matched_keywords": {"type": "array", "items": {"type": "string"}},
            "reasoning": {"type": "string"},
        },
        "required": ["matched", "match_score", "matched_keywords", "reasoning"],
    }
    system_prompt = (
        "Bewerte, ob ein Website-Snippet einen Trend inhaltlich abdeckt. "
        "Nutze nur den gegebenen Snippet-Text. "
        "match_score zwischen 0.0 und 1.0."
    )
    user_prompt = (
        f"Unternehmen: {company}\n"
        f"Trend: {trend_summary}\n"
        f"Keywords: {keywords}\n"
        f"Vorab gefundene Keywords im Snippet: {matched_keywords}\n"
        f"Snippet: {snippet}\n"
        "Antworte als JSON."
    )
    return _llm_json(
        provider=provider,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=schema,
        warnings=warnings,
        warning_key="llm_snippet_eval",
    )


def _first_snippet(text: str, keyword: str, *, window: int = 180) -> str:
    lower = text.lower()
    idx = lower.find(keyword.lower())
    if idx < 0:
        return ""
    start = max(0, idx - window // 2)
    end = min(len(text), idx + window // 2)
    snippet = text[start:end]
    return _sanitize_text(snippet, max_chars=260)


def _web_crawl_site_for_trend(
    *,
    url: str,
    keywords: List[str],
    trend_summary: str,
    timeout_ms: int,
    max_chars: int,
    warnings: List[str],
) -> Dict[str, Any]:
    if not url:
        return {"text": "", "snippet": "", "visited_urls": []}
    query_keywords = [k for k in keywords if _clean_text(k) and not _is_generic_keyword(k)]
    if not query_keywords:
        query_keywords = _extract_keywords(trend_summary, min_len=4, max_keywords=4)
    query = " ".join(query_keywords[:4]).strip() or _sanitize_text(trend_summary, max_chars=120)
    try:
        result = web_crawl_site(
            url=url,
            query=query,
            selector="body",
            max_matches=6,
            context_chars=220,
            timeout_ms=timeout_ms,
            max_pages=5,
            follow_links_matching=query,
        )
        matches = result.get("matches") if isinstance(result.get("matches"), list) else []
        snippets: List[str] = []
        texts: List[str] = []
        for m in matches:
            if not isinstance(m, dict):
                continue
            s = _sanitize_text(str(m.get("snippet") or ""), max_chars=380)
            t = _sanitize_text(str(m.get("text") or ""), max_chars=1200)
            if s:
                snippets.append(s)
            if t:
                texts.append(t)
        combined_text = _sanitize_text(" ".join(texts), max_chars=max_chars)
        best_snippet = snippets[0] if snippets else ""
        visited_raw = result.get("visited_urls") if isinstance(result.get("visited_urls"), list) else []
        visited_urls = [u for u in (_normalize_url(str(x)) for x in visited_raw) if u]
        return {
            "count": int(result.get("count") or 0),
            "text": combined_text,
            "snippet": best_snippet,
            "snippets": snippets,
            "visited_urls": visited_urls,
        }
    except Exception as exc:
        warnings.append(f"web_crawl_site_failed:{url}:{exc}")
        return {"count": 0, "text": "", "snippet": "", "snippets": [], "visited_urls": []}


def run_step_2_4_competitor_trends(
    *,
    req: Step24CompetitorTrendsRequest,
    user_root: Path,
    work_root: Path,
) -> Step24CompetitorTrendsResult:
    warnings: List[str] = []
    provider = _clean_text(str(req.provider or "ionos")).lower() or "ionos"
    if provider not in {"ionos", "openai"}:
        warnings.append(f"unsupported_provider:{provider};fallback_to_ionos")
        provider = "ionos"

    profiles_payload = _load_payload(
        inline_obj=req.competitor_profile_structured,
        path=req.competitor_profile_structured_path,
        user_root=user_root,
        work_root=work_root,
    )
    trends_payload = _load_payload(
        inline_obj=req.market_trends_summary,
        path=req.market_trends_summary_path,
        user_root=user_root,
        work_root=work_root,
    )

    profile_rows = _extract_profiles(profiles_payload)
    trend_rows = _extract_trends(trends_payload)[: req.max_trends]
    if not profile_rows:
        warnings.append("profiles_empty")
    if not trend_rows:
        warnings.append("trends_empty")
    if not profile_rows or not trend_rows:
        return Step24CompetitorTrendsResult(profiles=[], extraction_warnings=warnings)

    trend_specs: List[Dict[str, Any]] = []
    for row in trend_rows:
        summary = _sanitize_text(str(row.get("summary") or ""), max_chars=320)
        if not summary:
            continue
        source_urls = [u for u in (_normalize_url(str(x)) for x in row.get("source_urls", [])) if u]
        keywords = _llm_keywords_for_trend(
            provider=provider,
            trend_summary=summary,
            max_keywords=req.keywords_per_trend,
            min_keyword_len=req.min_keyword_len,
            warnings=warnings,
        )
        if not keywords:
            keywords = _extract_keywords(summary, min_len=req.min_keyword_len, max_keywords=req.keywords_per_trend)
            if keywords:
                warnings.append("llm_keywords_empty_fallback_heuristic")
        if not keywords:
            continue
        trend_specs.append(
            {
                "summary": summary,
                "source_urls": source_urls,
                "keywords": keywords,
            }
        )

    out_profiles: List[Step24CompetitorTrendProfile] = []
    for row in profile_rows[: req.max_companies]:
        company = _sanitize_text(str(row.get("company") or ""), max_chars=120)
        website = _normalize_url(str(row.get("website") or ""))
        region = _sanitize_text(str(row.get("region") or ""), max_chars=120)
        local_warnings: List[str] = []

        trend_matches: List[TrendMatchItem] = []
        for t in trend_specs:
            keywords = list(t["keywords"])
            browse = _web_crawl_site_for_trend(
                url=website,
                keywords=keywords,
                trend_summary=t["summary"],
                timeout_ms=req.website_timeout_ms,
                max_chars=req.website_max_chars,
                warnings=local_warnings,
            )
            browse_count = int(browse.get("count") or 0)
            site_text = _clean_text(str(browse.get("text") or ""))
            matched_keywords = [kw for kw in keywords if kw in site_text.lower()] if site_text else []
            hit_count = len(matched_keywords)
            raw_snippets = browse.get("snippets") if isinstance(browse.get("snippets"), list) else []
            cleaned_snippets: List[str] = []
            for s in raw_snippets:
                val = _sanitize_text(str(s or ""), max_chars=320)
                if val and val not in cleaned_snippets:
                    cleaned_snippets.append(val)

            snippet_hits = []
            for s in cleaned_snippets:
                lower_s = s.lower()
                if any(kw in lower_s for kw in keywords):
                    snippet_hits.append(s)

            snippet = _clean_text(str(browse.get("snippet") or ""))
            if not snippet and matched_keywords and site_text:
                snippet = _first_snippet(site_text, matched_keywords[0])
            if snippet and snippet not in snippet_hits:
                if any(kw in snippet.lower() for kw in keywords):
                    snippet_hits.insert(0, snippet)

            if snippet_hits:
                # Ausgabe aller Treffer-Snippets in einem Feld.
                snippet = " || ".join(snippet_hits[:8])
            evidence_snippets = snippet_hits[:8]
            matched = hit_count >= req.min_keyword_hits
            score = (hit_count / max(1, len(keywords))) if keywords else 0.0
            if snippet:
                eval_obj = _llm_evaluate_snippet(
                    provider=provider,
                    company=company,
                    trend_summary=t["summary"],
                    keywords=keywords,
                    snippet=snippet,
                    matched_keywords=matched_keywords,
                    warnings=local_warnings,
                )
                if isinstance(eval_obj, dict) and eval_obj:
                    matched = bool(eval_obj.get("matched"))
                    try:
                        score = float(eval_obj.get("match_score"))
                    except Exception:
                        score = (hit_count / max(1, len(keywords))) if keywords else 0.0
                    score = max(0.0, min(1.0, score))
                    llm_keywords = eval_obj.get("matched_keywords")
                    if isinstance(llm_keywords, list):
                        filtered = []
                        seen: set[str] = set()
                        for kw in llm_keywords:
                            token = _clean_text(str(kw or "")).lower()
                            if token and token in keywords and token not in seen:
                                seen.add(token)
                                filtered.append(token)
                        if filtered:
                            matched_keywords = filtered
            if not matched_keywords:
                snippet = ""
                evidence_snippets = []
            if browse_count == 0:
                snippet = ""
                evidence_snippets = []
            trend_matches.append(
                TrendMatchItem(
                    trend_summary=t["summary"],
                    trend_source_urls=t["source_urls"],
                    keywords=keywords,
                    matched=matched,
                    match_score=round(max(0.0, min(1.0, score)), 4),
                    matched_keywords=matched_keywords,
                    evidence_snippets=evidence_snippets,
                    source_urls=(browse.get("visited_urls") or ([website] if website else [])),
                )
            )

        section_default = {"summary": "", "source_urls": []}
        profile = Step24CompetitorTrendProfile(
            company=company,
            website=website,
            region=region,
            company_profile_target_audience=row.get("company_profile_target_audience") if isinstance(row.get("company_profile_target_audience"), dict) else section_default,
            offers_actions=row.get("offers_actions") if isinstance(row.get("offers_actions"), dict) else section_default,
            ratings_reach=row.get("ratings_reach") if isinstance(row.get("ratings_reach"), dict) else {},
            press_coverage=row.get("press_coverage") if isinstance(row.get("press_coverage"), dict) else section_default,
            trend_matches=trend_matches,
            source_urls=[u for u in (_normalize_url(str(x)) for x in row.get("source_urls", [])) if u],
            extraction_warnings=[_sanitize_text(w, max_chars=240) for w in local_warnings if _sanitize_text(w, max_chars=240)],
        )
        out_profiles.append(profile)
        warnings.extend([f"{company}:{w}" for w in local_warnings])

    return Step24CompetitorTrendsResult(
        provider=provider,
        profiles=out_profiles,
        extraction_warnings=list(dict.fromkeys([_clean_text(w) for w in warnings if _clean_text(w)])),
    )
