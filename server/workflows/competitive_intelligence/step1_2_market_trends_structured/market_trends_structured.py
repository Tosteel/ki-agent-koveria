from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from fastapi import HTTPException

from server.services.llm_ionos import IonosLLM
from server.services.llm_openai import LlmOpenai
from server.tools.browser.browser import get_website

from .models import (
    Step12MarketTrendSource,
    Step12MarketTrendsStructuredRequest,
    Step12MarketTrendsStructuredResult,
)


_CITATION_RE = re.compile(r"<citation>\s*(\{.*?\})\s*</citation>", flags=re.DOTALL | re.IGNORECASE)
_USAGE_RE = re.compile(r"<usage>\s*\{.*?\}\s*</usage>", flags=re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


def _clean_text(value: str) -> str:
    return " ".join(str(value or "").split()).strip()


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
    inline_obj: dict | None,
    path: str | None,
    user_root: Path,
    work_root: Path,
) -> dict:
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


def _extract_raw_text(payload: dict) -> str:
    candidates = [
        payload.get("raw_text"),
        payload.get("content"),
        (payload.get("market_trends_raw") or {}).get("raw_text") if isinstance(payload.get("market_trends_raw"), dict) else "",
        (payload.get("market_trends_raw") or {}).get("content") if isinstance(payload.get("market_trends_raw"), dict) else "",
    ]
    for value in candidates:
        txt = _clean_text(str(value or ""))
        if txt:
            return str(value)
    return ""


def _extract_market_query(payload: dict) -> str:
    candidates = [
        payload.get("query"),
        (payload.get("market_trends_raw") or {}).get("query") if isinstance(payload.get("market_trends_raw"), dict) else "",
        payload.get("market_context"),
        (payload.get("market_trends_raw") or {}).get("market_context")
        if isinstance(payload.get("market_trends_raw"), dict)
        else "",
    ]
    for value in candidates:
        txt = _clean_text(str(value or ""))
        if txt:
            return txt
    return "Kuechentrends 2026"


def _strip_markup(text: str) -> str:
    out = _USAGE_RE.sub(" ", str(text or ""))
    out = _CITATION_RE.sub(" ", out)
    out = _TAG_RE.sub(" ", out)
    return _clean_text(out)


def _extract_sources_from_raw_text(raw_text: str, *, max_sources: int) -> Tuple[List[Tuple[str, str]], List[str]]:
    warnings: List[str] = []
    text = str(raw_text or "")
    if not text.strip():
        return [], ["raw_text_empty"]

    per_url_texts: Dict[str, List[str]] = {}
    url_order: List[str] = []

    paragraphs = [p for p in re.split(r"\n\s*\n", text) if _clean_text(p)]
    for paragraph in paragraphs:
        citation_blobs = _CITATION_RE.findall(paragraph)
        if not citation_blobs:
            continue
        clean_paragraph = _strip_markup(paragraph)
        if not clean_paragraph:
            continue
        for blob in citation_blobs:
            try:
                item = json.loads(blob)
            except Exception:
                warnings.append("citation_parse_failed")
                continue
            if not isinstance(item, dict):
                continue
            url = _clean_text(str(item.get("url") or ""))
            if not url:
                continue
            if url not in per_url_texts:
                per_url_texts[url] = []
                url_order.append(url)
            per_url_texts[url].append(clean_paragraph)

    out: List[Tuple[str, str]] = []
    for url in url_order:
        chunks = per_url_texts.get(url) or []
        merged: List[str] = []
        seen: set[str] = set()
        for chunk in chunks:
            key = _clean_text(chunk).lower()
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(_clean_text(chunk))
        if not merged:
            continue
        out.append((url, "\n\n".join(merged)))
        if len(out) >= max_sources:
            break

    if not out:
        warnings.append("no_citation_urls_extracted")
    return out, warnings


def _extract_openai_output_text(resp: dict) -> str:
    out = ""
    for item in resp.get("output", []):
        for c in item.get("content", []):
            if c.get("type") == "output_text":
                out += str(c.get("text") or "")
    return out.strip()


def _parse_bullets(text: str, *, max_items: int) -> List[str]:
    rows = [r.strip() for r in str(text or "").splitlines() if r.strip()]
    out: List[str] = []
    for row in rows:
        cleaned = re.sub(r"^[\-\*\u2022\d\.\)\s]+", "", row).strip()
        if not cleaned:
            continue
        out.append(cleaned)
        if len(out) >= max_items:
            break
    return out


def _fallback_bullets(text: str, *, max_items: int) -> List[str]:
    snippets = [s.strip() for s in re.split(r"(?<=[\.\!\?])\s+", _clean_text(text)) if s.strip()]
    out: List[str] = []
    for snippet in snippets:
        out.append(snippet[:220].rstrip())
        if len(out) >= max_items:
            break
    return out if out else [_clean_text(text)[:220]]


def _summarize_with_llm(*, provider: str, url: str, source_text: str, max_items: int, warnings: List[str]) -> List[str]:
    user_prompt = (
        f"Fasse den folgenden Originaltext aus einer Trendquelle in genau {max_items} Stichpunkten auf Deutsch zusammen.\n"
        "Regeln: Nur Stichpunkte, keine Einleitung, keine Halluzinationen, nur Aussagen aus dem Text.\n"
        f"Quelle: {url}\n"
        f"Originaltext:\n{source_text}"
    )
    system_prompt = "Du extrahierst praezise Kernaussagen aus Quellen und antwortest nur mit Stichpunkten."
    p = str(provider or "ionos").strip().lower()
    if p not in {"ionos", "openai"}:
        p = "ionos"

    try:
        if p == "openai":
            client = LlmOpenai()
            if not client.enabled():
                warnings.append("openai_not_configured")
                return _fallback_bullets(source_text, max_items=max_items)
            resp = client._call(
                input_messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            parsed = _parse_bullets(_extract_openai_output_text(resp), max_items=max_items)
            return parsed or _fallback_bullets(source_text, max_items=max_items)

        client_i = IonosLLM()
        if not client_i.enabled():
            warnings.append("ionos_not_configured")
            return _fallback_bullets(source_text, max_items=max_items)
        comp = client_i.chat_completions(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        parsed = _parse_bullets(client_i.extract_text(comp), max_items=max_items)
        return parsed or _fallback_bullets(source_text, max_items=max_items)
    except Exception as exc:
        warnings.append(f"summary_llm_failed({p}): {exc}")
        return _fallback_bullets(source_text, max_items=max_items)


def run_step_1_2_market_trends_structured(
    *,
    req: Step12MarketTrendsStructuredRequest,
    user_root: Path,
    work_root: Path,
) -> Step12MarketTrendsStructuredResult:
    warnings: List[str] = []
    payload = _load_payload(
        inline_obj=req.market_trends_raw,
        path=req.market_trends_raw_path,
        user_root=user_root,
        work_root=work_root,
    )
    raw_text = _extract_raw_text(payload)
    if not _clean_text(raw_text):
        warnings.append("raw_text_missing")
        return Step12MarketTrendsStructuredResult(
            provider=str(req.provider or "ionos").strip().lower() or "ionos",
            sources=[],
            extraction_warnings=warnings,
        )

    extracted, parse_warnings = _extract_sources_from_raw_text(raw_text, max_sources=req.max_sources)
    warnings.extend(parse_warnings)

    provider = str(req.provider or "ionos").strip().lower()
    if provider not in {"ionos", "openai"}:
        warnings.append(f"unsupported_provider:{provider};fallback_to_ionos")
        provider = "ionos"

    sources: List[Step12MarketTrendSource] = []
    for _idx, (url, source_text) in enumerate(extracted, start=1):
        original_text_raw = _clean_text(source_text)[: req.max_chars_per_source]
        original_text = original_text_raw
        if req.use_view_website:
            try:
                vw = get_website(
                    url=url,
                    selector="body",
                    timeout_ms=req.view_timeout_ms,
                    max_chars=req.max_chars_per_source,
                    include_image_urls=True,
                )
                vw_text = _clean_text(str((vw or {}).get("text") or ""))
                if vw_text:
                    original_text = vw_text[: req.max_chars_per_source]
                else:
                    warnings.append(f"get_website_empty_text:{url}")
            except Exception as exc:
                warnings.append(f"get_website_failed:{url}:{exc}")

        bullets = _summarize_with_llm(
            provider=provider,
            url=url,
            source_text=original_text,
            max_items=req.summary_bullets,
            warnings=warnings,
        )
        sources.append(
            Step12MarketTrendSource(
                url=url,
                originaltext=original_text,
                originaltext_raw=original_text_raw,
                kernaussage=bullets,
            )
        )

    return Step12MarketTrendsStructuredResult(
        provider=provider,
        sources=sources,
        extraction_warnings=list(dict.fromkeys([w for w in warnings if _clean_text(w)])),
    )
