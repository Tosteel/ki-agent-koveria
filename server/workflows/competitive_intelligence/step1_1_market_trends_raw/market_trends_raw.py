from __future__ import annotations

from typing import List

from server.services.llm_brave import LlmBrave

from .models import MarketTrendsRawResult, Step11MarketTrendsRawRequest


def _clean_text(value: str) -> str:
    return " ".join(str(value or "").split()).strip()


def _build_plain_search_query(*, market_context: str, search_sources: List[str]) -> str:
    base = f"Trends zu: {_clean_text(market_context)}"
    cleaned_sources = [_clean_text(src) for src in (search_sources or []) if _clean_text(src)]
    if not cleaned_sources:
        return base
    return f"{base}, {', '.join(cleaned_sources)}"


def run_step_1_1_market_trends_raw(*, req: Step11MarketTrendsRawRequest) -> MarketTrendsRawResult:
    warnings: List[str] = []
    query = _build_plain_search_query(
        market_context=req.market_context,
        search_sources=req.search_sources,
    )

    llm = LlmBrave()
    if not llm.enabled():
        warnings.append("brave_not_configured")
        return MarketTrendsRawResult(
            provider=str(req.provider or "brave").strip().lower() or "brave",
            market_context=req.market_context,
            search_sources=req.search_sources,
            query=query,
            raw_text="",
            raw_response={},
            extraction_warnings=warnings,
        )

    raw_response = {}
    raw_text = ""
    try:
        # Important: send only a plain search query as user content (no system prompt, no extra instructions).
        raw_response = llm.chat_completions(
            messages=[{"role": "user", "content": query}],
            model="brave",
            stream=req.brave_stream,
            language=req.brave_language,
            country=req.brave_country,
            enable_entities=req.brave_enable_entities,
            enable_citations=req.brave_enable_citations,
            enable_research=req.brave_enable_research,
            timeout_s=req.timeout_s,
        )
        raw_text = llm.extract_text(raw_response) or ""
        if not raw_text:
            warnings.append("brave_empty_response")
    except Exception as exc:
        warnings.append(f"brave_error: {exc}")

    return MarketTrendsRawResult(
        provider=str(req.provider or "brave").strip().lower() or "brave",
        market_context=req.market_context,
        search_sources=req.search_sources,
        query=query,
        raw_text=raw_text,
        raw_response=raw_response if isinstance(raw_response, dict) else {},
        extraction_warnings=warnings,
    )
