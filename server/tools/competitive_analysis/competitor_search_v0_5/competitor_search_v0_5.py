from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from server.services.llm_ionos import IonosLLM
from server.services.llm_openai import LlmOpenai
from server.services.llm_perplexity import LlmPerplexity
from server.tools.competitive_analysis.backup.competitor_identification import (
    _clean_text,
    _clean_url,
    _cosine_similarity,
    _iter_queries,
    _langsearch_fallback,
    _load_json_obj,
    _openai_search,
    _perplexity_search,
)

from .models import CompetitorSearchResultsV05, ProductCompetitorSlim


_RETAILER_HINTS = (
    "amazon.",
    "ebay.",
    "otto.",
    "mediamarkt.",
    "saturn.",
    "galaxus.",
    "alltron.",
    "kaufland.",
    "walmart.",
    "bestbuy.",
    "aliexpress.",
)
_MARKETPLACE_HINTS = ("marketplace", "vergleich", "preisvergleich", "deals")
_TESTING_HINTS = (
    "test",
    "tests",
    "review",
    "vergleich",
    "bestenliste",
    "computerbild",
    "chip.",
    "homeandsmart",
    "businessinsider",
    "blog",
    "magazin",
)


def _domain(url: str) -> str:
    s = str(url or "").lower()
    s = re.sub(r"^https?://", "", s)
    s = s.split("/", 1)[0]
    return s.replace("www.", "")


def _search_results(
    *,
    provider: str,
    query: str,
    per_query_results: int,
    openai_key: str,
    openai_model: str,
    perplexity_key: str,
    perplexity_model: str,
) -> Tuple[List[Dict[str, str]], str]:
    p = str(provider or "openai").strip().lower()
    if p == "openai" and openai_key:
        try:
            return _openai_search(query, per_query_results, api_key=openai_key, model=openai_model), "web_search_openai"
        except Exception:
            pass
    if p == "perplexity" and perplexity_key:
        try:
            return _perplexity_search(query, per_query_results, api_key=perplexity_key, model=perplexity_model), "web_search_perplexity"
        except Exception:
            pass
    try:
        return _langsearch_fallback(query, per_query_results), "web_search_fallback"
    except Exception:
        return [], "web_search_fallback"


def _url_type(url: str) -> str:
    u = _clean_url(url).lower()
    d = _domain(u)
    if not d:
        return "unknown"
    if any(h in d for h in _RETAILER_HINTS):
        return "retailer"
    if any(h in u for h in _MARKETPLACE_HINTS):
        return "marketplace"
    if any(h in u for h in _TESTING_HINTS) or any(h in d for h in _TESTING_HINTS):
        return "testing"
    return "official"


def _company_from_domain(url: str) -> str:
    d = _domain(url)
    if not d:
        return ""
    base = d.split(".")[0]
    if base in {"de", "en", "eu", "us", "at", "ch"}:
        parts = d.split(".")
        if len(parts) >= 3:
            base = parts[1]
    return _clean_text(base.title())


def _manufacturer_from_title_or_domain(title: str, url: str) -> str:
    t = _clean_text(title)
    if t:
        first = re.split(r"\s+", t, maxsplit=1)[0]
        first = _clean_text(first)
        if first and len(first) >= 2:
            return first
    return _company_from_domain(url)


def _parse_json_strictish(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    t = str(text).strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
        if "\n" in t:
            first, rest = t.split("\n", 1)
            if first.strip().lower() in {"json", "javascript"}:
                t = rest.strip()
    try:
        obj = __import__("json").loads(t)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    s = t.find("{")
    e = t.rfind("}")
    if s >= 0 and e > s:
        try:
            obj = __import__("json").loads(t[s : e + 1])
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _extract_manufacturer_llm(
    *,
    provider: str,
    title: str,
    url: str,
    snippet: str,
    warnings: List[str],
) -> str:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"manufacturer": {"type": "string"}},
        "required": ["manufacturer"],
    }
    system = (
        "Extrahiere den Hersteller/Brand aus einem Web-Suchergebnis. "
        "Gib nur JSON gemaess Schema zurueck. "
        "Nicht das erste Titelwort raten, sondern Marke/Hersteller normalisiert extrahieren. "
        "Falls unklar: leeren String."
    )
    user = f"title={title}\nurl={url}\nsnippet={snippet}"

    p = str(provider or "openai").strip().lower()
    order = [p] + [x for x in ("openai", "perplexity", "ionos") if x != p]
    for engine in order:
        try:
            if engine == "openai":
                c = LlmOpenai()
                if not c.enabled():
                    continue
                resp = c._call(
                    input_messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    text_format={"type": "json_schema", "name": "manufacturer_extract", "schema": schema, "strict": False},
                )
                text = ""
                for item in resp.get("output", []):
                    for cc in item.get("content", []):
                        if cc.get("type") == "output_text":
                            text += str(cc.get("text") or "")
                parsed = _parse_json_strictish(text)
                m = _clean_text(str(parsed.get("manufacturer") or ""))
                if m:
                    return m
            elif engine == "perplexity":
                c = LlmPerplexity()
                if not c.enabled():
                    continue
                resp = c._call(
                    input_messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    text_format={"type": "json_schema", "name": "manufacturer_extract", "schema": schema, "strict": False},
                )
                text = ""
                for item in resp.get("output", []):
                    for cc in item.get("content", []):
                        if cc.get("type") == "output_text":
                            text += str(cc.get("text") or "")
                parsed = _parse_json_strictish(text)
                m = _clean_text(str(parsed.get("manufacturer") or ""))
                if m:
                    return m
            else:
                c = IonosLLM()
                if not c.enabled():
                    continue
                comp = c.chat_completions(
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "manufacturer_extract", "schema": schema, "strict": True},
                    },
                )
                parsed = _parse_json_strictish(c.extract_text(comp))
                m = _clean_text(str(parsed.get("manufacturer") or ""))
                if m:
                    return m
        except Exception as exc:
            warnings.append(f"manufacturer llm extraction failed ({engine}): {exc}")
            continue
    return ""


def _reference_text(plan: Dict[str, Any]) -> str:
    chunks: List[str] = []
    for k in ("product_category",):
        v = plan.get(k)
        if isinstance(v, str):
            chunks.append(_clean_text(v))
    for k in ("search_terms", "additional_search_terms"):
        vals = plan.get(k)
        if isinstance(vals, list):
            for x in vals:
                if isinstance(x, dict):
                    t = _clean_text(str(x.get("term") or ""))
                else:
                    t = _clean_text(str(x or ""))
                if t:
                    chunks.append(t)
    return "\n".join(chunks)


def search_competitors_v0_5(
    *,
    analysis_plan: Optional[Dict[str, Any]],
    analysis_plan_path: Optional[str],
    provider: str = "openai",
    max_queries: int = 20,
    per_query_results: int = 10,
    max_candidates_to_check: int = 200,
    verbose_terminal: bool = False,
    verbose_search_hits: bool = False,
    user_root=None,
    work_root=None,
) -> CompetitorSearchResultsV05:
    def _log(msg: str) -> None:
        if verbose_terminal:
            print(f"[competitor_search_v0_5] {msg}")

    plan = _load_json_obj(
        inline_obj=analysis_plan,
        path=analysis_plan_path,
        root_key="analysis_plan",
        user_root=user_root,
        work_root=work_root,
    )

    warnings: List[str] = [str(w).strip() for w in (plan.get("extraction_warnings") or []) if str(w).strip()]

    p = str(provider or "openai").strip().lower()
    if p not in {"openai", "perplexity", "ionos"}:
        p = "openai"

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    perplexity_key = os.getenv("PERPLEXITY_API_KEY", "").strip()
    perplexity_model = os.getenv("PERPLEXITY_MODEL", "sonar-pro").strip() or "sonar-pro"

    raw_queries = plan.get("search_queries") if isinstance(plan.get("search_queries"), list) else []
    queries = [_clean_text(str(q or "")) for q in raw_queries if _clean_text(str(q or ""))]
    if not queries:
        queries = _iter_queries(plan, max_queries=max_queries)
        warnings.append("v0.5 fallback to generated queries because analysis_plan.search_queries was empty.")
    queries = queries[: max(1, int(max_queries))]

    ref_text = _reference_text(plan)
    generated_queries: List[str] = []
    competitors: List[ProductCompetitorSlim] = []
    seen_urls: set[str] = set()
    manufacturer_cache: Dict[str, str] = {}

    _log(
        f"start provider={p} queries={len(queries)} per_query_results={per_query_results} "
        f"max_candidates_to_check={max_candidates_to_check}"
    )

    for i, q in enumerate(queries, start=1):
        if len(competitors) >= max_candidates_to_check:
            break
        generated_queries.append(q)
        _log(f"query {i}/{len(queries)}: {q}")
        results, source = _search_results(
            provider=p,
            query=q,
            per_query_results=per_query_results,
            openai_key=openai_key,
            openai_model=openai_model,
            perplexity_key=perplexity_key,
            perplexity_model=perplexity_model,
        )
        _log(f"search_results={len(results)} source={source}")

        for ridx, r in enumerate(results, start=1):
            if len(competitors) >= max_candidates_to_check:
                break
            title = _clean_text(str(r.get("title") or ""))
            url = _clean_url(str(r.get("url") or ""))
            snippet = _clean_text(str(r.get("snippet") or ""))
            if not title or not url:
                continue
            if url in seen_urls:
                continue
            seen_urls.add(url)

            if verbose_terminal and verbose_search_hits:
                _log(f"result {ridx}/{len(results)} title={title}")
                _log(f"result {ridx}/{len(results)} url={url}")
                _log(f"result {ridx}/{len(results)} snippet={snippet[:220]}")

            manufacturer = manufacturer_cache.get(url, "")
            if not manufacturer:
                manufacturer = _extract_manufacturer_llm(
                    provider=p,
                    title=title,
                    url=url,
                    snippet=snippet,
                    warnings=warnings,
                )
                if not manufacturer:
                    manufacturer = _manufacturer_from_title_or_domain(title, url)
                manufacturer_cache[url] = manufacturer
            url_type = _url_type(url)
            text = f"{title}\n{snippet}"
            semantic = _cosine_similarity(ref_text, text) if ref_text else 0.0
            query_sim = _cosine_similarity(q, text)
            relevance_score = max(0.0, min(1.0, 0.6 * semantic + 0.4 * query_sim))
            similarity_score = max(0.0, min(1.0, 0.5 * semantic + 0.5 * query_sim))

            competitors.append(
                ProductCompetitorSlim(
                    product_name=title,
                    manufacturer=manufacturer,
                    url=url,
                    url_type=url_type,
                    relevance_score=round(float(relevance_score), 4),
                    similarity_score=round(float(similarity_score), 4),
                )
            )

    _log(f"done competitors={len(competitors)} warnings={len(warnings)}")

    return CompetitorSearchResultsV05(
        schema_version="1.0",
        provider=p,
        generated_queries=generated_queries,
        competitors=competitors,
        extraction_warnings=list(dict.fromkeys([w for w in warnings if _clean_text(w)])),
    )
