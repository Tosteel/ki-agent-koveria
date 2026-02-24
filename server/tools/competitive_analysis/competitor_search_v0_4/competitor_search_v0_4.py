from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from server.services.llm_ionos import IonosLLM
from server.services.llm_openai import LlmOpenai
from server.services.llm_perplexity import LlmPerplexity
from server.tools.competitive_analysis.competitor_identification.competitor_identification import (
    _clean_text,
    _clean_url,
    _cluster_for_url,
    _domain,
    _iter_queries,
    _load_json_obj,
)

from bs4 import BeautifulSoup

from .models import CompanyCompetitorCandidate, CompetitorSearchResults

_COMPETITOR_TYPES = {"Direct competitor", "Indirect competitor", "Potential new competitor"}
_NON_COMPANY_DOMAIN_HINTS = {
    "google.",
    "bing.",
    "duckduckgo.com",
    "youtube.",
    "wikipedia.org",
    "reddit.com",
    "linkedin.com",
    "facebook.com",
    "instagram.com",
    "x.com",
    "twitter.com",
    "tiktok.com",
    "amazon.",
    "ebay.",
    "idealo.",
    "mediamarkt.",
    "saturn.",
    "otto.",
    "heise.de",
    "chip.de",
    "computerbild.de",
}
_RETAILER_DOMAIN_HINTS = {
    "amazon.",
    "ebay.",
    "otto.",
    "mediamarkt.",
    "saturn.",
    "galaxus.",
    "alltron.",
    "idealo.",
    "kaufland.",
    "walmart.",
    "bestbuy.",
    "aliexpress.",
}
_GENERIC_TITLE_TOKENS = {
    "home",
    "start",
    "seite",
    "official",
    "website",
    "shop",
    "store",
    "amazon",
    "vergleich",
    "test",
}


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str
    query: str


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
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    s = t.find("{")
    e = t.rfind("}")
    if s >= 0 and e > s:
        try:
            obj = json.loads(t[s : e + 1])
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _classifier_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "is_direct_competitor": {"type": "boolean"},
            "competitor_type": {"type": "string"},
            "relevance_score": {"type": "number"},
            "reasoning": {"type": "string"},
            "company_description": {"type": "string"},
            "year_founded": {"type": "integer"},
            "headquarters_country": {"type": "string"},
            "primary_business_segments": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "is_direct_competitor",
            "competitor_type",
            "relevance_score",
            "reasoning",
            "company_description",
            "year_founded",
            "headquarters_country",
            "primary_business_segments",
        ],
    }


def _clamp_score(value: Any) -> float:
    try:
        v = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, v))


def _clamp_year(value: Any) -> int:
    try:
        year = int(value)
    except Exception:
        return 0
    return year if 1800 <= year <= 2100 else 0


def _clean_segments(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
    seen: set[str] = set()
    for v in values:
        s = _clean_text(str(v or ""))
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out[:8]


def _domain_brand(url: str) -> str:
    d = _domain(url).lower()
    if not d:
        return ""
    host = d.split(":")[0]
    parts = [p for p in host.split(".") if p and p not in {"www", "com", "de", "eu", "co", "net", "org"}]
    return parts[0] if parts else ""


def _title_case_word(word: str) -> str:
    if not word:
        return ""
    if len(word) <= 3:
        return word.upper()
    return word[0].upper() + word[1:].lower()


def _is_retailer_domain(url: str) -> bool:
    d = _domain(url).lower()
    return bool(d and any(h in d for h in _RETAILER_DOMAIN_HINTS))


def _normalize_cluster(url: str, title: str, snippet: str) -> str:
    if _is_retailer_domain(url):
        return "retailer"
    c = _cluster_for_url(url, title, snippet)
    if c in {"marketplace", "media", "video"}:
        return "retailer"
    return "manufacturer"


def _normalize_url(value: str) -> str:
    u = _clean_url(str(value or "").strip())
    if not u:
        return ""
    if not re.match(r"^https?://", u, flags=re.IGNORECASE):
        return ""
    return u


def _is_bad_company_domain(url: str) -> bool:
    d = _domain(url).lower()
    return bool(not d or any(h in d for h in _NON_COMPANY_DOMAIN_HINTS))


def _company_name_key(name: str) -> str:
    n = _clean_text(name).lower()
    return re.sub(r"[^a-z0-9]+", " ", n).strip()


def _candidate_name_from_title(title: str, url: str) -> str:
    t = _clean_text(title)
    if t:
        chunks = [c.strip() for c in re.split(r"[|\-:•·–—]+", t) if _clean_text(c)]
        for c in chunks:
            words = [w for w in re.findall(r"[A-Za-zÄÖÜäöüß0-9\+\-]+", c) if w]
            if not words:
                continue
            trimmed = " ".join(words[:3]).strip()
            key = re.sub(r"[^a-z0-9]+", " ", trimmed.lower()).strip()
            if key and key not in _GENERIC_TITLE_TOKENS:
                return trimmed
    brand = _domain_brand(url)
    return _title_case_word(brand)


def _extract_http_from_text(text: str) -> str:
    t = str(text or "")
    m = re.search(r"https?://[^\s\"'<>]+", t, flags=re.IGNORECASE)
    return _clean_text(m.group(0)) if m else ""


def _decode_bing_u_param(raw_value: str) -> str:
    v = _clean_text(unquote(str(raw_value or "")))
    if not v:
        return ""
    # Sometimes already plain URL in query param.
    if v.startswith("http://") or v.startswith("https://"):
        return v

    # Common bing format: a1<base64-url>.
    candidates = [v]
    m = re.match(r"^[a-z]\d(.+)$", v, flags=re.IGNORECASE)
    if m:
        candidates.append(m.group(1))

    for c in candidates:
        b64 = c.strip()
        if not b64:
            continue
        b64 = b64.replace("-", "+").replace("_", "/")
        b64 += "=" * ((4 - len(b64) % 4) % 4)
        try:
            dec = base64.b64decode(b64).decode("utf-8", errors="ignore")
        except Exception:
            continue
        url = _extract_http_from_text(dec)
        if url:
            return url
    return ""


def _clean_bing_href(href: str) -> str:
    h = _clean_text(str(href or ""))
    if not h:
        return ""
    if h.startswith("/"):
        h = "https://www.bing.com" + h
    if h.startswith("http://") or h.startswith("https://"):
        parsed = urlparse(h)
        host = (parsed.netloc or "").lower()
        if "bing." not in host:
            return h
        # Bing redirect wrappers: /ck/a, /aclick, etc.
        qs = parse_qs(parsed.query or "")
        for key in ("u", "url", "r"):
            vals = qs.get(key) or []
            for v in vals:
                dec = _decode_bing_u_param(v)
                if dec:
                    return dec
        # No decodable target found.
        return h
    return ""


def _extract_bing_hits(html: str, *, query: str, max_results: int) -> List[SearchHit]:
    soup = BeautifulSoup(html or "", "html.parser")
    out: List[SearchHit] = []
    seen_urls: set[str] = set()
    raw_candidates = 0
    filtered_bad_domain = 0
    filtered_invalid_url = 0
    decoded_redirect = 0

    # Primary pattern: Bing organic results.
    primary_blocks = soup.select("li.b_algo")
    if not primary_blocks:
        primary_blocks = soup.select("main li[data-idx], #b_results > li")

    for block in primary_blocks:
        a = block.select_one("h2 a[href], a[href]")
        if a is None:
            continue
        raw_candidates += 1
        raw_href = str(a.get("href") or "")
        cleaned = _clean_bing_href(raw_href)
        if cleaned and cleaned != raw_href:
            decoded_redirect += 1
        url = _normalize_url(cleaned)
        if not url:
            filtered_invalid_url += 1
            continue
        if _is_bad_company_domain(url):
            filtered_bad_domain += 1
            continue
        key = url.lower()
        if key in seen_urls:
            continue
        seen_urls.add(key)
        h3 = block.select_one("h2, h3")
        snippet_node = block.select_one("div.b_caption p, p.b_lineclamp2, p")
        snippet = _clean_text(snippet_node.get_text(" ", strip=True) if snippet_node else "")
        title = _clean_text((h3 or a).get_text(" ", strip=True))
        out.append(SearchHit(title=title, url=url, snippet=snippet, query=query))
        if len(out) >= max_results:
            break

    # Fallback: any external absolute link on result page.
    if len(out) < max_results:
        for a in soup.select("a[href^='http']"):
            raw_candidates += 1
            raw_href = str(a.get("href") or "")
            cleaned = _clean_bing_href(raw_href)
            if cleaned and cleaned != raw_href:
                decoded_redirect += 1
            url = _normalize_url(cleaned)
            if not url:
                filtered_invalid_url += 1
                continue
            if _is_bad_company_domain(url):
                filtered_bad_domain += 1
                continue
            key = url.lower()
            if key in seen_urls:
                continue
            seen_urls.add(key)
            title = _clean_text(a.get_text(" ", strip=True)) or _title_case_word(_domain_brand(url))
            out.append(SearchHit(title=title, url=url, snippet="", query=query))
            if len(out) >= max_results:
                break

    out.append(
        SearchHit(
            title=(
                "__debug_bing__"
                f"raw={raw_candidates};invalid={filtered_invalid_url};"
                f"bad_domain={filtered_bad_domain};decoded_redirect={decoded_redirect};kept={len(out)}"
            ),
            url="debug://bing-parser",
            snippet="",
            query=query,
        )
    )
    return out


def _bing_search_hits_with_playwright(*, query: str, max_results: int, timeout_ms: int) -> List[SearchHit]:
    search_url = f"https://www.bing.com/search?setlang=de&count={max(5, min(20, max_results))}&q={quote_plus(query)}"
    html = ""

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            context = browser.new_context(
                locale="de-DE",
                user_agent=(
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                ),
            )
            page = context.new_page()
            page.goto(search_url, wait_until="domcontentloaded", timeout=timeout_ms)
            try:
                page.wait_for_selector("li.b_algo, #b_results", timeout=min(5000, timeout_ms))
            except Exception:
                pass
            page.wait_for_timeout(700)
            html = page.content()
            title = _clean_text(page.title())
            url = _clean_text(page.url)
            _ = (title, url)  # keep for symmetry with debug via parser below
            context.close()
        finally:
            browser.close()

    return _extract_bing_hits(html, query=query, max_results=max_results)


def _extract_duckduckgo_hits(html: str, *, query: str, max_results: int) -> List[SearchHit]:
    soup = BeautifulSoup(html or "", "html.parser")
    out: List[SearchHit] = []
    seen_urls: set[str] = set()
    raw_candidates = 0
    filtered_bad_domain = 0
    filtered_invalid_url = 0

    primary_blocks = soup.select("article[data-testid='result'], div[data-testid='result'], div.result")
    if not primary_blocks:
        primary_blocks = soup.select("main article, main div")

    for block in primary_blocks:
        a = block.select_one("a[data-testid='result-title-a'][href], h2 a[href], a.result__a[href], a[href]")
        if a is None:
            continue
        raw_candidates += 1
        url = _normalize_url(str(a.get("href") or ""))
        if not url:
            filtered_invalid_url += 1
            continue
        if _is_bad_company_domain(url):
            filtered_bad_domain += 1
            continue
        key = url.lower()
        if key in seen_urls:
            continue
        seen_urls.add(key)
        h3 = block.select_one("h2, h3, span[data-testid='result-title']")
        snippet_node = block.select_one("div[data-result='snippet'], div.result__snippet, p")
        snippet = _clean_text(snippet_node.get_text(" ", strip=True) if snippet_node else "")
        title = _clean_text((h3 or a).get_text(" ", strip=True))
        out.append(SearchHit(title=title, url=url, snippet=snippet, query=query))
        if len(out) >= max_results:
            break

    if len(out) < max_results:
        for a in soup.select("a[href^='http']"):
            raw_candidates += 1
            url = _normalize_url(str(a.get("href") or ""))
            if not url:
                filtered_invalid_url += 1
                continue
            if _is_bad_company_domain(url):
                filtered_bad_domain += 1
                continue
            key = url.lower()
            if key in seen_urls:
                continue
            seen_urls.add(key)
            title = _clean_text(a.get_text(" ", strip=True)) or _title_case_word(_domain_brand(url))
            out.append(SearchHit(title=title, url=url, snippet="", query=query))
            if len(out) >= max_results:
                break

    out.append(
        SearchHit(
            title=(
                "__debug_ddg__"
                f"raw={raw_candidates};invalid={filtered_invalid_url};"
                f"bad_domain={filtered_bad_domain};kept={len(out)}"
            ),
            url="debug://ddg-parser",
            snippet="",
            query=query,
        )
    )
    return out


def _duckduckgo_search_hits_with_playwright(*, query: str, max_results: int, timeout_ms: int) -> List[SearchHit]:
    search_url = f"https://duckduckgo.com/?q={quote_plus(query)}&kl=de-de"
    html = ""

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            context = browser.new_context(
                locale="de-DE",
                user_agent=(
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                ),
            )
            page = context.new_page()
            page.goto(search_url, wait_until="domcontentloaded", timeout=timeout_ms)
            try:
                page.wait_for_selector(
                    "article[data-testid='result'], a[data-testid='result-title-a'], a.result__a",
                    timeout=min(5000, timeout_ms),
                )
            except Exception:
                pass
            page.wait_for_timeout(700)
            html = page.content()
            context.close()
        finally:
            browser.close()

    return _extract_duckduckgo_hits(html, query=query, max_results=max_results)


def _llm_classify_candidate(
    *,
    provider: str,
    reference_company: str,
    product_category: str,
    query: str,
    candidate_name: str,
    candidate_domain: str,
    evidence_title: str,
    evidence_snippet: str,
) -> Dict[str, Any]:
    system = (
        "You classify if a company is a direct competitor in a specific product category. "
        "Return JSON only."
    )
    user = (
        f"Reference company: {reference_company or 'unknown'}\n"
        f"Product category: {product_category}\n"
        f"Search query: {query}\n"
        f"Candidate company: {candidate_name}\n"
        f"Candidate domain: {candidate_domain}\n"
        f"Evidence title: {evidence_title}\n"
        f"Evidence snippet: {evidence_snippet}\n\n"
        "Rules:\n"
        "1) is_direct_competitor=true only if candidate offers comparable products/services in this category.\n"
        "2) Mark media, retailers, marketplaces and generic blogs as not direct competitors.\n"
        "3) competitor_type must be one of: Direct competitor, Indirect competitor, Potential new competitor.\n"
        "4) relevance_score must be between 0 and 1.\n"
        "5) Keep reasoning concise."
    )

    p = str(provider or "openai").strip().lower()
    if p not in {"openai", "perplexity", "ionos"}:
        p = "openai"

    if p in {"openai", "perplexity"}:
        client = LlmOpenai() if p == "openai" else LlmPerplexity()
        if not client.enabled():
            return {}
        fmt = {
            "type": "json_schema",
            "name": "direct_competitor_check",
            "schema": _classifier_schema(),
            "strict": False,
        }
        try:
            resp = client._call(
                input_messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                text_format=fmt,
            )
            txt = ""
            if p == "openai":
                for item in resp.get("output", []):
                    for c in item.get("content", []):
                        if c.get("type") == "output_text":
                            txt += str(c.get("text") or "")
            else:
                txt = str(resp.get("choices", [{}])[0].get("message", {}).get("content") or "")
            return _parse_json_strictish(txt)
        except Exception:
            return {}

    ion = IonosLLM()
    if not ion.enabled():
        return {}
    try:
        completion = ion.chat_completions(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "direct_competitor_check",
                    "schema": _classifier_schema(),
                    "strict": True,
                },
            },
        )
        return _parse_json_strictish(ion.extract_text(completion))
    except Exception:
        return {}


def search_competitors_v0_4(
    *,
    analysis_plan: Optional[Dict[str, Any]],
    analysis_plan_path: Optional[str],
    provider: str = "openai",
    max_queries: int = 20,
    per_query_results: int = 10,
    shortlist_size: int = 12,
    max_candidates_to_check: int = 40,
    min_relevance_score: float = 0.15,
    search_timeout_ms: int = 20000,
    verbose_terminal: bool = False,
    user_root,
    work_root,
) -> CompetitorSearchResults:
    def _log(msg: str) -> None:
        if verbose_terminal:
            print(f"[competitor_search_v0_4] {msg}")

    plan = _load_json_obj(
        inline_obj=analysis_plan,
        path=analysis_plan_path,
        root_key="analysis_plan",
        user_root=user_root,
        work_root=work_root,
    )
    warnings = [str(w).strip() for w in (plan.get("extraction_warnings") or []) if str(w).strip()]

    def _load_v0_2_queries() -> List[str]:
        candidates = [
            "work/analysis_plan_v0_2.json",
            "analysis_plan_v0_2.json",
        ]
        for rel in candidates:
            try:
                obj = _load_json_obj(
                    inline_obj=None,
                    path=rel,
                    root_key="analysis_plan",
                    user_root=user_root,
                    work_root=work_root,
                )
            except Exception:
                _log(f"analysis_plan_v0_2 candidate not usable: {rel}")
                continue
            raw = obj.get("search_queries") if isinstance(obj.get("search_queries"), list) else []
            qs = [_clean_text(str(q or "")) for q in raw if _clean_text(str(q or ""))]
            if qs:
                _log(f"loaded search_queries from {rel}: {len(qs)}")
                return qs
        return []

    p = str(provider or "openai").strip().lower()
    if p not in {"openai", "perplexity", "ionos"}:
        p = "openai"

    category = _clean_text(str(plan.get("product_category") or "")) or "product"
    search_terms = plan.get("search_terms") if isinstance(plan.get("search_terms"), list) else []
    ref_company = ""
    for term_obj in search_terms:
        if not isinstance(term_obj, dict):
            continue
        intent = _clean_text(str(term_obj.get("intent") or "")).lower()
        if intent == "brand":
            ref_company = _clean_text(str(term_obj.get("term") or ""))
            if ref_company:
                break

    v0_2_queries = _load_v0_2_queries()
    raw_queries = plan.get("search_queries") if isinstance(plan.get("search_queries"), list) else []
    queries = [_clean_text(str(q or "")) for q in raw_queries if _clean_text(str(q or ""))]
    if v0_2_queries:
        queries = v0_2_queries
        _log(f"using analysis_plan_v0_2 queries: {len(queries)}")
    else:
        warnings.append("v0.4 analysis_plan_v0_2.json not found or without search_queries; using request plan queries.")
        _log(f"using request/fallback queries: {len(queries)}")
    if not queries:
        queries = _iter_queries(plan, max_queries=max_queries)
        warnings.append("v0.4 fallback to generated queries because analysis_plan.search_queries was empty.")
    # Add competitor-focused queries to avoid purely informational SERPs.
    focused_queries: List[str] = []
    if category:
        focused_queries.extend(
            [
                f"{category} hersteller marken",
                f"{category} direktvergleich marken",
                f"beste {category} marken",
            ]
        )
    if ref_company and category:
        focused_queries.extend(
            [
                f"alternativen zu {ref_company} {category}",
                f"{ref_company} wettbewerber {category}",
            ]
        )
    for fq in focused_queries:
        q = _clean_text(fq)
        if q and q not in queries:
            queries.append(q)
    queries = queries[: max(1, int(max_queries))]

    min_comp = int(plan.get("min_competitors") or 6)
    min_comp = max(2, min(50, min_comp))
    target_count = max(min_comp, min(50, int(shortlist_size)))
    min_rel = max(0.0, min(1.0, float(min_relevance_score)))

    _log(
        f"start provider={p} category={category} queries={len(queries)} "
        f"target_count={target_count} min_relevance={min_rel:.2f}"
    )

    all_hits: List[SearchHit] = []
    for idx, q in enumerate(queries, start=1):
        _log(f"query {idx}/{len(queries)}: {q}")
        try:
            hits = _duckduckgo_search_hits_with_playwright(query=q, max_results=per_query_results, timeout_ms=search_timeout_ms)
            real_hits: List[SearchHit] = []
            for h in hits:
                if h.url == "debug://ddg-parser":
                    _log(h.title)
                    continue
                real_hits.append(h)
            _log(f"ddg_hits={len(real_hits)}")
            all_hits.extend(real_hits)
        except PlaywrightTimeoutError:
            warnings.append(f"v0.4 Playwright/DuckDuckGo timeout for query: {q}")
        except Exception as exc:
            warnings.append(f"v0.4 DuckDuckGo search failed for query '{q}': {type(exc).__name__}")

        if len(all_hits) >= max_candidates_to_check * 2:
            _log("hit soft limit for collected SERP results")
            break

    # Deduplicate by domain and keep first hit as representative evidence.
    by_domain: Dict[str, SearchHit] = {}
    for hit in all_hits:
        d = _domain(hit.url).lower()
        if not d:
            continue
        if d not in by_domain:
            by_domain[d] = hit
        if len(by_domain) >= max_candidates_to_check:
            break
    _log(f"candidate domains after dedupe: {len(by_domain)}")

    competitors: List[CompanyCompetitorCandidate] = []
    by_name_index: Dict[str, int] = {}

    if not by_domain:
        warnings.append("v0.4 no parsable DuckDuckGo SERP hits extracted (possible block/selector mismatch).")

    for d, hit in by_domain.items():
        candidate_url = f"https://{d}"
        if _is_bad_company_domain(candidate_url):
            _log(f"drop candidate bad-domain: {candidate_url}")
            continue

        candidate_name = _candidate_name_from_title(hit.title, candidate_url)
        if not candidate_name:
            _log(f"drop candidate no-name: domain={d}")
            continue
        _log(f"classify candidate: {candidate_name} ({candidate_url})")

        llm_obj = _llm_classify_candidate(
            provider=p,
            reference_company=ref_company,
            product_category=category,
            query=hit.query,
            candidate_name=candidate_name,
            candidate_domain=d,
            evidence_title=hit.title,
            evidence_snippet=hit.snippet,
        )
        if not isinstance(llm_obj, dict) or not llm_obj:
            warnings.append(f"v0.4 LLM classification failed for candidate: {candidate_name}")
            _log(f"drop candidate llm-failed: {candidate_name}")
            continue

        is_direct = bool(llm_obj.get("is_direct_competitor"))
        competitor_type = _clean_text(str(llm_obj.get("competitor_type") or "Direct competitor"))
        competitor_type_l = competitor_type.lower()
        score = _clamp_score(llm_obj.get("relevance_score"))

        if competitor_type not in _COMPETITOR_TYPES:
            if "direct" in competitor_type_l or "direkt" in competitor_type_l:
                competitor_type = "Direct competitor"
            elif "indirect" in competitor_type_l or "indirekt" in competitor_type_l:
                competitor_type = "Indirect competitor"
            else:
                competitor_type = "Direct competitor" if is_direct else "Indirect competitor"

        if not is_direct:
            _log(f"drop candidate not-direct: {candidate_name} type={competitor_type} score={score:.3f}")
            continue
        if competitor_type != "Direct competitor":
            _log(f"drop candidate wrong-type: {candidate_name} type={competitor_type}")
            continue
        if score < min_rel:
            _log(f"drop candidate low-score: {candidate_name} score={score:.3f} min={min_rel:.3f}")
            continue

        item = CompanyCompetitorCandidate(
            name=candidate_name,
            cluster=_normalize_cluster(hit.url, hit.title, hit.snippet),
            year_founded=_clamp_year(llm_obj.get("year_founded")),
            headquarters_country=_clean_text(str(llm_obj.get("headquarters_country") or "")),
            company_description=_clean_text(str(llm_obj.get("company_description") or "")),
            primary_business_segments=_clean_segments(llm_obj.get("primary_business_segments")),
            relevance_in_reference_segment=_clean_text(str(llm_obj.get("reasoning") or "")),
            competitor_type="Direct competitor",
            company_website_url=candidate_url,
            brand_domain_whitelist=[candidate_url],
            relevance_score=score,
        )

        nkey = _company_name_key(item.name)
        if not nkey:
            continue
        if nkey in by_name_index:
            existing_idx = by_name_index[nkey]
            existing = competitors[existing_idx]
            if item.relevance_score > existing.relevance_score + 1e-6:
                competitors[existing_idx] = item
                _log(f"replace duplicate by higher score: {item.name} {existing.relevance_score:.3f}->{item.relevance_score:.3f}")
            else:
                _log(f"drop duplicate: {item.name}")
            continue

        by_name_index[nkey] = len(competitors)
        competitors.append(item)
        _log(f"accepted {item.name} relevance={item.relevance_score:.3f}")
        if len(competitors) >= target_count:
            break

    if len(competitors) < min_comp:
        warnings.append(f"v0.4 only {len(competitors)} companies found, below min_competitors={min_comp}.")

    warnings.append("Generated via Playwright + BeautifulSoup search parsing and LLM direct-competitor gate v0.4.")
    warnings = list(dict.fromkeys([w for w in warnings if _clean_text(w)]))
    _log(f"done companies={len(competitors)} warnings={len(warnings)}")

    return CompetitorSearchResults(
        provider=p,
        generated_queries=queries,
        min_competitors_target=min_comp,
        competitors=competitors,
        extraction_warnings=warnings,
    )
