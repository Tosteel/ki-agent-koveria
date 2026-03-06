from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from server.services.llm_ionos import IonosLLM
from server.services.llm_openai import LlmOpenai
from server.services.llm_perplexity import LlmPerplexity
from server.tools.competitive_analysis.backup.competitor_identification.competitor_identification import (
    _clean_text,
    _clean_url,
    _cluster_for_url,
    _domain,
    _iter_queries,
    _langsearch_fallback,
    _load_json_obj,
    _openai_search,
    _perplexity_search,
)

from .models import CompanyCompetitorCandidate, CompetitorSearchResults

_COMPETITOR_TYPES = {"Direct competitor", "Indirect competitor", "Potential new competitor"}
_MODEL_TOKENS = {
    "pro",
    "ultra",
    "max",
    "plus",
    "combo",
    "complete",
    "edition",
    "series",
    "model",
    "rvf",
    "rv",
    "x",
    "s",
    "q",
    "j",
    "v",
}
_NON_COMPANY_DOMAIN_HINTS = {
    "bild.de",
    "computerbild.de",
    "chip.de",
    "heise.de",
    "techradar.com",
    "testsieger.de",
    "mediamarkt.de",
    "mediamarkt.at",
    "saturn.de",
    "otto.de",
    "amazon.",
    "ebay.",
    "idealo.",
    "galaxus.",
    "alltron.",
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


def _company_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string"},
            "cluster": {"type": "string"},
            "year_founded": {"type": "integer"},
            "headquarters_country": {"type": "string"},
            "company_description": {"type": "string"},
            "primary_business_segments": {"type": "array", "items": {"type": "string"}},
            "relevance_in_reference_segment": {"type": "string"},
            "competitor_type": {"type": "string"},
            "company_website_url": {"type": "string"},
            "brand_domain_whitelist": {"type": "array", "items": {"type": "string"}},
            "brand_domain_customerlist": {"type": "array", "items": {"type": "string"}},
            "relevance_score": {"type": "number"},
        },
        "required": [
            "name",
            "cluster",
            "year_founded",
            "headquarters_country",
            "company_description",
            "primary_business_segments",
            "relevance_in_reference_segment",
            "competitor_type",
            "company_website_url",
            "brand_domain_whitelist",
            "relevance_score",
        ],
    }


def _normalize_url(value: str) -> str:
    u = _clean_url(str(value or "").strip())
    if not u:
        return ""
    if not re.match(r"^https?://", u, flags=re.IGNORECASE):
        return ""
    return u


def _domain_brand(url: str) -> str:
    d = _domain(url).lower()
    if not d:
        return ""
    host = d.split(":")[0]
    parts = [p for p in host.split(".") if p and p not in {"www", "com", "de", "eu", "co", "net", "org"}]
    if not parts:
        return ""
    return parts[0]


def _is_bad_company_domain(url: str) -> bool:
    d = _domain(url).lower()
    if not d:
        return True
    return any(h in d for h in _NON_COMPANY_DOMAIN_HINTS)


def _title_case_word(word: str) -> str:
    if not word:
        return ""
    if len(word) <= 3:
        return word.upper()
    return word[0].upper() + word[1:].lower()


def _normalize_company_name(raw_name: str, company_url: str) -> str:
    raw = _clean_text(raw_name)
    if not raw:
        b = _domain_brand(company_url)
        return _title_case_word(b)

    tokens = re.findall(r"[A-Za-zÄÖÜäöüß0-9\+\-]+", raw)
    if not tokens:
        b = _domain_brand(company_url)
        return _title_case_word(b)

    # Keep only leading organization-like tokens; strip model-like tail.
    kept: List[str] = []
    for i, tok in enumerate(tokens):
        low = tok.lower()
        has_digit = bool(re.search(r"\d", tok))
        if i == 0:
            kept.append(tok)
            continue
        if has_digit:
            break
        if low in _MODEL_TOKENS:
            break
        if len(tok) <= 1:
            break
        kept.append(tok)
        if len(kept) >= 3:
            break

    out = _clean_text(" ".join(kept))
    # Avoid generic non-company outputs.
    if out.lower() in {"product", "products", "model", "models", "category"}:
        out = ""

    if not out:
        b = _domain_brand(company_url)
        out = _title_case_word(b)
    return out


def _company_name_key(name: str) -> str:
    n = _clean_text(name).lower()
    n = re.sub(r"[^a-z0-9]+", " ", n).strip()
    return n


def _is_retailer_domain(url: str) -> bool:
    d = _domain(url).lower()
    if not d:
        return False
    return any(h in d for h in _RETAILER_DOMAIN_HINTS)


def _normalize_cluster(raw_cluster: str, company_url: str, ref_url: str) -> str:
    r = _clean_text(raw_cluster).lower()
    if "retail" in r or "shop" in r or "marketplace" in r:
        return "retailer"
    if _is_retailer_domain(company_url) or _is_retailer_domain(ref_url):
        return "retailer"
    return "manufacturer"


def _company_name_tokens(name: str) -> List[str]:
    toks = [t for t in re.findall(r"[a-z0-9]{3,}", _clean_text(name).lower()) if t]
    return toks[:4]


def _domain_matches_company(domain: str, company_name: str, company_url: str) -> bool:
    d = (domain or "").lower().strip()
    if not d:
        return False
    if company_url:
        cd = _domain(company_url).lower()
        if d == cd or d.endswith("." + cd) or cd.endswith("." + d):
            return True
    for tok in _company_name_tokens(company_name):
        if tok in d:
            return True
    return False


def _is_strong_company_domain_match(domain: str, company_name: str, company_url: str) -> bool:
    d = (domain or "").lower().strip()
    if not d:
        return False
    brand = _domain_brand(f"https://{d}")
    if not brand:
        return False
    tokens = _company_name_tokens(company_name)
    if company_url:
        tokens.extend(_company_name_tokens(_domain_brand(company_url)))
    for tok in tokens:
        if len(tok) < 4:
            continue
        if brand.startswith(tok) or tok.startswith(brand):
            return True
        if tok in brand:
            return True
    return False


def _normalize_brand_domain_whitelist(values: Any, company_url: str, company_name: str, observed_domains: set[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    if isinstance(values, list):
        candidates = [str(v or "").strip() for v in values]
    else:
        candidates = []

    if company_url:
        candidates.insert(0, company_url)

    for c in candidates:
        u = _normalize_url(c)
        if not u:
            continue
        if _is_bad_company_domain(u):
            continue
        # keep https://<domain> canonical form
        d = _domain(u)
        if not d:
            continue
        dl = d.lower()
        # Relax strict "must be observed" gate: keep strong brand-domain matches
        # so official sites with suffixes (e.g. dreametech.com) are not dropped.
        if observed_domains and dl not in observed_domains and not _is_strong_company_domain_match(dl, company_name, company_url):
            continue
        if not _domain_matches_company(dl, company_name, company_url):
            continue
        canon = f"https://{d}"
        k = canon.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(canon)
    return out[:12]


def _discover_brand_domains_via_search(
    *,
    provider: str,
    company_name: str,
    product_category: str,
    per_query_results: int,
    openai_key: str,
    openai_model: str,
    perplexity_key: str,
    perplexity_model: str,
) -> set[str]:
    queries = [
        f"{company_name} official website",
        f"{company_name} {product_category}",
        f"{company_name} {product_category} official website",
        f"{company_name} {product_category} official site",
        f"{company_name} global site",
    ]
    domains: set[str] = set()
    for q in queries:
        results, _src = _search_results(
            provider=provider,
            query=q,
            per_query_results=max(4, min(10, per_query_results)),
            openai_key=openai_key,
            openai_model=openai_model,
            perplexity_key=perplexity_key,
            perplexity_model=perplexity_model,
        )
        for r in results:
            u = _clean_url(str(r.get("url") or "").strip())
            if not u:
                continue
            if _is_bad_company_domain(u):
                continue
            d = _domain(u).lower()
            if not d:
                continue
            domains.add(d)
    return domains


def _clean_segments(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
    seen: set[str] = set()
    for v in values:
        s = _clean_text(str(v or ""))
        if not s:
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out[:8]


def _company_key(name: str, website: str) -> str:
    n = re.sub(r"[^a-z0-9]+", " ", _clean_text(name).lower()).strip()
    d = _domain(website) if website else ""
    return f"{n}|{d}".strip("|")


def _clamp_year(value: Any) -> int:
    try:
        year = int(value)
    except Exception:
        return 0
    return year if 1800 <= year <= 2100 else 0


def _clamp_score(value: Any) -> float:
    try:
        v = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, v))


def _extract_customer_links(payload: Dict[str, Any]) -> List[str]:
    vals = payload.get("competitor_links")
    if not isinstance(vals, list):
        return []
    out: List[str] = []
    seen: set[str] = set()
    for v in vals:
        u = _clean_url(str(v or "").strip())
        if not u:
            continue
        k = u.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(u)
    return out


def _link_matches_competitor(link: str, item: CompanyCompetitorCandidate) -> bool:
    d = _domain(link).lower()
    if not d:
        return False
    if item.company_website_url:
        cd = _domain(item.company_website_url).lower()
        if d == cd or d.endswith("." + cd) or cd.endswith("." + d):
            return True
    for u in item.brand_domain_whitelist:
        wd = _domain(u).lower()
        if not wd:
            continue
        if d == wd or d.endswith("." + wd) or wd.endswith("." + d):
            return True
    return _domain_matches_company(d, item.name, item.company_website_url)


def _llm_extract_company(
    *,
    provider: str,
    product_category: str,
    query: str,
    evidence_items: List[Dict[str, str]],
) -> Dict[str, Any]:
    evidence_text = "\n".join(
        [
            f"- title={_clean_text(i.get('title') or '')} | url={_clean_url(i.get('url') or '')} | snippet={_clean_text(i.get('snippet') or '')}"
            for i in evidence_items[:8]
        ]
    )
    system = (
        "You identify exactly one competitor company from search evidence. "
        "Return only JSON, no markdown."
    )
    user = (
        f"Reference segment/category: {product_category}\n"
        f"Search query: {query}\n"
        f"Evidence:\n{evidence_text}\n\n"
        "Task:\n"
        "1) Return exactly one relevant competitor company.\n"
        "2) Fill all fields with concise factual values.\n"
        "3) competitor_type must be one of: Direct competitor, Indirect competitor, Potential new competitor.\n"
        "4) relevance_score must be 0..1.\n"
        "5) Use real URLs from evidence whenever possible.\n"
        "6) IMPORTANT: name must be the company/brand name only, never a model or product variant.\n"
        "7) If evidence shows a model name, map it to the owning company (e.g., 'MIDEA S8+' -> 'Midea').\n"
        "8) company_website_url must be the official company domain (not media/test/shop article URL).\n"
        "9) brand_domain_whitelist must contain official brand domains (regional hosts allowed), as URLs.\n"
        "10) brand_domain_customerlist should be an empty array unless explicit customer links are provided externally.\n"
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
            "name": "competitor_company_profile",
            "schema": _company_schema(),
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
                    "name": "competitor_company_profile",
                    "schema": _company_schema(),
                    "strict": True,
                },
            },
        )
        return _parse_json_strictish(ion.extract_text(completion))
    except Exception:
        return {}


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


def search_competitors_v0_3(
    *,
    analysis_plan: Optional[Dict[str, Any]],
    analysis_plan_path: Optional[str],
    product_competitors: Optional[Dict[str, Any]] = None,
    product_competitors_path: Optional[str] = None,
    provider: str = "openai",
    max_queries: int = 20,
    per_query_results: int = 8,
    shortlist_size: int = 12,
    min_relevance_score: float = 0.15,
    verbose_terminal: bool = False,
    verbose_search_hits: bool = False,
    user_root,
    work_root,
) -> CompetitorSearchResults:
    def _log(msg: str) -> None:
        if verbose_terminal:
            print(f"[competitor_search_v0_3] {msg}")

    plan = _load_json_obj(
        inline_obj=analysis_plan,
        path=analysis_plan_path,
        root_key="analysis_plan",
        user_root=user_root,
        work_root=work_root,
    )
    customer_payload: Dict[str, Any] = {}
    if (isinstance(product_competitors, dict) and product_competitors) or (product_competitors_path or "").strip():
        customer_payload = _load_json_obj(
            inline_obj=product_competitors,
            path=product_competitors_path,
            root_key="product_competitors",
            user_root=user_root,
            work_root=work_root,
        )
    customer_links = _extract_customer_links(customer_payload) if customer_payload else []
    warnings = [str(w).strip() for w in (plan.get("extraction_warnings") or []) if str(w).strip()]

    p = str(provider or "openai").strip().lower()
    if p not in {"openai", "perplexity", "ionos"}:
        p = "openai"

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    perplexity_key = os.getenv("PERPLEXITY_API_KEY", "").strip()
    perplexity_model = os.getenv("PERPLEXITY_MODEL", "sonar-pro").strip() or "sonar-pro"

    category = _clean_text(str(plan.get("product_category") or "")) or "product"
    raw_queries = plan.get("search_queries") if isinstance(plan.get("search_queries"), list) else []
    queries = [_clean_text(str(q or "")) for q in raw_queries if _clean_text(str(q or ""))]
    if not queries:
        queries = _iter_queries(plan, max_queries=max_queries)
        warnings.append("v0.3 fallback to generated queries because analysis_plan.search_queries was empty.")
    queries = queries[: max(1, int(max_queries))]

    min_comp = int(plan.get("min_competitors") or 6)
    min_comp = max(2, min(50, min_comp))
    target_count = max(min_comp, min(50, int(shortlist_size)))
    min_rel = max(0.0, min(1.0, float(min_relevance_score)))

    _log(
        f"start provider={p} category={category} queries={len(queries)} "
        f"target_count={target_count} min_relevance={min_rel:.2f}"
    )
    if customer_links:
        _log(f"customer_links={len(customer_links)}")

    competitors: List[CompanyCompetitorCandidate] = []
    by_name_index: Dict[str, int] = {}

    for idx, q in enumerate(queries, start=1):
        if len(competitors) >= target_count:
            break
        _log(f"query {idx}/{len(queries)}: {q}")

        results, source_label = _search_results(
            provider=p,
            query=q,
            per_query_results=per_query_results,
            openai_key=openai_key,
            openai_model=openai_model,
            perplexity_key=perplexity_key,
            perplexity_model=perplexity_model,
        )
        if not results:
            _log("no search results")
            warnings.append(f"v0.3 no web results for query: {q}")
            continue

        _log(f"search_results={len(results)} source={source_label}")
        if verbose_terminal and verbose_search_hits:
            for ridx, r in enumerate(results, start=1):
                r_title = _clean_text(str(r.get("title") or ""))
                r_url = _clean_url(str(r.get("url") or ""))
                r_snippet = _clean_text(str(r.get("snippet") or ""))
                if len(r_snippet) > 220:
                    r_snippet = r_snippet[:217].rstrip() + "..."
                _log(f"result {ridx}/{len(results)} title={r_title}")
                _log(f"result {ridx}/{len(results)} url={r_url}")
                _log(f"result {ridx}/{len(results)} snippet={r_snippet}")
        obj = _llm_extract_company(
            provider=p,
            product_category=category,
            query=q,
            evidence_items=results,
        )
        if not isinstance(obj, dict) or not obj:
            _log("llm extraction failed")
            warnings.append(f"v0.3 LLM extraction failed for query: {q}")
            continue

        company_url = _normalize_url(str(obj.get("company_website_url") or ""))
        evidence_url = _clean_url(str(results[0].get("url") or "").strip())

        # Reject media/shop/news as company homepage.
        if company_url and _is_bad_company_domain(company_url):
            company_url = ""

        raw_name = _clean_text(str(obj.get("name") or ""))
        normalized_name = _normalize_company_name(raw_name, company_url)
        if raw_name and normalized_name and raw_name != normalized_name:
            _log(f"name normalized: '{raw_name}' -> '{normalized_name}'")

        observed_domains: set[str] = set()
        for r in results:
            u = _clean_url(str(r.get("url") or "").strip())
            if not u:
                continue
            d = _domain(u).lower()
            if not d:
                continue
            observed_domains.add(d)

        discovered_domains = _discover_brand_domains_via_search(
            provider=p,
            company_name=normalized_name or raw_name,
            product_category=category,
            per_query_results=per_query_results,
            openai_key=openai_key,
            openai_model=openai_model,
            perplexity_key=perplexity_key,
            perplexity_model=perplexity_model,
        )
        observed_domains |= discovered_domains
        _log(f"domains observed={len(observed_domains)} discovered={len(discovered_domains)}")

        whitelist = _normalize_brand_domain_whitelist(
            obj.get("brand_domain_whitelist"),
            company_url,
            normalized_name or raw_name,
            observed_domains,
        )
        # Ensure discovered domains are available even when LLM list is empty/incomplete.
        for d in sorted(discovered_domains):
            if not _domain_matches_company(d, normalized_name or raw_name, company_url):
                continue
            u = f"https://{d}"
            if u not in whitelist:
                whitelist.append(u)
            if len(whitelist) >= 12:
                break

        competitor_type = _clean_text(str(obj.get("competitor_type") or "Direct competitor"))
        if competitor_type not in _COMPETITOR_TYPES:
            competitor_type = "Direct competitor"

        item = CompanyCompetitorCandidate(
            name=normalized_name,
            cluster=_normalize_cluster(
                _clean_text(str(obj.get("cluster") or "")) or _cluster_for_url(
                    evidence_url,
                    _clean_text(str(results[0].get("title") or "")),
                    _clean_text(str(results[0].get("snippet") or "")),
                ),
                company_url,
                evidence_url,
            ),
            year_founded=_clamp_year(obj.get("year_founded")),
            headquarters_country=_clean_text(str(obj.get("headquarters_country") or "")),
            company_description=_clean_text(str(obj.get("company_description") or "")),
            primary_business_segments=_clean_segments(obj.get("primary_business_segments")),
            relevance_in_reference_segment=_clean_text(str(obj.get("relevance_in_reference_segment") or "")),
            competitor_type=competitor_type,
            company_website_url=company_url,
            brand_domain_whitelist=whitelist,
            brand_domain_customerlist=[],
            relevance_score=_clamp_score(obj.get("relevance_score")),
        )

        if not item.name:
            _log("empty company name dropped")
            warnings.append(f"v0.3 empty company name for query: {q}")
            continue
        if item.relevance_score < min_rel:
            _log(f"dropped below min relevance: {item.relevance_score:.3f}")
            continue

        # Strong dedupe by company name (domain variants like de.roborock.com collapse).
        nkey = _company_name_key(item.name)
        if not nkey:
            _log("empty normalized company key dropped")
            continue
        if nkey in by_name_index:
            existing_idx = by_name_index[nkey]
            existing = competitors[existing_idx]
            keep_new = False
            if item.relevance_score > existing.relevance_score + 1e-6:
                keep_new = True
            elif not existing.company_website_url and item.company_website_url:
                keep_new = True
            elif existing.company_website_url and _is_bad_company_domain(existing.company_website_url) and item.company_website_url and not _is_bad_company_domain(item.company_website_url):
                keep_new = True
            if keep_new:
                competitors[existing_idx] = item
                _log(f"duplicate replaced: {item.name} relevance={item.relevance_score:.3f}")
            else:
                _log(f"duplicate dropped: {item.name}")
            continue

        key = _company_key(item.name, item.company_website_url)
        by_name_index[nkey] = len(competitors)
        competitors.append(item)
        _log(f"accepted {item.name} ({item.competitor_type}) relevance={item.relevance_score:.3f}")

    # Map user-provided competitor links to detected competitors.
    unresolved_customer_links: List[str] = []
    if customer_links:
        for link in customer_links:
            mapped = False
            for i, c in enumerate(competitors):
                if _link_matches_competitor(link, c):
                    current = list(c.brand_domain_customerlist or [])
                    if link not in current:
                        current.append(link)
                    competitors[i] = c.model_copy(update={"brand_domain_customerlist": current})
                    mapped = True
            if not mapped:
                unresolved_customer_links.append(link)

    # If user provided competitor links that did not map, create competitors from those links.
    for link in unresolved_customer_links:
        if len(competitors) >= target_count:
            break
        _log(f"customer link unresolved -> seed competitor from {link}")
        link_results, source_label = _search_results(
            provider=p,
            query=link,
            per_query_results=max(5, per_query_results),
            openai_key=openai_key,
            openai_model=openai_model,
            perplexity_key=perplexity_key,
            perplexity_model=perplexity_model,
        )
        _log(f"customer_link_search_results={len(link_results)} source={source_label}")
        if not link_results:
            link_results = [{"title": "", "url": link, "snippet": ""}]
        obj = _llm_extract_company(
            provider=p,
            product_category=category,
            query=link,
            evidence_items=link_results,
        )
        raw_name = _clean_text(str((obj or {}).get("name") or ""))
        company_url = _normalize_url(str((obj or {}).get("company_website_url") or ""))
        if company_url and _is_bad_company_domain(company_url):
            company_url = ""
        normalized_name = _normalize_company_name(raw_name, company_url or link)
        if not normalized_name:
            normalized_name = _title_case_word(_domain_brand(link))
        if not normalized_name:
            warnings.append(f"v0.3 could not derive company from customer link: {link}")
            continue
        nkey = _company_name_key(normalized_name)
        if nkey in by_name_index:
            existing = competitors[by_name_index[nkey]]
            current = list(existing.brand_domain_customerlist or [])
            if link not in current:
                current.append(link)
                competitors[by_name_index[nkey]] = existing.model_copy(update={"brand_domain_customerlist": current})
            continue
        observed_domains = {_domain(link).lower()} if _domain(link) else set()
        whitelist = _normalize_brand_domain_whitelist(
            (obj or {}).get("brand_domain_whitelist"),
            company_url,
            normalized_name,
            observed_domains,
        )
        if link and _domain(link):
            lu = f"https://{_domain(link)}"
            if lu not in whitelist:
                whitelist.append(lu)
        competitor_type = _clean_text(str((obj or {}).get("competitor_type") or "Direct competitor"))
        if competitor_type not in _COMPETITOR_TYPES:
            competitor_type = "Direct competitor"
        item = CompanyCompetitorCandidate(
            name=normalized_name,
            cluster=_normalize_cluster(_clean_text(str((obj or {}).get("cluster") or "")), company_url, link),
            year_founded=_clamp_year((obj or {}).get("year_founded")),
            headquarters_country=_clean_text(str((obj or {}).get("headquarters_country") or "")),
            company_description=_clean_text(str((obj or {}).get("company_description") or "")),
            primary_business_segments=_clean_segments((obj or {}).get("primary_business_segments")),
            relevance_in_reference_segment=_clean_text(str((obj or {}).get("relevance_in_reference_segment") or "")),
            competitor_type=competitor_type,
            company_website_url=company_url,
            brand_domain_whitelist=whitelist[:12],
            brand_domain_customerlist=[link],
            relevance_score=max(min_rel, _clamp_score((obj or {}).get("relevance_score"))),
        )
        by_name_index[nkey] = len(competitors)
        competitors.append(item)
        warnings.append(f"v0.3 competitor added from customer link: {link}")

    if len(competitors) < min_comp:
        warnings.append(
            f"v0.3 only {len(competitors)} companies found, below min_competitors={min_comp}."
        )

    warnings.append("Generated via independent LLM-per-query company extraction v0.3.")
    warnings = list(dict.fromkeys([w for w in warnings if _clean_text(w)]))
    _log(f"done companies={len(competitors)} warnings={len(warnings)}")

    return CompetitorSearchResults(
        provider=p,
        generated_queries=queries,
        min_competitors_target=min_comp,
        competitors=competitors,
        extraction_warnings=warnings,
    )
