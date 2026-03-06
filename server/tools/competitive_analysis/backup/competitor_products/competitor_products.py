from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from server.tools.competitive_analysis.backup.competitor_identification import (
    _build_profile_text,
    _clean_text,
    _clean_url,
    _cosine_similarity,
    _domain,
    _langsearch_fallback,
    _load_json_obj,
    _openai_search,
    _perplexity_search,
)

from .models import (
    CompetitorProductsResults,
    CompetitorWithProducts,
    ReferenceProduct,
)


def _tokenize_text(s: str) -> List[str]:
    return [t for t in re.findall(r"[a-zA-Z0-9äöüÄÖÜß\-\+]{3,}", _clean_text(s).lower()) if t]


def _extract_feature_terms(profile: Dict[str, Any]) -> List[str]:
    terms: List[str] = []
    for f in (profile.get("normalized_features") or []):
        if isinstance(f, dict):
            n = _clean_text(str(f.get("name") or ""))
            if n:
                terms.append(n)
    for f in (profile.get("performance_parameters") or []):
        if isinstance(f, dict):
            n = _clean_text(str(f.get("name") or ""))
            if n:
                terms.append(n)
    for f in (profile.get("soft_features") or []):
        if isinstance(f, dict):
            n = _clean_text(str(f.get("name") or ""))
            if n:
                terms.append(n)
    for c in (profile.get("claims") or []):
        if isinstance(c, dict):
            t = _clean_text(str(c.get("text") or ""))
            if t:
                terms.append(t)

    # normalize + dedupe
    out: List[str] = []
    seen: set[str] = set()
    for x in terms:
        # keep concise phrase to improve literal matching
        phrase = " ".join(_tokenize_text(x)[:6]).strip()
        if not phrase:
            continue
        k = phrase.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(phrase)
    return out[:80]


def _feature_match_score(
    *,
    candidate_title: str,
    candidate_snippet: str,
    feature_terms: List[str],
) -> float:
    if not feature_terms:
        return 0.0
    hay = f"{_clean_text(candidate_title)} {_clean_text(candidate_snippet)}".lower()
    hits = 0
    for ft in feature_terms:
        ft_low = ft.lower()
        if ft_low in hay:
            hits += 1
            continue
        # relaxed token overlap for partially rephrased features
        toks = [t for t in _tokenize_text(ft_low) if len(t) >= 4]
        if not toks:
            continue
        hit_toks = sum(1 for t in toks if t in hay)
        if hit_toks >= max(1, min(2, len(toks))):
            hits += 1
    return max(0.0, min(1.0, hits / max(1, len(feature_terms))))


def _extract_numeric_features(profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for f in (profile.get("performance_parameters") or []):
        if not isinstance(f, dict):
            continue
        name = _clean_text(str(f.get("name") or ""))
        if not name:
            continue
        val = f.get("normalized_value")
        if val is None:
            val = f.get("value")
        try:
            num = float(val)
        except Exception:
            continue
        if abs(num) < 1e-12:
            continue
        out.append({"name": name, "value": num})
    return out


def _split_segments(text: str) -> List[str]:
    s = _clean_text(text)
    if not s:
        return []
    parts = re.split(r"[.;:|,\n]", s)
    return [p.strip() for p in parts if _clean_text(p)]


def _has_feature_name_similarity(feature_name: str, segment: str) -> bool:
    f_tokens = [t for t in _tokenize_text(feature_name) if len(t) >= 4]
    if not f_tokens:
        return False
    seg_low = _clean_text(segment).lower()
    hit = sum(1 for t in f_tokens if t in seg_low)
    return hit >= max(1, min(2, len(f_tokens)))


def _extract_numbers(text: str) -> List[float]:
    nums: List[float] = []
    for m in re.finditer(r"(?<!\w)(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?|\d+(?:[.,]\d+)?)(?!\w)", text):
        raw = m.group(1).strip()
        if raw.count(".") > 1 and "," not in raw:
            raw = raw.replace(".", "")
        elif raw.count(",") > 1 and "." not in raw:
            raw = raw.replace(",", "")
        else:
            if "," in raw and "." in raw:
                if raw.rfind(",") > raw.rfind("."):
                    raw = raw.replace(".", "").replace(",", ".")
                else:
                    raw = raw.replace(",", "")
            else:
                raw = raw.replace(",", ".")
        try:
            v = float(raw)
        except Exception:
            continue
        nums.append(v)
    return nums


def _numeric_closeness(ref_value: float, cand_value: float) -> float:
    denom = max(abs(ref_value), 1.0)
    rel_diff = abs(cand_value - ref_value) / denom
    return max(0.0, min(1.0, 1.0 - rel_diff))


def _extract_reference_price(profile: Dict[str, Any]) -> Optional[float]:
    prices = profile.get("price_indicators") or []
    if not isinstance(prices, list):
        return None
    vals: List[float] = []
    for p in prices:
        if not isinstance(p, dict):
            continue
        cur = _clean_text(str(p.get("currency") or "")).upper()
        if cur and cur not in {"EUR", "€"}:
            continue
        v = p.get("value")
        try:
            f = float(v)
        except Exception:
            continue
        if f > 0:
            vals.append(f)
    if not vals:
        return None
    # robust center: median
    vals.sort()
    n = len(vals)
    if n % 2 == 1:
        return vals[n // 2]
    return (vals[n // 2 - 1] + vals[n // 2]) / 2.0


def _extract_currency_numbers(text: str) -> List[float]:
    t = _clean_text(text)
    if not t:
        return []
    # Matches amounts near currency tokens/symbols.
    pattern = re.compile(
        r"(?:€|eur|euro)\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?)|(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?)\s*(?:€|eur|euro)",
        flags=re.IGNORECASE,
    )
    out: List[float] = []
    for m in pattern.finditer(t):
        raw = (m.group(1) or m.group(2) or "").strip()
        if not raw:
            continue
        if raw.count(".") > 1 and "," not in raw:
            raw = raw.replace(".", "")
        elif raw.count(",") > 1 and "." not in raw:
            raw = raw.replace(",", "")
        else:
            if "," in raw and "." in raw:
                if raw.rfind(",") > raw.rfind("."):
                    raw = raw.replace(".", "").replace(",", ".")
                else:
                    raw = raw.replace(",", "")
            else:
                raw = raw.replace(",", ".")
        try:
            v = float(raw)
        except Exception:
            continue
        if v > 0:
            out.append(v)
    return out


def _price_similarity(
    *,
    candidate_title: str,
    candidate_snippet: str,
    reference_price: Optional[float],
) -> float:
    if reference_price is None or reference_price <= 0:
        return 0.0
    vals = _extract_currency_numbers(f"{candidate_title} {candidate_snippet}")
    if not vals:
        return 0.0
    best = max(_numeric_closeness(reference_price, v) for v in vals)
    return max(0.0, min(1.0, best))


def _performance_similarity(
    *,
    candidate_title: str,
    candidate_snippet: str,
    numeric_features: List[Dict[str, Any]],
) -> float:
    if not numeric_features:
        return 0.0
    text = f"{_clean_text(candidate_title)}. {_clean_text(candidate_snippet)}"
    segments = _split_segments(text)
    if not segments:
        return 0.0

    feature_scores: List[float] = []
    for nf in numeric_features:
        name = _clean_text(str(nf.get("name") or ""))
        try:
            ref_value = float(nf.get("value"))
        except Exception:
            continue

        best_for_feature = -1.0
        for seg in segments:
            # Only evaluate numeric proximity if the segment is semantically close to feature name.
            if not _has_feature_name_similarity(name, seg):
                continue
            values = _extract_numbers(seg)
            if not values:
                continue
            local_best = max(_numeric_closeness(ref_value, v) for v in values)
            if local_best > best_for_feature:
                best_for_feature = local_best

        if best_for_feature >= 0.0:
            feature_scores.append(best_for_feature)

    if not feature_scores:
        return 0.0
    return max(0.0, min(1.0, sum(feature_scores) / len(feature_scores)))


def _search_results(
    *,
    provider: str,
    query: str,
    per_query_results: int,
    openai_key: str,
    openai_model: str,
    perplexity_key: str,
    perplexity_model: str,
) -> List[Dict[str, str]]:
    p = str(provider or "openai").strip().lower()
    if p == "openai" and openai_key:
        try:
            return _openai_search(query, per_query_results, api_key=openai_key, model=openai_model)
        except Exception:
            pass
    if p == "perplexity" and perplexity_key:
        try:
            return _perplexity_search(query, per_query_results, api_key=perplexity_key, model=perplexity_model)
        except Exception:
            pass
    try:
        return _langsearch_fallback(query, per_query_results)
    except Exception:
        return []


def _clean_product_name(title: str) -> str:
    t = _clean_text(title)
    if not t:
        return ""
    t = re.split(r"\s+[|\-–—]\s+", t, maxsplit=1)[0].strip()
    return t


def _build_company_queries(
    *,
    company_name: str,
    company_website_url: str,
    brand_domain_whitelist: List[str],
    category: str,
    max_queries_per_company: int,
    manufacturer_domain_only: bool,
) -> List[str]:
    q: List[str] = []
    cn = _clean_text(company_name)
    cat = _clean_text(category) or "product"
    cat_compact = _clean_text(re.sub(r"[-_/,&+]+", " ", cat))
    domains: List[str] = []
    seen_domains: set[str] = set()
    primary = _domain(company_website_url)
    if primary:
        domains.append(primary)
        seen_domains.add(primary.lower())
    for u in brand_domain_whitelist:
        d = _domain(_clean_url(str(u or "").strip()))
        if not d:
            continue
        k = d.lower()
        if k in seen_domains:
            continue
        seen_domains.add(k)
        domains.append(d)

    # Prioritize competitor-domain queries first.
    for d in domains:
        q.append(f"site:{d} {cat}")
    # Additional brand-oriented site query requested by user.
    if cn:
        q.append(f"site:{cn} {cat_compact or cat}")
    if manufacturer_domain_only:
        out: List[str] = []
        seen: set[str] = set()
        for x in q:
            s = _clean_text(x)
            if not s:
                continue
            k = s.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(s)
            if len(out) >= max_queries_per_company:
                break
        return out
    # Then broader queries.
    q.append(f"{cn} {cat} products")
    q.append(f"{cn} {cat} product page")
    q.append(f"{cn} {cat} models")
    q.append(f"{cn} {cat} specifications")

    out: List[str] = []
    seen: set[str] = set()
    for x in q:
        s = _clean_text(x)
        if not s:
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
        if len(out) >= max_queries_per_company:
            break
    return out


def _brand_tokens(company_name: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]{3,}", _clean_text(company_name).lower()) if t][:4]


def _contains_brand_token(text: str, token: str) -> bool:
    t = _clean_text(text).lower()
    if not t or not token:
        return False
    return re.search(rf"\b{re.escape(token)}\b", t) is not None


def _belongs_to_competitor_product(
    *,
    product_name: str,
    snippet: str,
    url: str,
    company_name: str,
    allowed_domains: set[str],
) -> bool:
    d = _domain(_clean_url(url)).lower()
    if d and d in allowed_domains:
        return True
    tokens = _brand_tokens(company_name)
    if not tokens:
        return False
    # Third-party result: keep only if the product name itself carries a brand token.
    # This avoids accepting unrelated products where only article context mentions the brand.
    if any(_contains_brand_token(product_name, tok) for tok in tokens):
        return True
    return False


def _candidate_score(
    *,
    candidate_title: str,
    candidate_snippet: str,
    candidate_url: str,
    profile_text: str,
    feature_terms: List[str],
    numeric_features: List[Dict[str, Any]],
    reference_price: Optional[float],
    company_domains: set[str],
    semantic_weight: float,
    feature_match_weight: float,
    performance_similarity_weight: float,
    price_weight: float,
) -> float:
    text = f"{_clean_text(candidate_title)}\n{_clean_text(candidate_snippet)}"
    sim = _cosine_similarity(profile_text, text)
    feat = _feature_match_score(
        candidate_title=candidate_title,
        candidate_snippet=candidate_snippet,
        feature_terms=feature_terms,
    )
    perf = _performance_similarity(
        candidate_title=candidate_title,
        candidate_snippet=candidate_snippet,
        numeric_features=numeric_features,
    )
    price_sim = _price_similarity(
        candidate_title=candidate_title,
        candidate_snippet=candidate_snippet,
        reference_price=reference_price,
    )
    # Blend semantic profile similarity with explicit feature overlap and numeric proximity.
    score = (
        semantic_weight * sim
        + feature_match_weight * feat
        + performance_similarity_weight * perf
        + price_weight * price_sim
    )

    u = _clean_url(candidate_url).lower()
    if company_domains and _domain(u).lower() in company_domains:
        score += 0.12
    if any(k in u for k in ("/product", "/products", "/produkt", "/model", "/models")):
        score += 0.07
    if any(k in u for k in ("datasheet", "datenblatt", ".pdf")):
        score += 0.04
    if any(k in u for k in ("/category", "/categories", "/search", "/blog", "/news")):
        score -= 0.08
    return max(0.0, min(1.0, score))


def extract_competitor_products(
    *,
    competitor_search_results: Optional[Dict[str, Any]],
    competitor_search_results_path: Optional[str],
    product_profile: Optional[Dict[str, Any]],
    product_profile_path: Optional[str],
    provider: str = "openai",
    per_query_results: int = 8,
    top_products_per_company: int = 3,
    max_queries_per_company: int = 5,
    semantic_weight: float = 0.35,
    feature_match_weight: float = 0.45,
    performance_similarity_weight: float = 0.20,
    price_weight: float = 0.0,
    emit_all_candidates: bool = False,
    manufacturer_domain_only: bool = False,
    verbose_terminal: bool = False,
    user_root,
    work_root,
) -> CompetitorProductsResults:
    def _log(msg: str) -> None:
        if verbose_terminal:
            print(f"[competitor_products] {msg}")

    csr = _load_json_obj(
        inline_obj=competitor_search_results,
        path=competitor_search_results_path,
        root_key="competitor_search_results",
        user_root=user_root,
        work_root=work_root,
    )
    profile = _load_json_obj(
        inline_obj=product_profile,
        path=product_profile_path,
        root_key="product_profile",
        user_root=user_root,
        work_root=work_root,
    )

    p = str(provider or "openai").strip().lower()
    if p not in {"openai", "perplexity", "ionos"}:
        p = "openai"

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    perplexity_key = os.getenv("PERPLEXITY_API_KEY", "").strip()
    perplexity_model = os.getenv("PERPLEXITY_MODEL", "sonar-pro").strip() or "sonar-pro"

    category = _clean_text(str(profile.get("product_category") or "")) or "product"
    profile_text = _build_profile_text(profile)
    feature_terms = _extract_feature_terms(profile)
    numeric_features = _extract_numeric_features(profile)
    reference_price = _extract_reference_price(profile)
    sw = max(0.0, float(semantic_weight))
    fw = max(0.0, float(feature_match_weight))
    pw = max(0.0, float(performance_similarity_weight))
    prw = max(0.0, float(price_weight))
    total_w = sw + fw + pw + prw
    if total_w <= 0:
        sw, fw, pw, prw = 0.35, 0.45, 0.20, 0.0
        total_w = 1.0
    # Normalize so request-body values can be any positive combination.
    sw, fw, pw, prw = sw / total_w, fw / total_w, pw / total_w, prw / total_w

    warnings = [str(w).strip() for w in (csr.get("extraction_warnings") or []) if str(w).strip()]
    generated_queries: List[str] = []
    out_competitors: List[CompetitorWithProducts] = []

    raw_competitors = csr.get("competitors") if isinstance(csr.get("competitors"), list) else []
    _log(f"start provider={p} competitors={len(raw_competitors)} top_products_per_company={top_products_per_company}")
    _log(f"feature_terms={len(feature_terms)}")
    _log(f"numeric_features={len(numeric_features)}")
    _log(f"reference_price={reference_price if reference_price is not None else 'n/a'}")
    _log(f"weights semantic={sw:.2f} feature_match={fw:.2f} performance={pw:.2f} price={prw:.2f}")
    _log(f"emit_all_candidates={bool(emit_all_candidates)}")
    _log(f"manufacturer_domain_only={bool(manufacturer_domain_only)}")

    for idx, c in enumerate(raw_competitors, start=1):
        if not isinstance(c, dict):
            continue
        company_name = _clean_text(str(c.get("name") or ""))
        company_url = _clean_url(str(c.get("company_website_url") or ""))
        raw_wl = c.get("brand_domain_whitelist") if isinstance(c.get("brand_domain_whitelist"), list) else []
        brand_domain_whitelist = [_clean_url(str(u or "").strip()) for u in raw_wl if _clean_url(str(u or "").strip())]

        _log(f"company {idx}/{len(raw_competitors)}: {company_name}")

        queries = _build_company_queries(
            company_name=company_name,
            company_website_url=company_url,
            brand_domain_whitelist=brand_domain_whitelist,
            category=category,
            max_queries_per_company=max_queries_per_company,
            manufacturer_domain_only=manufacturer_domain_only,
        )
        if manufacturer_domain_only and not queries:
            warnings.append(
                f"competitor_products: skipped '{company_name}' because manufacturer_domain_only=true and no company_website_url domain available."
            )
            out_competitors.append(
                CompetitorWithProducts(
                    name=company_name,
                    cluster=_clean_text(str(c.get("cluster") or "")),
                    year_founded=int(c.get("year_founded") or 0),
                    headquarters_country=_clean_text(str(c.get("headquarters_country") or "")),
                    company_description=_clean_text(str(c.get("company_description") or "")),
                    primary_business_segments=[
                        _clean_text(str(x))
                        for x in (c.get("primary_business_segments") or [])
                        if _clean_text(str(x))
                    ],
                    relevance_in_reference_segment=_clean_text(str(c.get("relevance_in_reference_segment") or "")),
                    competitor_type=_clean_text(str(c.get("competitor_type") or "")),
                    company_website_url=company_url,
                    brand_domain_whitelist=brand_domain_whitelist,
                    relevance_score=float(c.get("relevance_score") or 0.0),
                    reference_products=[],
                )
            )
            continue
        generated_queries.extend(queries)

        candidates: List[Tuple[float, Dict[str, str]]] = []
        seen_urls: set[str] = set()
        company_domains: set[str] = set()
        d0 = _domain(company_url)
        if d0:
            company_domains.add(d0.lower())
        for u in brand_domain_whitelist:
            d = _domain(u)
            if d:
                company_domains.add(d.lower())
        for q in queries:
            _log(f"search query={q}")
            results = _search_results(
                provider=p,
                query=q,
                per_query_results=per_query_results,
                openai_key=openai_key,
                openai_model=openai_model,
                perplexity_key=perplexity_key,
                perplexity_model=perplexity_model,
            )
            for r in results:
                url = _clean_url(str(r.get("url") or "").strip())
                if not url or url in seen_urls:
                    continue
                if manufacturer_domain_only and company_domains and _domain(url).lower() not in company_domains:
                    continue
                seen_urls.add(url)
                title = _clean_text(str(r.get("title") or ""))
                snippet = _clean_text(str(r.get("snippet") or ""))
                score = _candidate_score(
                    candidate_title=title,
                    candidate_snippet=snippet,
                    candidate_url=url,
                    profile_text=profile_text,
                    feature_terms=feature_terms,
                    numeric_features=numeric_features,
                    reference_price=reference_price,
                    company_domains=company_domains,
                    semantic_weight=sw,
                    feature_match_weight=fw,
                    performance_similarity_weight=pw,
                    price_weight=prw,
                )
                candidates.append((score, {"title": title, "snippet": snippet, "url": url}))

        candidates.sort(key=lambda x: x[0], reverse=True)
        reference_products: List[ReferenceProduct] = []
        seen_names: set[str] = set()
        for score, r in candidates:
            if not emit_all_candidates and len(reference_products) >= top_products_per_company:
                break
            pname = _clean_product_name(r.get("title") or "")
            if not pname:
                continue
            nk = pname.lower()
            if (not emit_all_candidates) and nk in seen_names:
                continue
            seen_names.add(nk)
            reference_products.append(
                ReferenceProduct(
                    product_name=pname,
                    category=category,
                    url=_clean_url(r.get("url") or ""),
                    snippet=_clean_text(r.get("snippet") or ""),
                    similarity_score=round(float(score), 4),
                )
            )

        # Final ownership guard: keep third-party URLs only if product still clearly belongs to the competitor.
        allowed_domains = company_domains
        if allowed_domains or company_name:
            before = len(reference_products)
            reference_products = [
                rp
                for rp in reference_products
                if _belongs_to_competitor_product(
                    product_name=rp.product_name,
                    snippet=rp.snippet,
                    url=rp.url,
                    company_name=company_name,
                    allowed_domains=allowed_domains,
                )
            ]
            removed = before - len(reference_products)
            if removed > 0:
                warnings.append(
                    f"competitor_products: removed {removed} products failing ownership check for company '{company_name}'."
                )
            if not emit_all_candidates and len(reference_products) > top_products_per_company:
                reference_products = reference_products[:top_products_per_company]

        if not reference_products:
            warnings.append(f"competitor_products: no reference products found for company '{company_name}'.")

        out_competitors.append(
            CompetitorWithProducts(
                name=company_name,
                cluster=_clean_text(str(c.get("cluster") or "")),
                year_founded=int(c.get("year_founded") or 0),
                headquarters_country=_clean_text(str(c.get("headquarters_country") or "")),
                company_description=_clean_text(str(c.get("company_description") or "")),
                primary_business_segments=[
                    _clean_text(str(x))
                    for x in (c.get("primary_business_segments") or [])
                    if _clean_text(str(x))
                ],
                relevance_in_reference_segment=_clean_text(str(c.get("relevance_in_reference_segment") or "")),
                competitor_type=_clean_text(str(c.get("competitor_type") or "")),
                company_website_url=company_url,
                brand_domain_whitelist=brand_domain_whitelist,
                relevance_score=float(c.get("relevance_score") or 0.0),
                reference_products=reference_products,
            )
        )

    # dedupe generated queries keep order
    unique_queries: List[str] = []
    seen_q: set[str] = set()
    for q in generated_queries:
        k = q.lower().strip()
        if not k or k in seen_q:
            continue
        seen_q.add(k)
        unique_queries.append(q)

    warnings.append("Generated competitor reference products via company-focused search and profile similarity scoring.")
    warnings = list(dict.fromkeys([w for w in warnings if _clean_text(w)]))
    _log(f"done competitors={len(out_competitors)} warnings={len(warnings)}")

    return CompetitorProductsResults(
        schema_version=str(csr.get("schema_version") or "1.0"),
        provider=p,
        generated_queries=unique_queries,
        min_competitors_target=int(csr.get("min_competitors_target") or 6),
        competitors=out_competitors,
        extraction_warnings=warnings,
    )
