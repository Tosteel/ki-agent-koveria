from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from server.services.llm_ionos import IonosLLM
from server.services.llm_openai import LlmOpenai
from server.services.llm_perplexity import LlmPerplexity
from server.tools.competitive_analysis.backup.competitor_identification import (
    _build_main_anchor,
    _build_market_fit_text,
    _build_profile_text,
    _clean_text,
    _clean_url,
    _cluster_for_url,
    _domain,
    _is_low_trust_url,
    _is_same_brand_candidate,
    _is_self_or_variant_candidate,
    _is_url_usable_for_candidate,
    _iter_queries,
    _langsearch_fallback,
    _load_json_obj,
    _name_key,
    _name_url_consistent_strict,
    _openai_search,
    _perplexity_search,
    _score_candidate,
    _tokenize,
)
from .models import CompetitorCandidate, CompetitorList

_ALLOWED_SOURCE_TYPES = {"product_page", "series_page", "manufacturer_page", "comparison_page"}


def _token_set(text: str) -> set[str]:
    return {t for t in _tokenize(_clean_text(text).lower()) if len(t) >= 3}


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


def _llm_candidate_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "manufacturer": {"type": "string"},
            "product_name": {"type": "string"},
            "search_query": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["manufacturer", "product_name", "search_query", "reason"],
    }


def _call_llm_single_competitor(
    *,
    provider: str,
    main_anchor: str,
    category: str,
    dimensions: List[str],
    market_fit_text: str,
    excluded_names: List[str],
) -> Dict[str, Any]:
    system = (
        "Du identifizierst genau EIN Wettbewerbsprodukt. "
        "Antworte nur als JSON. "
        "Liefere ein konkretes Produkt (Modellbezeichnung) plus Hersteller. "
        "Keine Kategorien, keine Katalogseiten, keine Brand-only Antworten."
    )
    user = (
        f"Hauptprodukt: {main_anchor}\n"
        f"Kategorie: {category}\n"
        f"Vergleichsdimensionen: {', '.join(dimensions[:8])}\n"
        f"Marktfit-Kontext: {market_fit_text[:1000]}\n"
        f"Bereits gefundene Wettbewerber (nicht wiederholen): {', '.join(excluded_names[:30])}\n\n"
        "Gib genau ein Wettbewerbsprodukt zurück mit manufacturer, product_name und einer Suchanfrage, "
        "die auf Produktseite/Datenblatt zielt (keine Katalog-/Shopübersichtsseite)."
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
            "name": "single_competitor",
            "schema": _llm_candidate_schema(),
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
                    "name": "single_competitor",
                    "schema": _llm_candidate_schema(),
                    "strict": True,
                },
            },
        )
        return _parse_json_strictish(ion.extract_text(completion))
    except Exception:
        return {}


def _compose_full_name(manufacturer: str, product_name: str) -> str:
    mf = _clean_text(manufacturer)
    pn = _clean_text(str(product_name or "").replace("_", " "))
    if not pn:
        return ""
    if mf and mf.lower() not in pn.lower():
        return f"{mf} {pn}".strip()
    return pn


def _looks_like_product_or_datasheet(*, url: str, title: str, snippet: str) -> bool:
    u = _clean_url(url).lower()
    t = _clean_text(title).lower()
    s = _clean_text(snippet).lower()
    txt = f"{u} {t} {s}"
    positive = (
        "/product", "/products", "/produkt", "/produkte", "/model", "/models",
        "datasheet", "datenblatt", "technical data", "specification", "specs", ".pdf",
    )
    negative = (
        "/category", "/categories", "/catalog", "/katalog", "/collections", "/collection",
        "/search", "/blog", "/news", "/forum", "/tag/", "/brands", "/marken",
    )
    has_pos = any(k in txt for k in positive)
    has_neg = any(k in txt for k in negative)
    if has_neg and not ("datasheet" in txt or "datenblatt" in txt or ".pdf" in txt):
        return False
    return has_pos


def _is_pdf_or_datasheet(url: str, title: str, snippet: str) -> bool:
    txt = f"{_clean_url(url).lower()} {_clean_text(title).lower()} {_clean_text(snippet).lower()}"
    return ".pdf" in txt or "datasheet" in txt or "datenblatt" in txt or "technical data" in txt


def _is_strong_product_page(url: str, title: str, snippet: str) -> bool:
    txt = f"{_clean_url(url).lower()} {_clean_text(title).lower()} {_clean_text(snippet).lower()}"
    positive = (
        "/product/", "/products/", "/produkt/", "/produkte/", "/model/", "/models/",
        "product page", "specifications", "specification", "technical specs", "produktseite",
    )
    negative = (
        "/category", "/categories", "/catalog", "/katalog", "/collections", "/collection",
        "/search", "/blog", "/news", "/forum", "/tag/", "/brands", "/marken",
        "/compare", "/vergleich",
    )
    if any(k in txt for k in negative):
        return False
    return any(k in txt for k in positive)


def _manufacturer_domain_match(manufacturer: str, url: str) -> bool:
    m_tokens = _token_set(manufacturer)
    if not m_tokens:
        return False
    d = _domain(url).lower()
    return any(tok in d for tok in m_tokens)


def _model_overlap_score(name: str, url: str, title: str, snippet: str) -> float:
    src = f"{_clean_url(url)} {_clean_text(title)} {_clean_text(snippet)}".lower()
    src_tokens = _token_set(src)
    name_tokens = _token_set(name)
    if not name_tokens or not src_tokens:
        return 0.0
    overlap = len(name_tokens & src_tokens)
    return overlap / max(1, len(name_tokens))


def _url_priority_score(
    *,
    candidate_name: str,
    manufacturer: str,
    url: str,
    title: str,
    snippet: str,
    url_priority_weight: float = 1.0,
    datasheet_priority_weight: float = 0.6,
) -> float:
    score = 0.0
    uw = max(0.0, float(url_priority_weight))
    dw = max(0.0, float(datasheet_priority_weight))
    if _manufacturer_domain_match(manufacturer, url):
        score += 45.0
    if _is_pdf_or_datasheet(url, title, snippet):
        score += 20.0 * dw
    if _is_strong_product_page(url, title, snippet):
        score += 40.0 * uw
    elif _looks_like_product_or_datasheet(url=url, title=title, snippet=snippet):
        score += 15.0 * max(0.5, uw)

    if _name_url_consistent_strict(candidate_name, url, title, snippet):
        score += 10.0

    overlap = _model_overlap_score(candidate_name, url, title, snippet)
    score += min(20.0, overlap * 40.0)

    u = _clean_url(url).lower()
    if any(k in u for k in ("/category", "/categories", "/collection", "/collections", "/search", "/compare", "/vergleich")):
        score -= 35.0

    return score


def _source_priority_rank(
    *,
    manufacturer: str,
    candidate_name: str,
    result: Dict[str, str],
) -> int:
    """
    Hard source ordering for final URL choice:
    5 official product page
    4 official datasheet/pdf
    3 non-official strong product page
    2 non-official datasheet/pdf
    1 other product-like page
    0 everything else
    """
    url = _clean_url(str(result.get("url") or "").strip())
    title = _clean_text(str(result.get("title") or ""))
    snippet = _clean_text(str(result.get("snippet") or ""))
    if not url:
        return 0

    official = _manufacturer_domain_match(manufacturer, url)
    strong_product = _is_strong_product_page(url, title, snippet)
    is_pdf = _is_pdf_or_datasheet(url, title, snippet)
    product_like = _looks_like_product_or_datasheet(url=url, title=title, snippet=snippet)

    if official and strong_product and not is_pdf:
        return 5
    if official and is_pdf:
        return 4
    if strong_product and not is_pdf:
        return 3
    if is_pdf:
        return 2
    if product_like:
        return 1
    return 0


def _normalize_candidate_label(raw: str) -> str:
    t = _clean_text(raw)
    if not t:
        return ""
    # Strip common page separators.
    t = re.split(r"\s+[|\-–—]\s+", t, maxsplit=1)[0].strip()
    t = re.sub(r"\s+", " ", t).strip(" -_,.;:")
    return t


def _split_manufacturer_product(raw_name: str, url: str) -> Tuple[str, str]:
    full = _normalize_candidate_label(raw_name)
    if not full:
        return "", ""
    toks = full.split()
    if len(toks) == 1:
        return toks[0], full

    split_idx = 1
    legal_or_org_tokens = {
        "robotics",
        "technology",
        "technologies",
        "corp",
        "corporation",
        "inc",
        "gmbh",
        "ag",
        "co",
        "ltd",
        "llc",
    }
    if len(toks) >= 2 and toks[1].lower() in legal_or_org_tokens:
        split_idx = 2
    for i, tk in enumerate(toks[1:], start=1):
        if re.search(r"\d", tk):
            split_idx = i
            break
    manufacturer = " ".join(toks[:split_idx]).strip()
    product = " ".join(toks[split_idx:]).strip()
    if not product:
        product = full

    # Lightweight domain hint fallback.
    d = _domain(url).split(".")[0].replace("-", " ").strip()
    if not manufacturer and d:
        manufacturer = d
    if not manufacturer:
        manufacturer = toks[0]
    return manufacturer, product


def _strip_manufacturer_prefix(product_name: str, manufacturer: str) -> str:
    p = _clean_text(product_name)
    m = _clean_text(manufacturer)
    if not p:
        return ""
    if not m:
        return p
    p_tokens = p.split()
    m_tokens = m.split()
    if len(p_tokens) >= len(m_tokens) and [t.lower() for t in p_tokens[: len(m_tokens)]] == [t.lower() for t in m_tokens]:
        stripped = " ".join(p_tokens[len(m_tokens) :]).strip()
        return stripped or p
    # Handle repeated brand/org prefixes like:
    # "Ecovacs Robotics ECOVACS Deebot X1 Omni"
    # with manufacturer "Ecovacs Robotics"
    work = p_tokens[:]
    m_low = [t.lower() for t in m_tokens]
    while len(work) >= len(m_low) and [t.lower() for t in work[: len(m_low)]] == m_low:
        work = work[len(m_low) :]
    while work and work[0].lower() in {x.lower() for x in m_tokens}:
        work = work[1:]
    stripped2 = " ".join(work).strip()
    if stripped2:
        return stripped2
    return p


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


def identify_competitors_v0_2(
    *,
    analysis_plan: Optional[Dict[str, Any]],
    analysis_plan_path: Optional[str],
    product_profile: Optional[Dict[str, Any]],
    product_profile_path: Optional[str],
    provider: str = "openai",
    max_queries: int = 12,
    per_query_results: int = 6,
    shortlist_size: int = 10,
    min_relevance_score: float = 0.20,
    min_similarity_score: float = 0.12,
    url_priority_weight: float = 1.0,
    datasheet_priority_weight: float = 0.6,
    exclude_below_threshold: bool = True,
    exhaust_all_attempts: bool = False,
    verbose_terminal: bool = False,
    user_root,
    work_root,
) -> CompetitorList:
    def _log(msg: str) -> None:
        if not verbose_terminal:
            return
        print(f"[competitor_identification_v0_2] {msg}")

    plan = _load_json_obj(
        inline_obj=analysis_plan,
        path=analysis_plan_path,
        root_key="analysis_plan",
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

    warnings = [
        *[str(w).strip() for w in (plan.get("extraction_warnings") or []) if str(w).strip()],
        *[str(w).strip() for w in (profile.get("extraction_warnings") or []) if str(w).strip()],
    ]

    p = str(provider or "openai").strip().lower()
    if p not in {"openai", "ionos", "perplexity"}:
        p = "openai"

    openai_key = __import__("os").getenv("OPENAI_API_KEY", "").strip()
    openai_model = __import__("os").getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    perplexity_key = __import__("os").getenv("PERPLEXITY_API_KEY", "").strip()
    perplexity_model = __import__("os").getenv("PERPLEXITY_MODEL", "sonar-pro").strip() or "sonar-pro"

    metadata = profile.get("metadata") if isinstance(profile.get("metadata"), dict) else {}
    product_name = str(metadata.get("product_name") or "").strip()
    manufacturer = str(metadata.get("manufacturer") or "").strip()
    main_anchor = _build_main_anchor(product_name, manufacturer)

    dimensions = [
        str(d.get("name") or "").strip()
        for d in (plan.get("comparison_dimensions") or [])
        if isinstance(d, dict) and str(d.get("name") or "").strip()
    ]
    profile_text = _build_profile_text(profile)
    market_fit_text = _build_market_fit_text(profile)

    raw_plan_queries = plan.get("search_queries") if isinstance(plan.get("search_queries"), list) else []
    base_queries = [str(q).strip() for q in raw_plan_queries if str(q).strip()]
    if not base_queries:
        base_queries = _iter_queries(plan, max_queries=max_queries)
        warnings.append("v0.2 fallback to generated queries because analysis_plan.search_queries was empty.")
    # Keep strict 1:1 order/content from analysis_plan; only cap by max_queries for runtime control.
    if max_queries and len(base_queries) > max_queries:
        base_queries = base_queries[: max(1, int(max_queries))]
    min_comp = int(plan.get("min_competitors") or 5)
    min_comp = max(2, min(50, min_comp))
    target_count = max(min_comp, min(shortlist_size, 50))
    min_rel = max(0.0, min(1.0, float(min_relevance_score)))
    min_sim = max(0.0, min(1.0, float(min_similarity_score)))
    upw = max(0.0, float(url_priority_weight))
    dpw = max(0.0, float(datasheet_priority_weight))
    product_type = _clean_text(str(plan.get("product_category") or "")) or "Produkt"
    _log(
        f"start provider={p} target_count={target_count} min_comp={min_comp} "
        f"base_queries={len(base_queries)} min_rel={min_rel:.2f} min_sim={min_sim:.2f} "
        f"url_priority_weight={upw:.2f} datasheet_priority_weight={dpw:.2f} "
        f"exclude_below_threshold={exclude_below_threshold} "
        f"exhaust_all_attempts={exhaust_all_attempts}"
    )

    competitors: List[CompetitorCandidate] = []
    seen_keys: set[str] = set()
    excluded_names: List[str] = []

    # Step 1: Build candidate seeds from search_queries (not from LLM memory).
    seed_pool: Dict[str, Dict[str, Any]] = {}
    for q in base_queries:
        _log(f"seed query={q}")
        results, source_label = _search_results(
            provider=p,
            query=q,
            per_query_results=max(4, per_query_results),
            openai_key=openai_key,
            openai_model=openai_model,
            perplexity_key=perplexity_key,
            perplexity_model=perplexity_model,
        )
        for r in results:
            u = _clean_url(str(r.get("url") or "").strip())
            t = _clean_text(str(r.get("title") or ""))
            s = _clean_text(str(r.get("snippet") or ""))
            source_type = _clean_text(str(r.get("source_type") or "")).lower()
            if not u:
                continue
            strong_product_seed = _is_strong_product_page(u, t, s)
            looks_like_seed = _looks_like_product_or_datasheet(url=u, title=t, snippet=s)
            # Loosen early seed gate: keep strong product pages even from mixed/trust-limited domains.
            if _is_low_trust_url(u) and not strong_product_seed:
                continue
            if not (looks_like_seed or strong_product_seed):
                continue

            raw_label = _clean_text(str(r.get("name") or t))
            m_seed, p_seed = _split_manufacturer_product(raw_label, u)
            if not p_seed:
                continue
            combined = _compose_full_name(m_seed, p_seed)
            if not combined:
                continue
            if manufacturer and _is_same_brand_candidate(candidate_name=combined, candidate_url=u, manufacturer=manufacturer):
                continue
            if _is_self_or_variant_candidate(
                candidate_name=combined,
                candidate_url=u,
                candidate_snippet=s,
                product_name=product_name,
                manufacturer=manufacturer,
            ):
                continue

            skey = f"{_name_key(combined)}|{_domain(u)}"
            seed_score = _url_priority_score(
                candidate_name=combined,
                manufacturer=m_seed,
                url=u,
                title=t,
                snippet=s,
                url_priority_weight=upw,
                datasheet_priority_weight=dpw,
            )
            prev = seed_pool.get(skey)
            if prev is None or seed_score > float(prev.get("score") or 0.0):
                seed_pool[skey] = {
                    "manufacturer": m_seed,
                    "name": p_seed,
                    "combined_name": combined,
                    "seed_query": q,
                    "seed_result": r,
                    "score": seed_score,
                    "source_label": source_label,
                }

    seeds = sorted(seed_pool.values(), key=lambda x: float(x.get("score") or 0.0), reverse=True)
    if not seeds:
        warnings.append("v0.2 candidate seed generation from search_queries returned no candidates.")
    _log(f"seed candidates={len(seeds)}")

    # Step 2: Intensive per-product URL qualification.
    max_attempts = max(target_count * 7, 20)
    attempt_limit = min(len(seeds), max_attempts)
    for attempt, seed in enumerate(seeds[:attempt_limit], start=1):
        if not exhaust_all_attempts and len(competitors) >= target_count:
            break
        _log(f"attempt={attempt}/{attempt_limit} found={len(competitors)}/{target_count}")

        m = _clean_text(seed.get("manufacturer"))
        pn = _clean_text(seed.get("name"))
        combined_name = _clean_text(seed.get("combined_name"))
        cand_name = pn or combined_name
        if not cand_name:
            continue
        _log(f"candidate={combined_name or cand_name}")

        search_query = _clean_text(seed.get("seed_query")) or f'"{combined_name}" product specifications'
        seed_result = seed.get("seed_result") if isinstance(seed.get("seed_result"), dict) else {}
        seed_url = _clean_url(str(seed_result.get("url") or "").strip())
        query_list = [search_query, f'"{combined_name}" datasheet', f'"{combined_name}" product specifications']
        query_list.extend(base_queries[:2])

        ranked_candidates: List[Tuple[float, Dict[str, str], str, str]] = []
        visited_urls: List[Tuple[str, str]] = []

        if seed_url:
            seed_title = _clean_text(str(seed_result.get("title") or ""))
            seed_snip = _clean_text(str(seed_result.get("snippet") or ""))
            visited_urls.append((seed_url, str(seed.get("source_label") or "seed")))
            ranked_candidates.append(
                (
                    _url_priority_score(
                        candidate_name=combined_name or cand_name,
                        manufacturer=m,
                        url=seed_url,
                        title=seed_title,
                        snippet=seed_snip,
                        url_priority_weight=upw,
                        datasheet_priority_weight=dpw,
                    ),
                    seed_result,
                    search_query,
                    str(seed.get("source_label") or "seed"),
                )
            )

        for q in query_list:
            _log(f"search query={q}")
            results, source_label = _search_results(
                provider=p,
                query=q,
                per_query_results=max(4, per_query_results),
                openai_key=openai_key,
                openai_model=openai_model,
                perplexity_key=perplexity_key,
                perplexity_model=perplexity_model,
            )
            for r in results:
                u = _clean_url(str(r.get("url") or "").strip())
                t = _clean_text(str(r.get("title") or ""))
                s = _clean_text(str(r.get("snippet") or ""))
                if u:
                    visited_urls.append((u, source_label))
                if not u or _is_low_trust_url(u):
                    continue
                if not _is_url_usable_for_candidate(u, combined_name or cand_name):
                    continue
                url_score = _url_priority_score(
                    candidate_name=combined_name or cand_name,
                    manufacturer=m,
                    url=u,
                    title=t,
                    snippet=s,
                    url_priority_weight=upw,
                    datasheet_priority_weight=dpw,
                )
                overlap = _model_overlap_score(combined_name or cand_name, u, t, s)
                consistent = _name_url_consistent_strict(combined_name or cand_name, u, t, s)
                official = _manufacturer_domain_match(m, u)
                good_page = _is_pdf_or_datasheet(u, t, s) or _is_strong_product_page(u, t, s)

                if not official and not consistent and overlap < 0.25:
                    continue
                if not good_page and url_score < 18:
                    continue

                ranked_candidates.append((url_score, r, q, source_label))
                _log(
                    f"candidate_url accepted score={url_score:.2f} "
                    f"official={official} consistent={consistent} overlap={overlap:.2f} url={u}"
                )

            if ranked_candidates and max(sc for sc, _r, _q, _src in ranked_candidates) >= 55:
                break

        if not ranked_candidates:
            warnings.append(f"v0.2 no product/datasheet URL found for candidate: {combined_name or cand_name}")
            excluded_names.append(combined_name or cand_name)
            _log(f"no_valid_url candidate={combined_name or cand_name}")
            continue

        ranked_candidates.sort(
            key=lambda x: (
                _source_priority_rank(
                    manufacturer=m,
                    candidate_name=combined_name or cand_name,
                    result=x[1],
                ),
                x[0],
            ),
            reverse=True,
        )
        _best_score, r, used_query, source_label = ranked_candidates[0]
        title = _clean_text(str(r.get("title") or ""))
        snippet = _clean_text(str(r.get("snippet") or ""))
        url = _clean_url(str(r.get("url") or "").strip())
        source_type = _clean_text(str(r.get("source_type") or "")).lower()

        scored_base = _score_candidate(
            profile_text=profile_text,
            market_fit_text=market_fit_text,
            title=title or (combined_name or cand_name),
            snippet=snippet,
            url=url,
            source_query=used_query,
            source_type=source_type,
            dimensions=dimensions,
        )
        # _score_candidate comes from v0.1 and returns v0.1 model type.
        # Re-wrap into the v0.2 schema model (with `category` and `manufacturer`).
        base_payload: Dict[str, Any]
        if hasattr(scored_base, "model_dump"):
            base_payload = dict(scored_base.model_dump())
        elif isinstance(scored_base, dict):
            base_payload = dict(scored_base)
        else:
            base_payload = {}
        scored = CompetitorCandidate(**base_payload)
        scored.category = product_type
        out_manufacturer = _clean_text(m)
        out_name = _strip_manufacturer_prefix(cand_name, out_manufacturer)
        # Fallback if split was weak and manufacturer empty.
        if not out_manufacturer:
            guessed_m, guessed_p = _split_manufacturer_product(cand_name, url)
            out_manufacturer = _clean_text(guessed_m)
            out_name = _strip_manufacturer_prefix(guessed_p or cand_name, out_manufacturer)

        scored.manufacturer = out_manufacturer
        scored.name = out_name or cand_name
        scored.url = url
        scored.cluster = _cluster_for_url(url, title, snippet)
        scored.source_query = used_query
        scored.source_type = source_type if source_type in _ALLOWED_SOURCE_TYPES else "unknown"

        dedup_seen: set[str] = set()
        merged_candidates: List[str] = []
        if url:
            merged_candidates.append(url)
            dedup_seen.add(url)
        for u, _src in visited_urls:
            if not u or u in dedup_seen:
                continue
            merged_candidates.append(u)
            dedup_seen.add(u)
            if len(merged_candidates) >= 10:
                break
        scored.url_candidates = merged_candidates
        prov: Dict[str, str] = {}
        for u, src in visited_urls:
            if not u or u in prov:
                continue
            prov[u] = src
            if len(prov) >= 10:
                break
        if url and url not in prov:
            prov[url] = source_label
        scored.url_provenance = prov
        scored.reasons = list(
            dict.fromkeys(
                [
                    *(scored.reasons or []),
                    "v0.2 query-seeded candidate discovery",
                    "v0.2 source-priority ranking (official product page/datasheet preferred)",
                ]
            )
        )

        if exclude_below_threshold and (scored.relevance_score < min_rel or scored.similarity_score < min_sim):
            excluded_names.append(f"{scored.manufacturer} {scored.name}".strip())
            _log(
                f"dropped thresholds manufacturer={scored.manufacturer} name={scored.name} "
                f"relevance={scored.relevance_score:.4f}<{min_rel:.2f} "
                f"similarity={scored.similarity_score:.4f}<{min_sim:.2f}"
            )
            continue

        key = f"{_name_key(scored.manufacturer)}|{_name_key(scored.name)}|{_domain(scored.url)}"
        if key in seen_keys:
            excluded_names.append(f"{scored.manufacturer} {scored.name}".strip())
            _log(f"duplicate dropped={scored.manufacturer} {scored.name} domain={_domain(scored.url)}")
            continue
        seen_keys.add(key)
        competitors.append(scored)
        excluded_names.append(f"{scored.manufacturer} {scored.name}".strip())
        _log(f"accepted manufacturer={scored.manufacturer} name={scored.name} score={_best_score:.2f} url={scored.url}")

    competitors = sorted(competitors, key=lambda c: (c.relevance_score, c.similarity_score), reverse=True)[:target_count]

    if len(competitors) < min_comp and exclude_below_threshold:
        warnings.append(
            f"v0.2 only {len(competitors)} competitors found, below target min_competitors={min_comp} "
            f"after applying thresholds relevance>={min_rel:.2f}, similarity>={min_sim:.2f}."
        )

    warnings.append("Generated via iterative LLM single-candidate search v0.2.")
    warnings = list(dict.fromkeys([w.strip() for w in warnings if str(w).strip()]))
    _log(f"done competitors={len(competitors)} warnings={len(warnings)}")

    return CompetitorList(
        provider=p,
        generated_queries=base_queries,
        min_competitors_target=min_comp,
        competitors=competitors,
        extraction_warnings=warnings,
    )
