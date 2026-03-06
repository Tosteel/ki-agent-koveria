from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

from server.services.llm_ionos import IonosLLM
from server.services.llm_openai import LlmOpenai
from server.services.llm_perplexity import LlmPerplexity
from server.tools.competitive_analysis.backup.competitor_identification.competitor_identification import (
    _clean_text,
    _clean_url,
    _cosine_similarity,
    _iter_queries,
    _langsearch_fallback,
    _load_json_obj,
    _openai_search,
    _perplexity_search,
)

from .models import (
    ClaimValue,
    CompetitorSearchResultsV04,
    FeatureValue,
    PriceIndicatorValue,
    ProductCompetitorCandidate,
    SoftFeatureValue,
)


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
_CLAIM_TYPES = {"value", "benefit", "differentiation"}


def _fetch_page_text(url: str, timeout_s: int, max_chars: int) -> str:
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            )
        }
        r = requests.get(url, headers=headers, timeout=timeout_s, allow_redirects=True)
        if r.status_code >= 400:
            return ""
        html = r.text or ""
        if not html:
            return ""
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]):
            tag.decompose()
        for sel in (
            "[role='banner']",
            "[role='navigation']",
            "[role='contentinfo']",
            ".header",
            ".site-header",
            ".main-header",
            ".footer",
            ".site-footer",
            ".main-footer",
            ".nav",
            ".navbar",
            ".breadcrumbs",
            ".cookie",
            ".cookie-banner",
            ".newsletter",
        ):
            for tag in soup.select(sel):
                tag.decompose()
        text = _clean_text(soup.get_text(separator=" "))
        if not text:
            return ""
        return text[: max(1000, int(max_chars))]
    except Exception:
        return ""


def _http_status_code(url: str, timeout_s: int) -> Optional[int]:
    try:
        r = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=timeout_s,
            allow_redirects=True,
            stream=True,
        )
        return int(r.status_code)
    except Exception:
        return None


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


def _llm_enrichment_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "performance_parameters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string"},
                        "value": {"type": ["number", "string", "null"]},
                        "unit": {"type": "string"},
                    },
                    "required": ["name", "value", "unit"],
                },
            },
            "price_indicators": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "raw": {"type": "string"},
                        "value": {"type": ["number", "null"]},
                        "currency": {"type": "string"},
                        "period": {"type": "string"},
                        "context": {"type": "string"},
                    },
                    "required": ["raw", "value", "currency", "period", "context"],
                },
            },
            "soft_features": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"name": {"type": "string"}, "available": {"type": "boolean"}},
                    "required": ["name", "available"],
                },
            },
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "text": {"type": "string"},
                        "claim_type": {"type": "string", "enum": ["value", "benefit", "differentiation"]},
                        "evidence": {"type": "string"},
                    },
                    "required": ["text", "claim_type", "evidence"],
                },
            },
        },
        "required": ["performance_parameters", "price_indicators", "soft_features", "claims"],
    }


def _llm_enrich_features(
    *,
    provider: str,
    category: str,
    product_name: str,
    manufacturer: str,
    url: str,
    text: str,
    perf: List[FeatureValue],
    price: List[PriceIndicatorValue],
    soft: List[SoftFeatureValue],
    claims: List[ClaimValue],
) -> Tuple[Dict[str, Any], str]:
    schema = _llm_enrichment_schema()
    payload = {
        "category": category,
        "product_name": product_name,
        "manufacturer": manufacturer,
        "url": url,
        "content": _clean_text(text)[:9000],
        "current": {
            "performance_parameters": [x.model_dump() for x in perf],
            "price_indicators": [x.model_dump() for x in price],
            "soft_features": [x.model_dump() for x in soft],
            "claims": [x.model_dump() for x in claims],
        },
    }
    system = (
        "You enrich competitor product features from provided text evidence. "
        "Only return valid JSON. Do not invent values without textual evidence."
    )
    user = (
        "Update/fill the existing features and optionally add new ones if clearly supported by evidence.\n"
        "Keep claim_type in [value, benefit, differentiation].\n"
        f"Payload:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    p = str(provider or "openai").strip().lower()
    try:
        if p == "openai":
            llm = LlmOpenai()
            if not llm.enabled():
                return {}, "openai_not_enabled"
            resp = llm._call(
                input_messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                text_format={"type": "json_schema", "name": "feature_enrichment", "schema": schema, "strict": False},
            )
            out = ""
            for item in resp.get("output", []):
                for c in item.get("content", []):
                    if c.get("type") == "output_text":
                        out += c.get("text", "")
            parsed = _parse_json_strictish(out)
            if not parsed:
                return {}, "openai_empty_or_invalid_json"
            return parsed, "ok"
        if p == "perplexity":
            llm = LlmPerplexity()
            if not llm.enabled():
                return {}, "perplexity_not_enabled"
            resp = llm._call(
                input_messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={"type": "json_schema", "json_schema": {"name": "feature_enrichment", "schema": schema}},
            )
            parsed = _parse_json_strictish(llm._extract_text(resp))
            if not parsed:
                return {}, "perplexity_empty_or_invalid_json"
            return parsed, "ok"
        if p == "ionos":
            llm = IonosLLM()
            if not llm.enabled():
                return {}, "ionos_not_enabled"
            resp = llm.chat_completions(
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={"type": "json_schema", "json_schema": {"name": "feature_enrichment", "schema": schema, "strict": False}},
            )
            parsed = _parse_json_strictish(IonosLLM.extract_text(resp))
            if not parsed:
                return {}, "ionos_empty_or_invalid_json"
            return parsed, "ok"
    except Exception as exc:
        return {}, f"exception:{exc.__class__.__name__}"
    return {}, "provider_not_supported"


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
    if d:
        return "official"
    return "unknown"


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


def _extract_number_unit(text: str) -> Tuple[Optional[float], str]:
    m = re.search(r"(\d+(?:[\.,]\d+)?)\s*(pa|°c|c|l|ml|mm|cm|m|w|wh|v|min|h|eur|€)", text.lower())
    if not m:
        return None, ""
    raw = m.group(1).replace(",", ".")
    try:
        v = float(raw)
    except Exception:
        v = None
    return v, m.group(2)


def _feature_name(item: Any) -> str:
    if isinstance(item, dict):
        return _clean_text(str(item.get("name") or item.get("feature") or item.get("term") or ""))
    return _clean_text(str(item or ""))


def _feature_unit(item: Any) -> str:
    if isinstance(item, dict):
        return _clean_text(str(item.get("unit") or ""))
    return ""


def _clone_perf_features(source: Any) -> List[FeatureValue]:
    out: List[FeatureValue] = []
    if not isinstance(source, list):
        return out
    for x in source:
        n = _feature_name(x)
        if not n:
            continue
        out.append(FeatureValue(name=n, value=None, unit=_feature_unit(x)))
    return out


def _build_price_indicators_template() -> List[PriceIndicatorValue]:
    return [
        PriceIndicatorValue(raw="", value=None, currency="", period="", context="Preis"),
        PriceIndicatorValue(raw="", value=None, currency="", period="", context="UVP"),
    ]


def _clone_soft_features(source: Any) -> List[SoftFeatureValue]:
    out: List[SoftFeatureValue] = []
    if not isinstance(source, list):
        return out
    for x in source:
        n = _feature_name(x)
        if not n:
            continue
        out.append(SoftFeatureValue(name=n, available=False))
    return out


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9äöüß]+", " ", (s or "").lower())).strip()


def _mentioned(feature_name: str, text: str) -> bool:
    fn = _norm(feature_name)
    tx = _norm(text)
    if not fn or not tx:
        return False
    if fn in tx:
        return True
    toks = [t for t in fn.split() if len(t) >= 3]
    if not toks:
        return False
    hit = sum(1 for t in toks if t in tx)
    return (hit / len(toks)) >= 0.6


def _fill_features(
    *,
    text: str,
    perf: List[FeatureValue],
    price: List[PriceIndicatorValue],
    soft: List[SoftFeatureValue],
) -> None:
    txt = _clean_text(text)
    for p in perf:
        if _mentioned(p.name, txt):
            v, u = _extract_number_unit(txt)
            if p.value is None and v is not None:
                p.value = v
            if not p.unit and u:
                p.unit = u
    lower = txt.lower()
    parsed_value: Optional[float] = None
    parsed_currency = ""
    m_price = re.search(r"(\d{2,6}(?:[.,]\d{1,2})?)\s*(€|eur)", lower, flags=re.IGNORECASE)
    if m_price:
        try:
            parsed_value = float(m_price.group(1).replace(",", "."))
            parsed_currency = "EUR"
        except Exception:
            parsed_value = None
            parsed_currency = ""
    for pr in price:
        ctx = (pr.context or "").lower()
        if ctx == "uvp":
            if "uvp" in lower or "listenpreis" in lower:
                pr.raw = txt
                pr.value = parsed_value
                pr.currency = parsed_currency
        else:
            if "preis" in lower or "angebot" in lower or parsed_value is not None:
                pr.raw = txt
                pr.value = parsed_value
                pr.currency = parsed_currency
    for s in soft:
        if _mentioned(s.name, txt):
            s.available = True



def _augment_detected_features(text: str, perf: List[FeatureValue], soft: List[SoftFeatureValue]) -> None:
    txt = _clean_text(text)
    matches = re.findall(r"(\d+(?:[\.,]\d+)?)\s*(pa|°c|l|ml|mm|cm|h|min)", txt.lower())
    existing = {_norm(x.name) for x in perf}
    for raw_v, unit in matches[:4]:
        name = f"Detected metric ({unit})"
        if _norm(name) in existing:
            continue
        try:
            v = float(raw_v.replace(",", "."))
        except Exception:
            continue
        perf.append(FeatureValue(name=name, value=v, unit=unit))
        existing.add(_norm(name))

    keyword_soft = {
        "navigation": "Navigation",
        "anti": "Anti-Haarverhedderung",
        "entleer": "Automatische Staubentleerung",
        "mopp": "Mopp-Trocknung",
    }
    existing_soft = {_norm(x.name): x for x in soft}
    lower = txt.lower()
    for kw, name in keyword_soft.items():
        if kw in lower:
            key = _norm(name)
            if key in existing_soft:
                existing_soft[key].available = True
            else:
                sf = SoftFeatureValue(name=name, available=True)
                soft.append(sf)
                existing_soft[key] = sf



def _claims_from_features(
    *,
    perf: List[FeatureValue],
    price: List[PriceIndicatorValue],
    soft: List[SoftFeatureValue],
    evidence_text: str,
) -> List[ClaimValue]:
    claims: List[ClaimValue] = []
    claim_types = ["value", "benefit", "differentiation"]

    scored: List[Tuple[str, bool]] = []
    for p in perf:
        scored.append((p.name, p.value is not None))
    for pr in price:
        scored.append((pr.context, pr.value is not None))
    for s in soft:
        scored.append((s.name, s.available))

    scored.sort(key=lambda x: (1 if x[1] else 0), reverse=True)
    evidence = _clean_text(evidence_text)[:240]
    for i, (name, _available) in enumerate(scored[:3]):
        claims.append(
            ClaimValue(
                text=name,
                claim_type=claim_types[i % len(claim_types)],
                evidence=evidence,
            )
        )

    while len(claims) < 3:
        i = len(claims)
        claims.append(
            ClaimValue(
                text=f"General claim {i + 1}",
                claim_type=claim_types[i % len(claim_types)],
                evidence=evidence,
            )
        )
    return claims



def _differentiators_from_snippet(snippet: str) -> List[str]:
    txt = _clean_text(snippet)
    if not txt:
        return []
    parts = [p.strip() for p in re.split(r"[.;]\s+", txt) if _clean_text(p)]
    return parts[:3]


def _merge_perf(existing: List[FeatureValue], incoming: Any) -> List[FeatureValue]:
    out: List[FeatureValue] = []
    seen: set[str] = set()
    base = existing if isinstance(existing, list) else []
    cand = incoming if isinstance(incoming, list) else []
    for src in [base, cand]:
        for x in src:
            if isinstance(x, FeatureValue):
                name = _clean_text(x.name)
                val = x.value
                unit = _clean_text(x.unit)
            elif isinstance(x, dict):
                name = _clean_text(str(x.get("name") or ""))
                val = x.get("value")
                unit = _clean_text(str(x.get("unit") or ""))
            else:
                continue
            if not name:
                continue
            k = _norm(name)
            if k in seen:
                continue
            seen.add(k)
            if isinstance(val, str):
                try:
                    val = float(val.replace(",", "."))
                except Exception:
                    pass
            out.append(FeatureValue(name=name, value=val, unit=unit))
    return out


def _merge_price(existing: List[PriceIndicatorValue], incoming: Any) -> List[PriceIndicatorValue]:
    if not isinstance(existing, list):
        existing = []
    ctx_map: Dict[str, PriceIndicatorValue] = {}
    for x in existing:
        if not isinstance(x, PriceIndicatorValue):
            continue
        key = _norm(x.context) or _norm(x.raw) or f"ctx_{len(ctx_map)}"
        ctx_map[key] = x
    if isinstance(incoming, list):
        for r in incoming:
            if not isinstance(r, dict):
                continue
            context = _clean_text(str(r.get("context") or "")) or "Preis"
            key = _norm(context)
            value = r.get("value")
            if isinstance(value, str):
                try:
                    value = float(value.replace(",", "."))
                except Exception:
                    value = None
            cand = PriceIndicatorValue(
                raw=_clean_text(str(r.get("raw") or "")),
                value=value if isinstance(value, (int, float)) else None,
                currency=_clean_text(str(r.get("currency") or "")),
                period=_clean_text(str(r.get("period") or "")),
                context=context,
            )
            if key in ctx_map:
                prev = ctx_map[key]
                if prev.value is None and cand.value is not None:
                    ctx_map[key] = cand
                elif not prev.raw and cand.raw:
                    ctx_map[key] = cand
            else:
                ctx_map[key] = cand
    out = list(ctx_map.values())
    return out if out else _build_price_indicators_template()


def _merge_soft(existing: List[SoftFeatureValue], incoming: Any) -> List[SoftFeatureValue]:
    out_map: Dict[str, SoftFeatureValue] = {}
    for x in existing:
        if not isinstance(x, SoftFeatureValue):
            continue
        k = _norm(x.name)
        if k:
            out_map[k] = x
    if isinstance(incoming, list):
        for r in incoming:
            if not isinstance(r, dict):
                continue
            name = _clean_text(str(r.get("name") or ""))
            if not name:
                continue
            k = _norm(name)
            avail = bool(r.get("available"))
            if k in out_map:
                out_map[k].available = out_map[k].available or avail
            else:
                out_map[k] = SoftFeatureValue(name=name, available=avail)
    return list(out_map.values())


def _merge_claims(existing: List[ClaimValue], incoming: Any) -> List[ClaimValue]:
    out: List[ClaimValue] = []
    if isinstance(incoming, list):
        for r in incoming:
            if not isinstance(r, dict):
                continue
            text = _clean_text(str(r.get("text") or ""))
            if not text:
                continue
            ctype = _clean_text(str(r.get("claim_type") or "value")).lower()
            if ctype not in _CLAIM_TYPES:
                ctype = "value"
            evidence = _clean_text(str(r.get("evidence") or ""))
            out.append(ClaimValue(text=text, claim_type=ctype, evidence=evidence[:240]))
            if len(out) >= 3:
                break
    if out:
        return out
    return existing[:3]


def _count_filled(
    perf: List[FeatureValue],
    price: List[PriceIndicatorValue],
    soft: List[SoftFeatureValue],
    claims: List[ClaimValue],
) -> Dict[str, int]:
    return {
        "performance_filled": sum(1 for x in perf if x.value is not None),
        "price_filled": sum(1 for x in price if x.value is not None),
        "soft_available": sum(1 for x in soft if x.available),
        "claims_count": len([c for c in claims if _clean_text(c.text)]),
    }



def _manufacturer_from_title_or_domain(title: str, url: str) -> str:
    t = _clean_text(title)
    if t:
        first = re.split(r"\s+", t, maxsplit=1)[0]
        first = _clean_text(first)
        if first and len(first) >= 2:
            return first
    return _company_from_domain(url)



def _reference_text(profile: Dict[str, Any]) -> str:
    chunks: List[str] = []
    for key in ("name", "product_name", "category", "description"):
        v = profile.get(key)
        if isinstance(v, str):
            chunks.append(_clean_text(v))
    for key in ("performance_parameters", "price_indicators", "soft_features"):
        vals = profile.get(key)
        if isinstance(vals, list):
            for x in vals:
                n = _feature_name(x)
                if n:
                    chunks.append(n)
    return "\n".join([c for c in chunks if c])



def search_competitors_v0_4(
    *,
    analysis_plan: Optional[Dict[str, Any]],
    analysis_plan_path: Optional[str],
    product_profile: Optional[Dict[str, Any]],
    product_profile_path: Optional[str],
    provider: str = "openai",
    max_queries: int = 20,
    per_query_results: int = 10,
    max_candidates_to_check: int = 200,
    use_llm_feature_enrichment: bool = False,
    llm_min_relevance_for_enrichment: float = 0.2,
    include_page_fetch: bool = False,
    page_fetch_timeout_s: int = 8,
    page_fetch_max_chars: int = 6000,
    verbose_terminal: bool = False,
    verbose_search_hits: bool = False,
    user_root=None,
    work_root=None,
) -> CompetitorSearchResultsV04:
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
    profile = _load_json_obj(
        inline_obj=product_profile,
        path=product_profile_path,
        root_key="product_profile",
        user_root=user_root,
        work_root=work_root,
    )

    warnings: List[str] = [str(w).strip() for w in (plan.get("extraction_warnings") or []) if str(w).strip()]

    p = str(provider or "openai").strip().lower()
    if p not in {"openai", "perplexity", "ionos"}:
        p = "openai"
    llm_min_rel = max(0.0, min(1.0, float(llm_min_relevance_for_enrichment)))

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    perplexity_key = os.getenv("PERPLEXITY_API_KEY", "").strip()
    perplexity_model = os.getenv("PERPLEXITY_MODEL", "sonar-pro").strip() or "sonar-pro"

    raw_queries = plan.get("search_queries") if isinstance(plan.get("search_queries"), list) else []
    queries = [_clean_text(str(q or "")) for q in raw_queries if _clean_text(str(q or ""))]
    if not queries:
        queries = _iter_queries(plan, max_queries=max_queries)
        warnings.append("v0.4 fallback to generated queries because analysis_plan.search_queries was empty.")
    queries = queries[: max(1, int(max_queries))]

    category = _clean_text(str(plan.get("product_category") or profile.get("category") or "product")) or "product"
    perf_template = _clone_perf_features(profile.get("performance_parameters"))
    price_template = _build_price_indicators_template()
    soft_template = _clone_soft_features(profile.get("soft_features"))
    ref_text = _reference_text(profile)

    _log(
        f"start provider={p} queries={len(queries)} per_query_results={per_query_results} "
        f"max_candidates_to_check={max_candidates_to_check}"
    )

    candidates: List[ProductCompetitorCandidate] = []
    seen_urls: set[str] = set()
    generated_queries: List[str] = []

    for i, q in enumerate(queries, start=1):
        _log(f"query {i}/{len(queries)}: {q}")
        generated_queries.append(q)
        results, source_label = _search_results(
            provider=p,
            query=q,
            per_query_results=per_query_results,
            openai_key=openai_key,
            openai_model=openai_model,
            perplexity_key=perplexity_key,
            perplexity_model=perplexity_model,
        )
        _log(f"search_results={len(results)} source={source_label}")

        for ridx, r in enumerate(results, start=1):
            if len(candidates) >= max_candidates_to_check:
                break
            title = _clean_text(str(r.get("title") or ""))
            url = _clean_url(str(r.get("url") or ""))
            snippet = _clean_text(str(r.get("snippet") or ""))
            if not title or not url:
                continue
            if url in seen_urls:
                continue
            status_code = _http_status_code(url, timeout_s=page_fetch_timeout_s)
            if status_code == 404:
                if verbose_terminal:
                    _log(f"skip 404 url={url}")
                continue
            seen_urls.add(url)

            if verbose_terminal and verbose_search_hits:
                _log(f"result {ridx}/{len(results)} title={title}")
                _log(f"result {ridx}/{len(results)} url={url}")
                _log(f"result {ridx}/{len(results)} snippet={snippet[:220]}")

            product_name = title
            manufacturer = _manufacturer_from_title_or_domain(title, url)
            url_type = _url_type(url)

            perf = [FeatureValue(name=x.name, value=x.value, unit=x.unit) for x in perf_template]
            price = [
                PriceIndicatorValue(
                    raw=x.raw,
                    value=x.value,
                    currency=x.currency,
                    period=x.period,
                    context=x.context,
                )
                for x in price_template
            ]
            soft = [SoftFeatureValue(name=x.name, available=x.available) for x in soft_template]

            page_text = ""
            if include_page_fetch:
                page_text = _fetch_page_text(url, timeout_s=page_fetch_timeout_s, max_chars=page_fetch_max_chars)
                if verbose_terminal:
                    _log(f"page_fetch={'ok' if page_text else 'skip'} url={url}")

            text = f"{title}\n{snippet}\n{page_text}"
            _fill_features(text=text, perf=perf, price=price, soft=soft)
            _augment_detected_features(text=text, perf=perf, soft=soft)

            claims = _claims_from_features(perf=perf, price=price, soft=soft, evidence_text=snippet or title)
            differentiators = _differentiators_from_snippet(snippet)
            before_counts = _count_filled(perf, price, soft, claims)
            after_counts = dict(before_counts)
            enrich_status = "not_requested"

            matched_fields = sum(1 for x in perf if x.value is not None) + sum(1 for x in price if x.value is not None) + sum(
                1 for x in soft if x.available
            )
            total_fields = max(1, len(perf) + len(price) + len(soft))
            feature_coverage = matched_fields / total_fields
            semantic = _cosine_similarity(ref_text, text) if ref_text else 0.0

            relevance_score = max(0.0, min(1.0, 0.55 * semantic + 0.45 * feature_coverage))
            similarity_score = max(0.0, min(1.0, 0.65 * semantic + 0.35 * feature_coverage))

            if use_llm_feature_enrichment and relevance_score >= llm_min_rel:
                enriched, enrich_status = _llm_enrich_features(
                    provider=p,
                    category=category,
                    product_name=product_name,
                    manufacturer=manufacturer,
                    url=url,
                    text=text,
                    perf=perf,
                    price=price,
                    soft=soft,
                    claims=claims,
                )
                if enriched:
                    perf = _merge_perf(perf, enriched.get("performance_parameters"))
                    price = _merge_price(price, enriched.get("price_indicators"))
                    soft = _merge_soft(soft, enriched.get("soft_features"))
                    claims = _merge_claims(claims, enriched.get("claims"))
                    after_counts = _count_filled(perf, price, soft, claims)
                    if verbose_terminal:
                        _log(f"llm_enrichment=applied url={url} before={before_counts} after={after_counts}")
                else:
                    after_counts = dict(before_counts)
                    if verbose_terminal:
                        _log(f"llm_enrichment=failed url={url} reason={enrich_status}")
                    warnings.append(f"v0.4 llm enrichment skipped/failed for url: {url} ({enrich_status})")
            elif use_llm_feature_enrichment and relevance_score < llm_min_rel:
                enrich_status = "skipped_relevance"
                if verbose_terminal:
                    _log(
                        f"llm_enrichment=skipped url={url} "
                        f"reason=relevance_below_threshold score={relevance_score:.4f} threshold={llm_min_rel:.4f}"
                    )

            candidates.append(
                ProductCompetitorCandidate(
                    product_name=product_name,
                    manufacturer=manufacturer,
                    url=url,
                    url_type=url_type,
                    performance_parameters=perf,
                    price_indicators=price,
                    soft_features=soft,
                    claims=claims,
                    differentiators=differentiators,
                    enrichment_delta={
                        "status": enrich_status,
                        "before": before_counts,
                        "after": after_counts,
                        "delta": {
                            "performance_filled": int(after_counts.get("performance_filled", 0))
                            - int(before_counts.get("performance_filled", 0)),
                            "price_filled": int(after_counts.get("price_filled", 0))
                            - int(before_counts.get("price_filled", 0)),
                            "soft_available": int(after_counts.get("soft_available", 0))
                            - int(before_counts.get("soft_available", 0)),
                            "claims_count": int(after_counts.get("claims_count", 0))
                            - int(before_counts.get("claims_count", 0)),
                        },
                    },
                    relevance_score=round(float(relevance_score), 4),
                    similarity_score=round(float(similarity_score), 4),
                )
            )
        if len(candidates) >= max_candidates_to_check:
            warnings.append(f"v0.4 candidate limit reached ({max_candidates_to_check}).")
            break

    warnings.append("Generated via hit-level competitor product extraction v0.4.")
    warnings = list(dict.fromkeys([w for w in warnings if _clean_text(w)]))
    _log(f"done competitors={len(candidates)} warnings={len(warnings)}")

    return CompetitorSearchResultsV04(
        schema_version="1.0",
        provider=p,
        generated_queries=generated_queries,
        competitors=candidates,
        extraction_warnings=warnings,
    )
