from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from server.services.llm_brave import LlmBrave
from server.tools.competitive_analysis.backup.competitor_identification import (
    _clean_text,
    _load_json_obj,
)

from .models import (
    ClaimValue,
    CompetitorEnrichedV05,
    CompetitorProfileExtractionResultsV05,
    FeatureValue,
    PriceIndicatorValue,
    SoftFeatureValue,
)

_CLAIM_TYPES = {"value", "benefit", "differentiation"}


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


def _feature_name(item: Any) -> str:
    if isinstance(item, dict):
        return _clean_text(str(item.get("name") or item.get("feature") or item.get("term") or ""))
    return _clean_text(str(item or ""))


def _feature_unit(item: Any) -> str:
    if isinstance(item, dict):
        return _clean_text(str(item.get("unit") or ""))
    return ""


def _template_perf(profile: Dict[str, Any]) -> List[FeatureValue]:
    vals = profile.get("performance_parameters")
    if not isinstance(vals, list):
        return []
    out: List[FeatureValue] = []
    for x in vals:
        n = _feature_name(x)
        if not n:
            continue
        out.append(FeatureValue(name=n, value=None, unit=_feature_unit(x)))
    return out


def _template_price(profile: Dict[str, Any]) -> List[PriceIndicatorValue]:
    vals = profile.get("price_indicators")
    out: List[PriceIndicatorValue] = []
    if isinstance(vals, list) and vals:
        for x in vals:
            if not isinstance(x, dict):
                continue
            out.append(
                PriceIndicatorValue(
                    raw="",
                    value=None,
                    currency=_clean_text(str(x.get("currency") or "")),
                    period=_clean_text(str(x.get("period") or "")),
                    context=_clean_text(str(x.get("context") or "")) or "Preis",
                )
            )
    if not out:
        out = [
            PriceIndicatorValue(raw="", value=None, currency="", period="", context="Preis"),
            PriceIndicatorValue(raw="", value=None, currency="", period="", context="UVP"),
        ]
    return out


def _template_soft(profile: Dict[str, Any]) -> List[SoftFeatureValue]:
    vals = profile.get("soft_features")
    if not isinstance(vals, list):
        return []
    out: List[SoftFeatureValue] = []
    for x in vals:
        n = _feature_name(x)
        if not n:
            continue
        out.append(SoftFeatureValue(name=n, available=False))
    return out


def _template_claims(profile: Dict[str, Any]) -> List[ClaimValue]:
    vals = profile.get("claims")
    out: List[ClaimValue] = []
    if isinstance(vals, list):
        for x in vals[:3]:
            if not isinstance(x, dict):
                continue
            ctype = _clean_text(str(x.get("claim_type") or "value")).lower()
            if ctype not in _CLAIM_TYPES:
                ctype = "value"
            out.append(ClaimValue(text="", claim_type=ctype, evidence=""))
    claim_types = ["value", "benefit", "differentiation"]
    while len(out) < 3:
        out.append(ClaimValue(text="", claim_type=claim_types[len(out)], evidence=""))
    return out


def _template_differentiators(profile: Dict[str, Any]) -> List[str]:
    # No fixed differentiator templates: keep empty and fill only from evidence.
    return []


def _append_term(bucket: List[str], value: Any) -> None:
    s = _clean_text(str(value or ""))
    if s:
        bucket.append(s)


def _collect_query_terms(
    candidate: Dict[str, Any],
    perf_template: List[FeatureValue],
    price_template: List[PriceIndicatorValue],
    soft_template: List[SoftFeatureValue],
    claims_template: List[ClaimValue],
    differentiators_template: List[str],
) -> Dict[str, List[str]]:
    buckets: Dict[str, List[str]] = {
        "performance_parameters": [],
        "price_indicators": [],
        "soft_features": [],
        "claims": [],
        "differentiators": [],
    }

    for x in perf_template:
        _append_term(buckets["performance_parameters"], x.name)
    for x in price_template:
        _append_term(buckets["price_indicators"], x.context)
    for x in soft_template:
        _append_term(buckets["soft_features"], x.name)
    for x in claims_template:
        _append_term(buckets["claims"], x.text)
        _append_term(buckets["claims"], x.evidence)
        _append_term(buckets["claims"], x.claim_type)
    for x in differentiators_template:
        _append_term(buckets["differentiators"], x)

    for key in (
        "performance_parameters",
        "soft_parameters",
        "price_indicators",
        "soft_features",
        "claims",
        "differentiators",
    ):
        vals = candidate.get(key)
        if not isinstance(vals, list):
            continue
        for v in vals:
            if isinstance(v, dict):
                if key in ("performance_parameters", "soft_parameters"):
                    _append_term(buckets["performance_parameters"], v.get("name"))
                if key == "price_indicators":
                    _append_term(buckets["price_indicators"], v.get("context"))
                    _append_term(buckets["price_indicators"], v.get("raw"))
                if key == "soft_features":
                    _append_term(buckets["soft_features"], v.get("name"))
                if key == "claims":
                    _append_term(buckets["claims"], v.get("text"))
                    _append_term(buckets["claims"], v.get("evidence"))
                    _append_term(buckets["claims"], v.get("claim_type"))
                if key == "differentiators":
                    _append_term(buckets["differentiators"], v.get("text"))
            else:
                if key in ("performance_parameters", "soft_parameters"):
                    _append_term(buckets["performance_parameters"], v)
                elif key == "price_indicators":
                    _append_term(buckets["price_indicators"], v)
                elif key == "soft_features":
                    _append_term(buckets["soft_features"], v)
                elif key == "claims":
                    _append_term(buckets["claims"], v)
                elif key == "differentiators":
                    _append_term(buckets["differentiators"], v)

    # Guarantee all 5 categories are represented in the final query.
    for k in list(buckets.keys()):
        if not buckets[k]:
            buckets[k] = [k]

    return buckets


def _build_brave_query(
    *,
    product_name: str,
    manufacturer: str,
    candidate: Dict[str, Any],
    perf_template: List[FeatureValue],
    price_template: List[PriceIndicatorValue],
    soft_template: List[SoftFeatureValue],
    claims_template: List[ClaimValue],
    differentiators_template: List[str],
) -> str:
    seeds: List[str] = []
    if product_name:
        seeds.append(product_name)
    if manufacturer:
        seeds.append(manufacturer)
    category = _clean_text(str(candidate.get("category") or ""))
    if category:
        seeds.append(category)
    # Force explicit broadening term so search can surface features
    # not already present as exact query tokens.
    seeds.append("+ further features")

    terms_by_category = _collect_query_terms(
        candidate,
        perf_template,
        price_template,
        soft_template,
        claims_template,
        differentiators_template,
    )
    seen: set[str] = set()
    compact_terms: List[str] = []

    # Ensure each of the 5 categories contributes terms.
    for cat in ("performance_parameters", "price_indicators", "soft_features", "claims", "differentiators"):
        for t in terms_by_category.get(cat, []):
            key = t.lower()
            if key in seen:
                continue
            seen.add(key)
            compact_terms.append(t)
            break

    # Add more terms (deduped) until size limit.
    for cat in ("performance_parameters", "price_indicators", "soft_features", "claims", "differentiators"):
        for t in terms_by_category.get(cat, []):
            key = t.lower()
            if key in seen:
                continue
            seen.add(key)
            compact_terms.append(t)
            if len(compact_terms) >= 25:
                break
        if len(compact_terms) >= 25:
            break

    query = _clean_text(", ".join(seeds + compact_terms))
    return query[:600]


def _build_retry_query(
    *,
    product_name: str,
    manufacturer: str,
    candidate: Dict[str, Any],
    perf_template: List[FeatureValue],
    price_template: List[PriceIndicatorValue],
    soft_template: List[SoftFeatureValue],
) -> str:
    seeds: List[str] = []
    if product_name:
        seeds.append(product_name)
    if manufacturer:
        seeds.append(manufacturer)
    category = _clean_text(str(candidate.get("category") or ""))
    if category:
        seeds.append(category)
    seeds.append("weitere Features")
    seeds.extend([x.name for x in perf_template if x.name])
    seeds.extend([x.context for x in price_template if x.context])
    seeds.extend([x.name for x in soft_template if x.name])
    seeds.append("official product specifications")
    query = _clean_text(", ".join(seeds))
    return query


def _is_null_only_enrichment(enriched: Dict[str, Any]) -> bool:
    if not isinstance(enriched, dict) or not enriched:
        return True

    perf = enriched.get("performance_parameters")
    price = enriched.get("price_indicators")
    soft = enriched.get("soft_features")
    claims = enriched.get("claims")
    diffs = enriched.get("differentiators")

    has_perf = False
    if isinstance(perf, list) and perf:
        has_perf = any(isinstance(x, dict) and x.get("value") is not None for x in perf)

    has_price = False
    if isinstance(price, list) and price:
        has_price = any(
            isinstance(x, dict)
            and ((x.get("value") is not None) or _clean_text(str(x.get("raw") or "")))
            for x in price
        )

    has_soft = False
    if isinstance(soft, list) and soft:
        has_soft = any(isinstance(x, dict) and bool(x.get("available")) for x in soft)

    has_claims = False
    if isinstance(claims, list) and claims:
        has_claims = any(
            isinstance(x, dict)
            and (_clean_text(str(x.get("text") or "")) or _clean_text(str(x.get("evidence") or "")))
            for x in claims
        )

    has_diffs = False
    if isinstance(diffs, list) and diffs:
        has_diffs = any(_clean_text(str(x or "")) for x in diffs)

    return not (has_perf or has_price or has_soft or has_claims or has_diffs)


def _llm_schema() -> Dict[str, Any]:
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
                        "value": {"type": ["number", "null"]},
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
            "differentiators": {"type": "array", "items": {"type": "string"}},
            "source_url": {"type": "string"},
            "source_url_type": {
                "type": "string",
                "enum": ["official", "retailer", "marketplace", "testing", "unknown"],
            },
        },
        "required": [
            "performance_parameters",
            "price_indicators",
            "soft_features",
            "claims",
            "differentiators",
        ],
    }


def _llm_enrich_with_brave(
    *,
    product: Dict[str, Any],
    search_query: str,
    perf_template: List[FeatureValue],
    price_template: List[PriceIndicatorValue],
    soft_template: List[SoftFeatureValue],
    claims_template: List[ClaimValue],
    differentiators_template: List[str],
) -> tuple[Dict[str, Any], str]:
    llm = LlmBrave()
    if not llm.enabled():
        return {}, ""

    payload = {
        "product": product,
        "search_query": _clean_text(search_query)[:600],
        "reference_templates": {
            "performance_parameters": [x.model_dump() for x in perf_template],
            "price_indicators": [x.model_dump() for x in price_template],
            "soft_features": [x.model_dump() for x in soft_template],
            "claims": [x.model_dump() for x in claims_template],
        },
    }

    user = (
        "You enrich competitor product profiles via Brave web search. "
        "Search the web using the provided query and product. "
        "Fill exact parameters from reference templates. "
        "If a value is not explicitly present, keep null/false/empty. "
        "Return JSON only.\n\n"
        "Task:\n"
        "0) Ignore any pre-existing product URL as source input; use web search based on search_query.\n"
        "1) Keep template feature names.\n"
        "2) Fill values only if explicitly supported by evidence.\n"
        "3) You may add new features only if clearly explicit.\n"
        "4) Measurable/numeric values (e.g. Pa, mm, ml, L, kg, °C, W) must go to performance_parameters (or price_indicators for prices), not soft_features.\n"
        "5) soft_features are qualitative availability flags only (boolean), no numeric values.\n"
        "6) claims must use claim_type in value|benefit|differentiation.\n"
        "7) differentiators must be evidence-based only; never use fixed/default/template text.\n"
        "8) If no clear differentiator is found, return differentiators as an empty list.\n"
        "9) Provide source URLs and types: source_url/source_url_type, optionally source_url_2/source_url_type_2 and source_url_3/source_url_type_3.\n"
        f"Input:\n{json.dumps(payload, ensure_ascii=False)}"
    )

    try:
        resp = llm.chat_completions(
            messages=[{"role": "user", "content": user}],
            stream=False,
            timeout_s=60,
        )
        text = llm.extract_text(resp)
        return _parse_json_strictish(text), text
    except Exception:
        return {}, ""


def _merge_perf(template: List[FeatureValue], incoming: Any) -> List[FeatureValue]:
    out: Dict[str, FeatureValue] = {_clean_text(x.name).lower(): x for x in template}
    if isinstance(incoming, list):
        for r in incoming:
            if not isinstance(r, dict):
                continue
            n = _clean_text(str(r.get("name") or ""))
            if not n:
                continue
            v = r.get("value")
            if isinstance(v, str):
                try:
                    v = float(v.replace(",", "."))
                except Exception:
                    v = None
            u = _clean_text(str(r.get("unit") or ""))
            out[n.lower()] = FeatureValue(name=n, value=v if isinstance(v, (int, float)) else None, unit=u)
    return list(out.values())


def _merge_price(template: List[PriceIndicatorValue], incoming: Any) -> List[PriceIndicatorValue]:
    out: Dict[str, PriceIndicatorValue] = {(_clean_text(x.context).lower() or x.context.lower()): x for x in template}
    if isinstance(incoming, list):
        for r in incoming:
            if not isinstance(r, dict):
                continue
            ctx = _clean_text(str(r.get("context") or "")) or "Preis"
            v = r.get("value")
            if isinstance(v, str):
                try:
                    v = float(v.replace(",", "."))
                except Exception:
                    v = None
            out[ctx.lower()] = PriceIndicatorValue(
                raw=_clean_text(str(r.get("raw") or "")),
                value=v if isinstance(v, (int, float)) else None,
                currency=_clean_text(str(r.get("currency") or "")),
                period=_clean_text(str(r.get("period") or "")),
                context=ctx,
            )
    return list(out.values())


def _merge_soft(template: List[SoftFeatureValue], incoming: Any) -> List[SoftFeatureValue]:
    out: Dict[str, SoftFeatureValue] = {_clean_text(x.name).lower(): x for x in template}
    if isinstance(incoming, list):
        for r in incoming:
            if not isinstance(r, dict):
                continue
            n = _clean_text(str(r.get("name") or ""))
            if not n:
                continue
            avail = bool(r.get("available"))
            key = n.lower()
            if key in out:
                out[key].available = out[key].available or avail
            else:
                out[key] = SoftFeatureValue(name=n, available=avail)
    return list(out.values())


def _merge_claims(template: List[ClaimValue], incoming: Any) -> List[ClaimValue]:
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
            out.append(ClaimValue(text=text, claim_type=ctype, evidence=evidence))
            if len(out) >= 3:
                break
    if out:
        while len(out) < 3:
            out.append(ClaimValue(text="", claim_type=template[len(out)].claim_type, evidence=""))
        return out
    return template


def _merge_differentiators(template: List[str], incoming: Any) -> List[str]:
    out: List[str] = []
    if isinstance(incoming, list):
        for x in incoming:
            s = _clean_text(str(x or ""))
            if s:
                out.append(s)
            if len(out) >= 3:
                break
    return out


def _collect_source_candidates(enriched: Dict[str, Any]) -> List[tuple[str, str]]:
    out: List[tuple[str, str]] = []

    def _add(url_val: Any, type_val: Any) -> None:
        u = _clean_text(str(url_val or ""))
        if not u:
            return
        t = _clean_text(str(type_val or "")) or "unknown"
        out.append((u, t))

    _add(enriched.get("source_url"), enriched.get("source_url_type"))
    _add(enriched.get("source_url_2"), enriched.get("source_url_type_2"))
    _add(enriched.get("source_url_3"), enriched.get("source_url_type_3"))
    _add(enriched.get("url_source"), enriched.get("url_source_type"))
    _add(enriched.get("url_source_2"), enriched.get("url_source_type_2"))
    _add(enriched.get("url_source_3"), enriched.get("url_source_type_3"))

    for key in ("performance_parameters", "price_indicators", "soft_features", "claims"):
        vals = enriched.get(key)
        if not isinstance(vals, list):
            continue
        for item in vals:
            if not isinstance(item, dict):
                continue
            _add(item.get("source_url"), item.get("source_url_type"))
            _add(item.get("url_source"), item.get("url_source_type"))

    dedup: List[tuple[str, str]] = []
    seen: set[str] = set()
    for u, t in out:
        k = u.lower()
        if k in seen:
            continue
        seen.add(k)
        dedup.append((u, t))
    return dedup


def extract_competitor_profiles_v0_5(
    *,
    competitor_search_results: Optional[Dict[str, Any]],
    competitor_search_results_path: Optional[str],
    product_profile: Optional[Dict[str, Any]],
    product_profile_path: Optional[str],
    provider: str = "brave",
    max_competitors: int = 200,
    include_page_fetch: bool = True,
    page_fetch_timeout_s: int = 8,
    page_fetch_max_chars: int = 8000,
    verbose_terminal: bool = False,
    user_root=None,
    work_root=None,
) -> CompetitorProfileExtractionResultsV05:
    def _log(msg: str) -> None:
        if verbose_terminal:
            print(f"[competitor_profile_extraction_v0_5] {msg}")

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

    warnings: List[str] = [str(w).strip() for w in (csr.get("extraction_warnings") or []) if str(w).strip()]
    competitors_raw = csr.get("competitors") if isinstance(csr.get("competitors"), list) else []

    perf_template = _template_perf(profile)
    price_template = _template_price(profile)
    soft_template = _template_soft(profile)
    claims_template = _template_claims(profile)
    differentiators_template = _template_differentiators(profile)

    out: List[CompetitorEnrichedV05] = []
    limit = max(1, int(max_competitors))

    _log(f"start provider={provider} competitors_in={len(competitors_raw)} max_competitors={limit}")
    if include_page_fetch:
        _log("include_page_fetch is ignored in v0.5 search-first mode")

    for i, c in enumerate(competitors_raw[:limit], start=1):
        if not isinstance(c, dict):
            continue
        product_name = _clean_text(str(c.get("product_name") or ""))
        manufacturer = _clean_text(str(c.get("manufacturer") or ""))
        input_url = _clean_text(str(c.get("url") or ""))
        input_url_type = _clean_text(str(c.get("url_type") or "unknown"))
        relevance = float(c.get("relevance_score") or 0.0)
        similarity = float(c.get("similarity_score") or 0.0)

        if not product_name:
            warnings.append(f"v0.5 skipped competitor #{i} due to missing product_name.")
            continue

        search_query = _build_brave_query(
            product_name=product_name,
            manufacturer=manufacturer,
            candidate=c,
            perf_template=perf_template,
            price_template=price_template,
            soft_template=soft_template,
            claims_template=claims_template,
            differentiators_template=differentiators_template,
        )
        _log(f"[{i}/{min(len(competitors_raw), limit)}] product={product_name}")
        _log(f"search query={search_query}")

        enriched, llm_raw_text = _llm_enrich_with_brave(
            product={
                "product_name": product_name,
                "manufacturer": manufacturer,
                "url": input_url,
                "url_type": input_url_type,
            },
            search_query=search_query,
            perf_template=[FeatureValue(name=x.name, value=x.value, unit=x.unit) for x in perf_template],
            price_template=[
                PriceIndicatorValue(raw=x.raw, value=x.value, currency=x.currency, period=x.period, context=x.context)
                for x in price_template
            ],
            soft_template=[SoftFeatureValue(name=x.name, available=x.available) for x in soft_template],
            claims_template=[ClaimValue(text=x.text, claim_type=x.claim_type, evidence=x.evidence) for x in claims_template],
            differentiators_template=list(differentiators_template),
        )
        if llm_raw_text and verbose_terminal:
            _log(f"search response={_clean_text(llm_raw_text)[:1200]}")

        if _is_null_only_enrichment(enriched):
            retry_query = _build_retry_query(
                product_name=product_name,
                manufacturer=manufacturer,
                candidate=c,
                perf_template=perf_template,
                price_template=price_template,
                soft_template=soft_template,
            )
            _log("retry start (reason=null-only)")
            _log(f"retry query={retry_query}")
            retry_enriched, retry_raw = _llm_enrich_with_brave(
                product={
                    "product_name": product_name,
                    "manufacturer": manufacturer,
                    "url": input_url,
                    "url_type": input_url_type,
                },
                search_query=retry_query,
                perf_template=[FeatureValue(name=x.name, value=x.value, unit=x.unit) for x in perf_template],
                price_template=[
                    PriceIndicatorValue(raw=x.raw, value=x.value, currency=x.currency, period=x.period, context=x.context)
                    for x in price_template
                ],
                soft_template=[SoftFeatureValue(name=x.name, available=x.available) for x in soft_template],
                claims_template=[ClaimValue(text=x.text, claim_type=x.claim_type, evidence=x.evidence) for x in claims_template],
                differentiators_template=list(differentiators_template),
            )
            if verbose_terminal:
                _log(f"retry response={_clean_text(retry_raw)[:1200] if retry_raw else '<empty>'}")
            if not _is_null_only_enrichment(retry_enriched):
                enriched = retry_enriched
                _log("retry accepted")
            else:
                _log("retry still null-only")

        if not enriched:
            warnings.append(f"v0.5 brave enrichment failed/empty for product: {product_name}")

        # Keep original url/url_type unchanged; append additional discovered sources as url2/url3.
        resolved_url = input_url
        resolved_url_type = input_url_type or "unknown"

        source_candidates = _collect_source_candidates(enriched or {})
        additional_sources: List[tuple[str, str]] = []
        seen: set[str] = set()
        if resolved_url:
            seen.add(resolved_url.lower())
        for u, t in source_candidates:
            k = u.lower()
            if k in seen:
                continue
            seen.add(k)
            additional_sources.append((u, t))
            if len(additional_sources) >= 2:
                break

        resolved_url2, resolved_url2_type = ("", "unknown")
        resolved_url3, resolved_url3_type = ("", "unknown")
        if len(additional_sources) >= 1:
            resolved_url2, resolved_url2_type = additional_sources[0]
        if len(additional_sources) >= 2:
            resolved_url3, resolved_url3_type = additional_sources[1]

        perf = _merge_perf(
            [FeatureValue(name=x.name, value=x.value, unit=x.unit) for x in perf_template],
            enriched.get("performance_parameters") if enriched else None,
        )
        price = _merge_price(
            [
                PriceIndicatorValue(raw=x.raw, value=x.value, currency=x.currency, period=x.period, context=x.context)
                for x in price_template
            ],
            enriched.get("price_indicators") if enriched else None,
        )
        soft = _merge_soft(
            [SoftFeatureValue(name=x.name, available=x.available) for x in soft_template],
            enriched.get("soft_features") if enriched else None,
        )
        claims = _merge_claims(
            [ClaimValue(text=x.text, claim_type=x.claim_type, evidence=x.evidence) for x in claims_template],
            enriched.get("claims") if enriched else None,
        )
        differentiators = _merge_differentiators(
            list(differentiators_template),
            enriched.get("differentiators") if enriched else None,
        )

        out.append(
            CompetitorEnrichedV05(
                product_name=product_name,
                manufacturer=manufacturer,
                url=resolved_url,
                url_type=resolved_url_type,
                url2=resolved_url2,
                url2_type=resolved_url2_type,
                url3=resolved_url3,
                url3_type=resolved_url3_type,
                performance_parameters=perf,
                price_indicators=price,
                soft_features=soft,
                claims=claims,
                differentiators=differentiators,
                relevance_score=relevance,
                similarity_score=similarity,
            )
        )

    warnings = list(dict.fromkeys([w for w in warnings if _clean_text(w)]))
    _log(f"done competitors={len(out)} warnings={len(warnings)}")

    return CompetitorProfileExtractionResultsV05(
        schema_version="1.0",
        provider="brave",
        competitors=out,
        extraction_warnings=warnings,
    )
