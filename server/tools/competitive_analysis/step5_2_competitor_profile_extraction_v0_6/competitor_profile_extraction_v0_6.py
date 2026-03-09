from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from server.services.llm_ionos import IonosLLM
from server.services.llm_openai import LlmOpenai
from server.services.llm_perplexity import LlmPerplexity
from server.tools.competitive_analysis.backup.competitor_identification import (
    _clean_text,
    _load_json_obj,
)
from server.tools.competitive_analysis.step5_competitor_profile_extraction_v0_5.competitor_profile_extraction_v0_5 import (
    _derive_metric_features_from_perf,
    _llm_schema,
    _merge_claims,
    _merge_differentiators,
    _merge_perf,
    _merge_price,
    _merge_soft,
    _template_claims,
    _template_differentiators,
    _template_metric,
    _template_perf,
    _template_price,
    _template_soft,
)

from .models import (
    ClaimValue,
    CompetitorEnrichedV06,
    CompetitorProfileExtractionResultsV06,
    FeatureValue,
    PriceIndicatorValue,
    SoftFeatureValue,
)


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


def _llm_extract_structured(
    *,
    provider: str,
    product: Dict[str, Any],
    plain_text: str,
    perf_template: List[FeatureValue],
    metric_template: List[FeatureValue],
    price_template: List[PriceIndicatorValue],
    soft_template: List[SoftFeatureValue],
    claims_template: List[ClaimValue],
) -> Dict[str, Any]:
    schema = _llm_schema()
    system = (
        "Extract structured competitor profile data from plain text. "
        "Return strict JSON according to schema. "
        "Use reference template names where possible. "
        "Map physical measurable specs to metric_features when applicable. "
        "Only use explicitly present evidence from plain_text."
    )
    user_payload = {
        "product": product,
        "plain_text": str(plain_text or "")[:40000],
        "reference_templates": {
            "performance_parameters": [x.model_dump() for x in perf_template],
            "metric_features": [x.model_dump() for x in metric_template],
            "price_indicators": [x.model_dump() for x in price_template],
            "soft_features": [x.model_dump() for x in soft_template],
            "claims": [x.model_dump() for x in claims_template],
        },
    }
    user = "Input:\n" + json.dumps(user_payload, ensure_ascii=False)

    p = str(provider or "openai").strip().lower()
    if p not in {"openai", "perplexity", "ionos"}:
        p = "openai"

    if p in {"openai", "perplexity"}:
        c = LlmOpenai() if p == "openai" else LlmPerplexity()
        if not c.enabled():
            return {}
        try:
            resp = c._call(
                input_messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                text_format={"type": "json_schema", "name": "competitor_profile_extraction_v06", "schema": schema, "strict": False},
            )
            text = ""
            for item in resp.get("output", []):
                for cc in item.get("content", []):
                    if cc.get("type") == "output_text":
                        text += str(cc.get("text") or "")
            return _parse_json_strictish(text)
        except Exception:
            return {}

    c2 = IonosLLM()
    if not c2.enabled():
        return {}
    try:
        comp = c2.chat_completions(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "competitor_profile_extraction_v06", "schema": schema, "strict": True},
            },
        )
        return _parse_json_strictish(c2.extract_text(comp))
    except Exception:
        return {}


def extract_competitor_profiles_v0_6(
    *,
    competitor_profile_text: Optional[Dict[str, Any]],
    competitor_profile_text_path: Optional[str],
    product_profile: Optional[Dict[str, Any]],
    product_profile_path: Optional[str],
    provider: str = "openai",
    max_competitors: int = 200,
    verbose_terminal: bool = False,
    user_root=None,
    work_root=None,
) -> CompetitorProfileExtractionResultsV06:
    def _log(msg: str) -> None:
        if verbose_terminal:
            print(f"[competitor_profile_extraction_v0_6] {msg}")

    text_payload = _load_json_obj(
        inline_obj=competitor_profile_text,
        path=competitor_profile_text_path,
        root_key="competitor_profile_text",
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

    warnings: List[str] = [str(w).strip() for w in (text_payload.get("extraction_warnings") or []) if str(w).strip()]
    competitors_raw = text_payload.get("competitors") if isinstance(text_payload.get("competitors"), list) else []
    competitors_raw = [c for c in competitors_raw if isinstance(c, dict)]
    limit = max(1, int(max_competitors))

    non_empty_plain = sum(1 for c in competitors_raw if _clean_text(str(c.get("plain_text") or "")))
    if competitors_raw and non_empty_plain == 0:
        warnings.append(
            "v0.6 input check: 0 nicht-leere plain_text-Einträge in competitor_profile_text. "
            "Prüfe competitor_profile_text_path (erwartet: Output aus Schritt 5.1)."
        )

    perf_template = _template_perf(profile)
    metric_template = _template_metric(profile, perf_template)
    price_template = _template_price(profile)
    soft_template = _template_soft(profile)
    claims_template = _template_claims(profile)
    differentiators_template = _template_differentiators(profile)

    out: List[CompetitorEnrichedV06] = []
    _log(f"start provider={provider} competitors_in={len(competitors_raw)} max_competitors={limit}")
    for idx, c in enumerate(competitors_raw[:limit], start=1):
        product_name = _clean_text(str(c.get("product_name") or ""))
        manufacturer = _clean_text(str(c.get("manufacturer") or ""))
        url = _clean_text(str(c.get("url") or ""))
        url_type = _clean_text(str(c.get("url_type") or "unknown")) or "unknown"
        relevance = float(c.get("relevance_score") or 0.0)
        similarity = float(c.get("similarity_score") or 0.0)
        plain_text = str(c.get("plain_text") or "")

        if not product_name:
            warnings.append(f"v0.6 skipped competitor #{idx} due to missing product_name.")
            continue
        _log(f"[{idx}/{min(len(competitors_raw), limit)}] product={product_name}")

        enriched = _llm_extract_structured(
            provider=provider,
            product={
                "product_name": product_name,
                "manufacturer": manufacturer,
                "url": url,
                "url_type": url_type,
            },
            plain_text=plain_text,
            perf_template=[FeatureValue(name=x.name, value=x.value, unit=x.unit) for x in perf_template],
            metric_template=[FeatureValue(name=x.name, value=x.value, unit=x.unit) for x in metric_template],
            price_template=[
                PriceIndicatorValue(raw=x.raw, value=x.value, currency=x.currency, period=x.period, context=x.context)
                for x in price_template
            ],
            soft_template=[SoftFeatureValue(name=x.name, available=x.available) for x in soft_template],
            claims_template=[ClaimValue(text=x.text, claim_type=x.claim_type, evidence=x.evidence) for x in claims_template],
        )
        if not enriched:
            warnings.append(f"v0.6 structured extraction failed/empty for product: {product_name}")

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

        # _merge_* helpers are imported from v0.5 and return v0.5 model instances.
        # Convert explicitly into v0.6 model instances to avoid cross-model validation errors.
        perf_v06 = [FeatureValue(**(x.model_dump() if hasattr(x, "model_dump") else dict(x))) for x in perf]
        metric_merged = _merge_perf(
            [FeatureValue(name=x.name, value=x.value, unit=x.unit) for x in metric_template],
            enriched.get("metric_features") if enriched else None,
        )
        derived_metric = _derive_metric_features_from_perf(perf)
        if derived_metric:
            metric_merged = _merge_perf(
                [FeatureValue(name=x.name, value=x.value, unit=x.unit) for x in metric_merged],
                [x.model_dump() if hasattr(x, "model_dump") else dict(x) for x in derived_metric],
            )
        metric_v06 = [
            FeatureValue(**(x.model_dump() if hasattr(x, "model_dump") else dict(x)))
            for x in (metric_merged or metric_template)
        ]
        price_v06 = [PriceIndicatorValue(**(x.model_dump() if hasattr(x, "model_dump") else dict(x))) for x in price]
        soft_v06 = [SoftFeatureValue(**(x.model_dump() if hasattr(x, "model_dump") else dict(x))) for x in soft]
        claims_v06 = [ClaimValue(**(x.model_dump() if hasattr(x, "model_dump") else dict(x))) for x in claims]

        out.append(
            CompetitorEnrichedV06(
                product_name=product_name,
                manufacturer=manufacturer,
                url=url,
                url_type=url_type,
                performance_parameters=perf_v06,
                metric_features=metric_v06,
                price_indicators=price_v06,
                soft_features=soft_v06,
                claims=claims_v06,
                differentiators=differentiators,
                relevance_score=round(relevance, 4),
                similarity_score=round(similarity, 4),
            )
        )

    _log(f"done competitors={len(out)} warnings={len(warnings)}")
    return CompetitorProfileExtractionResultsV06(
        schema_version="1.0",
        provider=str(provider or "openai").strip().lower(),
        competitors=out,
        extraction_warnings=list(dict.fromkeys([w for w in warnings if _clean_text(w)])),
    )
