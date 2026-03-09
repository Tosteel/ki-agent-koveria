from __future__ import annotations
from typing import Any, Dict, List, Optional

from server.services.llm_brave import LlmBrave
from server.tools.competitive_analysis.backup.competitor_identification import (
    _clean_text,
    _load_json_obj,
)
from server.tools.competitive_analysis.step5_competitor_profile_extraction_v0_5.competitor_profile_extraction_v0_5 import (
    _build_brave_query,
    _template_claims,
    _template_differentiators,
    _template_metric,
    _template_perf,
    _template_price,
    _template_soft,
)

from .models import CompetitorProductTextV06, CompetitorProfileTextResultsV06


def _brave_raw_text(
    *,
    query: str,
    product: Dict[str, Any],
    brave_enable_research: bool,
    brave_stream: bool,
    brave_language: Optional[str],
    brave_country: Optional[str],
) -> tuple[str, str]:
    llm = LlmBrave()
    if not llm.enabled():
        return "", "brave_not_configured"
    # Intentionally send only the plain search query to Brave Answers (no additional instruction prompt).
    user = _clean_text(query)
    try:
        resp = llm.chat_completions(
            messages=[{"role": "user", "content": user}],
            stream=brave_stream,
            enable_research=brave_enable_research,
            language=brave_language,
            country=brave_country,
            timeout_s=90,
        )
        text = llm.extract_text(resp) or ""
        if not text:
            return "", "brave_empty_response"
        return text, ""
    except Exception as exc:
        return "", f"brave_error: {exc}"


def build_competitor_profile_text_v0_6(
    *,
    competitor_product_results: Optional[Dict[str, Any]],
    competitor_product_results_path: Optional[str],
    product_profile: Optional[Dict[str, Any]],
    product_profile_path: Optional[str],
    provider: str = "brave",
    max_competitors: int = 200,
    brave_enable_research: bool = False,
    brave_stream: bool = True,
    brave_language: Optional[str] = "de",
    brave_country: Optional[str] = "DE",
    verbose_terminal: bool = False,
    user_root=None,
    work_root=None,
) -> CompetitorProfileTextResultsV06:
    def _log(msg: str) -> None:
        if verbose_terminal:
            print(f"[competitor_profile_text_v0_6] {msg}")

    payload = _load_json_obj(
        inline_obj=competitor_product_results,
        path=competitor_product_results_path,
        root_key="competitor_product_results",
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

    warnings: List[str] = [str(w).strip() for w in (payload.get("extraction_warnings") or []) if str(w).strip()]
    generated_queries = [str(q).strip() for q in (payload.get("generated_queries") or []) if str(q).strip()]

    competitors_raw = payload.get("competitors") if isinstance(payload.get("competitors"), list) else []
    competitors_raw = [c for c in competitors_raw if isinstance(c, dict)]
    limit = max(1, int(max_competitors))

    perf_template = _template_perf(profile)
    metric_template = _template_metric(profile, perf_template)
    price_template = _template_price(profile)
    soft_template = _template_soft(profile)
    claims_template = _template_claims(profile)
    differentiators_template = _template_differentiators(profile)

    out: List[CompetitorProductTextV06] = []
    _log(
        f"start competitors_in={len(competitors_raw)} max_competitors={limit} "
        f"research={brave_enable_research} stream={brave_stream} lang={brave_language} country={brave_country}"
    )
    for idx, c in enumerate(competitors_raw[:limit], start=1):
        product_name = _clean_text(str(c.get("product_name") or ""))
        manufacturer = _clean_text(str(c.get("manufacturer") or ""))
        url = _clean_text(str(c.get("url") or ""))
        url_type = _clean_text(str(c.get("url_type") or "unknown")) or "unknown"
        relevance = float(c.get("relevance_score") or 0.0)
        similarity = float(c.get("similarity_score") or 0.0)

        if not product_name:
            warnings.append(f"v0.6 skipped competitor #{idx} due to missing product_name.")
            continue

        query = _build_brave_query(
            product_name=product_name,
            manufacturer=manufacturer,
            candidate=c,
            perf_template=perf_template,
            metric_template=metric_template,
            price_template=price_template,
            soft_template=soft_template,
            claims_template=claims_template,
            differentiators_template=differentiators_template,
        )
        _log(f"[{idx}/{min(len(competitors_raw), limit)}] product={product_name}")
        _log(f"search query={query}")
        plain_text, cause = _brave_raw_text(
            query=query,
            product={
                "product_name": product_name,
                "manufacturer": manufacturer,
                "url": url,
                "url_type": url_type,
            },
            brave_enable_research=brave_enable_research,
            brave_stream=brave_stream,
            brave_language=brave_language,
            brave_country=brave_country,
        )
        if verbose_terminal:
            _log(f"search response={_clean_text(plain_text)[:1200] if plain_text else '<empty>'}")
        if not plain_text:
            cause_txt = cause or "unknown"
            warnings.append(
                f"v0.6 brave text empty for product: {product_name} ({cause_txt})"
            )

        out.append(
            CompetitorProductTextV06(
                product_name=product_name,
                manufacturer=manufacturer,
                url=url,
                url_type=url_type,
                relevance_score=round(relevance, 4),
                similarity_score=round(similarity, 4),
                plain_text=str(plain_text or ""),
            )
        )

    _log(f"done competitors={len(out)} warnings={len(warnings)}")
    return CompetitorProfileTextResultsV06(
        schema_version="1.0",
        provider=str(provider or "brave").strip().lower(),
        generated_queries=generated_queries,
        competitors=out,
        extraction_warnings=list(dict.fromkeys([w for w in warnings if _clean_text(w)])),
    )
