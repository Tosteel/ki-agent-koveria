from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

from server.services.llm_ionos import IonosLLM
from server.services.llm_openai import LlmOpenai
from server.services.llm_perplexity import LlmPerplexity
from server.tools.competitive_analysis.backup.adaptive_analysis_plan.adaptive_analysis_plan import (
    _heuristic_plan,
    _load_product_profile,
    _openai_extract_output_text,
    _parse_json_strictish,
    _sanitize_llm_queries,
    _sanitize_plan,
)
from server.tools.competitive_analysis.backup.adaptive_analysis_plan.models import AnalysisPlan, ComparisonDimension, SearchTerm
from server.tools.competitive_analysis.backup.competitor_identification.competitor_identification import (
    _langsearch_fallback,
    _openai_search,
    _perplexity_search,
)


def _dedupe_keep_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for it in items:
        t = str(it or "").strip()
        if not t:
            continue
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out


def _generic_soft_feature_name(name: str) -> str:
    n = str(name or "").strip()
    low = n.lower()
    mapping = [
        (("proleap", "hindernis"), "Hindernisüberwindung"),
        (("versalift", "navigation"), "Navigation"),
        (("vormax", "saug"), "Saugsystem"),
        (("hyperstream", "duobrush", "anti-haar", "haar"), "Anti-Haarverhedderung"),
        (("powerdock", "entleer"), "Automatische Staubentleerung"),
        (("heißluft", "heissluft", "trocknung"), "Mopp-Trocknung"),
        (("app",), "App-Steuerung"),
        (("sprach", "alexa", "assistant"), "Sprachsteuerung"),
    ]
    for keys, label in mapping:
        if any(k in low for k in keys):
            return label
    n = re.sub(r"[™®©]", "", n)
    n = re.sub(r"\s+", " ", n).strip(" -")
    return n


_MEASURE_TERM_HINTS = (
    "breite",
    "höhe",
    "hoehe",
    "länge",
    "laenge",
    "tiefe",
    "abmessung",
    "maße",
    "masse",
    "dimension",
    "size",
    "width",
    "height",
    "depth",
    "length",
)


def _is_measure_term(name: str) -> bool:
    n = str(name or "").strip().lower()
    if not n:
        return False
    return any(h in n for h in _MEASURE_TERM_HINTS)


_SIZE_WEIGHT_QUERY_HINTS = (
    "gewicht",
    "gesamtgewicht",
    "maße",
    "masse",
    "größe",
    "groesse",
    "abmessung",
    "dimension",
    "breite",
    "höhe",
    "hoehe",
    "länge",
    "laenge",
    "tiefe",
    "width",
    "height",
    "length",
    "depth",
)


def _is_size_weight_query(text: str) -> bool:
    t = str(text or "").strip().lower()
    if not t:
        return False
    if any(h in t for h in _SIZE_WEIGHT_QUERY_HINTS):
        return True
    return False


def _primary_category_term(category: str) -> str:
    cat = str(category or "").strip()
    if not cat:
        return "Produkt"
    low = cat.lower()
    # Example: "Saug- und Wischroboter" -> "Saugroboter"
    m = re.match(r"^\s*([A-Za-zÄÖÜäöüß]+)-\s*und\s*([A-Za-zÄÖÜäöüß]+)\s*$", low)
    if m:
        left, right = m.group(1), m.group(2)
        if right.endswith("roboter"):
            return (left.capitalize() + "roboter").strip()
    if " und " in low:
        first = re.split(r"\bund\b", cat, flags=re.IGNORECASE)[0]
        first = re.sub(r"[-/,\s]+$", "", first).strip()
        if first:
            return first
    return cat


def _build_priority_queries(
    *,
    category: str,
    price_terms: List[str],
    perf_terms: List[str],
    soft_terms: List[str],
    llm_queries: List[str],
    max_queries: int,
) -> List[str]:
    cat = _primary_category_term(category)
    out: List[str] = []

    # Category + performance
    for t in perf_terms[:8]:
        out.append(f"{cat} {t}")

    # Category + soft feature
    for t in soft_terms[:8]:
        out.append(f"{cat} {t}")

    # Keep LLM queries as secondary fill-up.
    out.extend(llm_queries)
    out = _dedupe_keep_order(out)
    # Always keep analysis-plan query count capped to 20.
    cap = min(20, max(16, int(max_queries)))
    return out[:cap]


def _additional_terms_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "additional_search_terms": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "term": {"type": "string"},
                        "intent": {"type": "string"},
                    },
                    "required": ["term", "intent"],
                },
            }
        },
        "required": ["additional_search_terms"],
    }


def _quick_web_search(provider: str, query: str, per_query_results: int = 6) -> List[Dict[str, str]]:
    p = str(provider or "openai").strip().lower()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    perplexity_key = os.getenv("PERPLEXITY_API_KEY", "").strip()
    perplexity_model = os.getenv("PERPLEXITY_MODEL", "sonar-pro").strip() or "sonar-pro"
    try:
        if p == "openai" and openai_key:
            return _openai_search(query, per_query_results, api_key=openai_key, model=openai_model)
        if p == "perplexity" and perplexity_key:
            return _perplexity_search(query, per_query_results, api_key=perplexity_key, model=perplexity_model)
        return _langsearch_fallback(query, per_query_results)
    except Exception:
        return []


def _llm_additional_terms_from_web(
    *,
    provider: str,
    category: str,
    web_context: str,
    warnings: List[str],
) -> List[SearchTerm]:
    schema = _additional_terms_schema()
    system = (
        "Erzeuge additional_search_terms als JSON. "
        "Liefere mehrere Einträge intent='Model-Tiering' als konkrete Wettbewerber-Produktnamen, "
        "mehrere Einträge intent='Brand-Tiering' nur als Herstellernamen/Brands "
        "und genau einen Eintrag intent='Market positioning' mit Wert aus [Premium, Midrange, Budget]."
    )
    user = (
        f"Kategorie: {category}\n"
        f"Webtreffer (kurz):\n{web_context[:9000]}\n\n"
        "Gib nur JSON gemäß Schema zurück."
    )
    p = str(provider or "ionos").strip().lower()
    if p not in {"ionos", "openai", "perplexity"}:
        p = "ionos"

    parsed: Dict[str, Any] = {}
    if p in {"openai", "perplexity"}:
        client = LlmOpenai() if p == "openai" else LlmPerplexity()
        if not client.enabled():
            return []
        try:
            resp = client._call(
                input_messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                text_format={"type": "json_schema", "name": "additional_terms", "schema": schema, "strict": False},
            )
            parsed = _parse_json_strictish(_openai_extract_output_text(resp))
        except Exception as exc:
            warnings.append(f"v0.2 additional_search_terms web-LLM failed ({p}): {exc}")
            return []
    else:
        ion = IonosLLM()
        if not ion.enabled():
            return []
        try:
            completion = ion.chat_completions(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "additional_terms",
                        "schema": schema,
                        "strict": True,
                    },
                },
            )
            parsed = _parse_json_strictish(ion.extract_text(completion))
        except Exception as exc:
            warnings.append(f"v0.2 additional_search_terms web-LLM failed (ionos): {exc}")
            return []

    out: List[SearchTerm] = []
    for item in parsed.get("additional_search_terms") or []:
        if not isinstance(item, dict):
            continue
        term = str(item.get("term") or "").strip()
        intent = str(item.get("intent") or "").strip()
        if term and intent:
            out.append(SearchTerm(term=term, intent=intent))
    return out


def _base_dimensions() -> List[ComparisonDimension]:
    return [
        ComparisonDimension(
            name="Leistung",
            weight=0.25,
            rationale="Technische Performance ist primäres Differenzierungsmerkmal.",
            required_fields=["performance_parameters", "normalized_features"],
        ),
        ComparisonDimension(
            name="Preis",
            weight=0.2,
            rationale="Preisniveau beeinflusst Wettbewerbsposition und Zielsegment.",
            required_fields=["price_indicators"],
        ),
        ComparisonDimension(
            name="Features",
            weight=0.25,
            rationale="Feature-Abdeckung bestimmt Markt-Fit.",
            required_fields=["normalized_features", "extended_feature_schema"],
        ),
        ComparisonDimension(
            name="Compliance",
            weight=0.15,
            rationale="Normen/Zertifikate sind branchenkritisch.",
            required_fields=["notes", "relevance_criteria"],
        ),
        ComparisonDimension(
            name="Zielgruppe- & Use-Case-Fit",
            weight=0.15,
            rationale="Ähnliche Zielsegmente sichern Vergleichbarkeit.",
            required_fields=["target_segments", "use_cases"],
        ),
    ]


def _step_schema(step: int) -> Dict[str, Any]:
    if step == 1:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "product_category": {"type": "string"},
                "comparison_dimensions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "name": {"type": "string"},
                            "weight": {"type": "number"},
                            "rationale": {"type": "string"},
                            "required_fields": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["name", "weight", "rationale", "required_fields"],
                    },
                },
            },
            "required": ["product_category", "comparison_dimensions"],
        }
    if step == 2:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "extended_feature_schema": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["extended_feature_schema"],
        }
    if step == 3:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "search_terms": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "term": {"type": "string"},
                            "intent": {"type": "string"},
                        },
                        "required": ["term", "intent"],
                    },
                },
                "additional_search_terms": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "term": {"type": "string"},
                            "intent": {"type": "string"},
                        },
                        "required": ["term", "intent"],
                    },
                },
                "search_queries": {"type": "array", "items": {"type": "string"}},
                "min_competitors": {"type": "integer"},
            },
            "required": ["search_terms", "additional_search_terms", "search_queries", "min_competitors"],
        }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "relevance_criteria": {"type": "array", "items": {"type": "string"}},
            "analysis_scope": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "depth": {"type": "string", "enum": ["shallow", "medium", "deep"]},
                    "breadth": {"type": "string", "enum": ["narrow", "medium", "broad"]},
                    "include_regional": {"type": "boolean"},
                    "include_global": {"type": "boolean"},
                    "max_results_per_query": {"type": "integer"},
                },
                "required": ["depth", "breadth", "include_regional", "include_global", "max_results_per_query"],
            },
            "notes": {"type": "array", "items": {"type": "string"}, "default": []},
        },
        "required": ["relevance_criteria", "analysis_scope", "notes"],
    }


def _llm_step(provider: str, *, step: int, context: str, partial: Dict[str, Any], warnings: List[str]) -> Dict[str, Any]:
    schema = _step_schema(step)
    system = (
        "Du erzeugst einen Teil eines Wettbewerbsanalyse-Plans als JSON. "
        "Nur gültiges JSON gemäß Schema. "
        "Keine 'vs'-Queries, keine site:-Queries, keine URLs. "
        "required_fields nur generisch."
    )
    if step == 3:
        system += (
            " search_terms müssen zusätzlich performance- und soft-feature-Terme enthalten "
            "(intent='performance' bzw. intent='soft_feature'). "
            " Ergänze additional_search_terms per LLM mit mehreren Einträgen intent='Model-Tiering' "
            "als konkrete Wettbewerber-Produktnamen, mehreren Einträgen intent='Brand-Tiering' nur als Herstellernamen "
            "sowie genau einem Eintrag "
            "intent='Market positioning' mit einem Wert aus [Premium, Midrange, Budget]. "
            "Erzeuge intelligente search_queries primär aus Kombinationen: "
            "category+performance_parameter, category+soft_feature sowie category+Brand-Tiering+Market positioning "
            "und einzelne category+Model-Tiering Queries. "
            "Keine generischen Phrasen ohne Spezifikations-/Produktseiten-Intent."
        )
    user = (
        f"Schritt {step}/4. Erzeuge nur den angefragten Teilbereich.\n"
        f"Bisheriger Plan-Stand:\n{json.dumps(partial, ensure_ascii=False)}\n\n"
        f"product_profile:\n{context}"
    )

    p = str(provider or "ionos").strip().lower()
    if p not in {"ionos", "openai", "perplexity"}:
        p = "ionos"

    if p in {"openai", "perplexity"}:
        client = LlmOpenai() if p == "openai" else LlmPerplexity()
        if not client.enabled():
            warnings.append(f"{p} not configured; staged plan generation skipped at step {step}.")
            return {}
        fmt = {
            "type": "json_schema",
            "name": f"analysis_plan_step_{step}",
            "schema": schema,
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
            return _parse_json_strictish(_openai_extract_output_text(resp))
        except Exception as exc:
            warnings.append(f"{p} staged analysis plan step {step} failed: {exc}")
            return {}

    client_i = IonosLLM()
    if not client_i.enabled():
        warnings.append(f"IONOS not configured; staged plan generation skipped at step {step}.")
        return {}

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": f"analysis_plan_step_{step}",
            "schema": schema,
            "strict": True,
        },
    }
    try:
        completion = client_i.chat_completions(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=response_format,
        )
        parsed = _parse_json_strictish(client_i.extract_text(completion))
        if parsed:
            return parsed
        completion2 = client_i.chat_completions(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
        )
        return _parse_json_strictish(client_i.extract_text(completion2))
    except Exception as exc:
        warnings.append(f"IONOS staged analysis plan step {step} failed: {exc}")
        return {}


def generate_adaptive_analysis_plan_v0_2(
    *,
    product_profile: Dict[str, Any] | None,
    product_profile_path: str | None,
    provider: str = "ionos",
    max_context_chars: int = 14000,
    user_root,
    work_root,
) -> AnalysisPlan:
    profile = _load_product_profile(
        product_profile=product_profile,
        product_profile_path=product_profile_path,
        user_root=user_root,
        work_root=work_root,
    )
    warnings = [str(w).strip() for w in (profile.get("extraction_warnings") or []) if str(w).strip()]

    perf = profile.get("performance_parameters") or []
    soft = profile.get("soft_features") or []
    restricted_feature_names: List[str] = []
    for f in perf:
        if isinstance(f, dict):
            n = str(f.get("name") or "").strip()
            if n:
                restricted_feature_names.append(n)
    for s in soft:
        if isinstance(s, dict):
            n = str(s.get("name") or "").strip()
            if n:
                restricted_feature_names.append(n)
    restricted_feature_names = list(dict.fromkeys(restricted_feature_names))

    compact_profile = {
        "product_category": profile.get("product_category"),
        "metadata": profile.get("metadata"),
        "normalized_features": [{"name": n} for n in restricted_feature_names],
        "performance_parameters": profile.get("performance_parameters"),
        "soft_features": profile.get("soft_features"),
        "price_indicators": profile.get("price_indicators"),
        "claims": profile.get("claims"),
        "differentiators": profile.get("differentiators"),
        "target_segments": profile.get("target_segments"),
        "use_cases": profile.get("use_cases"),
    }
    context = json.dumps(compact_profile, ensure_ascii=False)[:max_context_chars]

    staged: Dict[str, Any] = {}
    for step in (1, 2, 3, 4):
        part = _llm_step(provider, step=step, context=context, partial=staged, warnings=warnings)
        if isinstance(part, dict) and part:
            staged.update(part)

    base_dimensions = _base_dimensions()
    if staged:
        try:
            # Force extended_feature_schema source to performance_parameters + soft_features only.
            profile_for_plan = dict(profile)
            profile_for_plan["normalized_features"] = [{"name": n} for n in restricted_feature_names]
            llm_data_for_plan = dict(staged)
            llm_data_for_plan.pop("extended_feature_schema", None)
            plan = _sanitize_plan(
                profile=profile_for_plan,
                provider=provider,
                warnings=warnings,
                base_dimensions=base_dimensions,
                llm_data=llm_data_for_plan,
            )
            # Force-add performance and soft-feature search terms.
            perf_terms: List[str] = []
            for f in (profile.get("performance_parameters") or []):
                if isinstance(f, dict):
                    n = str(f.get("name") or "").strip()
                    if n and not _is_measure_term(n):
                        perf_terms.append(n)
            soft_terms: List[str] = []
            for s in (profile.get("soft_features") or []):
                if isinstance(s, dict):
                    n = str(s.get("name") or "").strip()
                    if n:
                        soft_terms.append(_generic_soft_feature_name(n))
            perf_terms = list(dict.fromkeys(perf_terms))[:8]
            soft_terms = list(dict.fromkeys(soft_terms))[:8]
            price_terms: List[str] = []
            for p_item in (profile.get("price_indicators") or []):
                if isinstance(p_item, dict):
                    ctx = str(p_item.get("context") or "").strip()
                    cur = str(p_item.get("currency") or "").strip()
                    pval = p_item.get("value")
                    vtxt = str(pval).strip() if pval is not None else ""
                    raw = str(p_item.get("raw") or "").strip()
                    if ctx and vtxt and cur:
                        price_terms.append(f"{ctx} {vtxt} {cur}".strip())
                    elif vtxt and cur:
                        price_terms.append(f"Preis {vtxt} {cur}".strip())
                    elif ctx and cur:
                        price_terms.append(f"{ctx} {cur}".strip())
                    elif ctx and vtxt:
                        price_terms.append(f"{ctx} {vtxt}".strip())
                    elif cur:
                        price_terms.append(f"Preis {cur}".strip())
                    elif ctx:
                        price_terms.append(ctx)
                    elif raw:
                        price_terms.append("Preis")
            if not price_terms:
                price_terms.append("Preis")
            price_terms = list(dict.fromkeys(price_terms))[:3]

            merged_terms: List[SearchTerm] = list(plan.search_terms or [])
            # Normalize any existing soft_feature terms from LLM output to generic labels.
            normalized_existing: List[SearchTerm] = []
            for st in merged_terms:
                if str(st.intent or "").strip().lower() == "soft_feature":
                    normalized_existing.append(
                        SearchTerm(term=_generic_soft_feature_name(str(st.term or "")), intent="soft_feature")
                    )
                else:
                    normalized_existing.append(st)
            merged_terms = normalized_existing
            for t in price_terms:
                merged_terms.append(SearchTerm(term=t, intent="price"))
            for t in perf_terms:
                merged_terms.append(SearchTerm(term=t, intent="performance"))
            for t in soft_terms:
                merged_terms.append(SearchTerm(term=t, intent="soft_feature"))
            dedup: Dict[tuple[str, str], SearchTerm] = {}
            for st in merged_terms:
                key = (str(st.term or "").strip().lower(), str(st.intent or "").strip().lower())
                if not key[0]:
                    continue
                dedup[key] = st
            plan.search_terms = list(dedup.values())

            # Fallback pool from step-3 LLM output (used only if web-first is insufficient).
            raw_additional = staged.get("additional_search_terms") if isinstance(staged.get("additional_search_terms"), list) else []
            fallback_mt_terms: List[SearchTerm] = []
            fallback_bt_terms: List[SearchTerm] = []
            fallback_mp_terms: List[SearchTerm] = []
            for item in raw_additional:
                if not isinstance(item, dict):
                    continue
                term = str(item.get("term") or "").strip()
                intent = str(item.get("intent") or "").strip()
                if not term or not intent:
                    continue
                if intent == "Model-Tiering":
                    fallback_mt_terms.append(SearchTerm(term=term, intent="Model-Tiering"))
                elif intent == "Brand-Tiering":
                    fallback_bt_terms.append(SearchTerm(term=term, intent="Brand-Tiering"))
                elif intent == "Market positioning":
                    t = term.lower()
                    if t in {"premium", "midrange", "budget"}:
                        canonical = "Premium" if t == "premium" else ("Midrange" if t == "midrange" else "Budget")
                        fallback_mp_terms.append(SearchTerm(term=canonical, intent="Market positioning"))

            # Web-first generation for additional_search_terms.
            mt_dedup: Dict[str, SearchTerm] = {}
            bt_dedup: Dict[str, SearchTerm] = {}
            mp_value: SearchTerm | None = None
            queries = [
                f"{plan.product_category} Top Modelle",
                f"{plan.product_category} flagship premium models",
                f"{plan.product_category} premium midrange budget segment",
            ]
            lines: List[str] = []
            for q in queries:
                results = _quick_web_search(provider, q, per_query_results=6)
                for r in results[:6]:
                    title = str(r.get("title") or "").strip()
                    snippet = str(r.get("snippet") or "").strip()
                    url = str(r.get("url") or "").strip()
                    if title or snippet:
                        lines.append(f"- {title} | {snippet} | {url}")
            if lines:
                web_terms = _llm_additional_terms_from_web(
                    provider=provider,
                    category=plan.product_category,
                    web_context="\n".join(lines[:24]),
                    warnings=plan.extraction_warnings,
                )
                for st in web_terms:
                    if st.intent == "Model-Tiering":
                        k = st.term.lower()
                        if k not in mt_dedup and len(mt_dedup) < 10:
                            mt_dedup[k] = SearchTerm(term=st.term, intent="Model-Tiering")
                    elif st.intent == "Brand-Tiering":
                        k = st.term.lower()
                        if k not in bt_dedup and len(bt_dedup) < 8:
                            bt_dedup[k] = SearchTerm(term=st.term, intent="Brand-Tiering")
                    elif st.intent == "Market positioning":
                        t = st.term.lower()
                        if t in {"premium", "midrange", "budget"} and mp_value is None:
                            canonical = "Premium" if t == "premium" else ("Midrange" if t == "midrange" else "Budget")
                            mp_value = SearchTerm(term=canonical, intent="Market positioning")
            else:
                plan.extraction_warnings = list(
                    dict.fromkeys((plan.extraction_warnings or []) + ["v0.2 web-first additional_search_terms produced no web snippets."])
                )

            # Fallback to LLM step-3 knowledge if web-first is insufficient.
            if len(mt_dedup) < 4:
                for st in fallback_mt_terms:
                    k = st.term.lower()
                    if k in mt_dedup:
                        continue
                    mt_dedup[k] = st
                    if len(mt_dedup) >= 10:
                        break
            if len(bt_dedup) < 3:
                for st in fallback_bt_terms:
                    k = st.term.lower()
                    if k in bt_dedup:
                        continue
                    bt_dedup[k] = st
                    if len(bt_dedup) >= 8:
                        break
            if mp_value is None and fallback_mp_terms:
                mp_value = fallback_mp_terms[0]

            additional_terms: List[SearchTerm] = list(mt_dedup.values()) + list(bt_dedup.values())
            if mp_value is not None:
                additional_terms.append(mp_value)
            else:
                plan.extraction_warnings = list(
                    dict.fromkeys((plan.extraction_warnings or []) + ["v0.2 missing 'Market positioning' in web-first and fallback additional_search_terms."])
                )
            plan.additional_search_terms = additional_terms

            # Build priority queries: category + price/performance/soft_feature first.
            raw_llm_queries = staged.get("search_queries") if isinstance(staged.get("search_queries"), list) else []
            llm_queries = _sanitize_llm_queries([str(q) for q in raw_llm_queries], min_queries=6)
            max_q = min(20, max(16, len(plan.search_queries) if plan.search_queries else 16))
            fallback_queries = _build_priority_queries(
                category=plan.product_category or profile.get("product_category") or "Produkt",
                price_terms=price_terms,
                perf_terms=perf_terms,
                soft_terms=soft_terms,
                llm_queries=llm_queries,
                max_queries=max_q,
            )
            # Primary query set:
            # - one query per Brand-Tiering term combined with market positioning
            # - Model-Tiering is intentionally NOT used in search_queries
            cat_main = _primary_category_term(plan.product_category or profile.get("product_category") or "Produkt")
            bt_list = [x.term for x in (plan.additional_search_terms or []) if str(x.intent or "") == "Brand-Tiering" and str(x.term or "").strip()]
            mp_value = next((x.term for x in (plan.additional_search_terms or []) if str(x.intent or "") == "Market positioning" and str(x.term or "").strip()), "")
            primary_brand_queries: List[str] = []
            for bt in bt_list:
                q = f"{cat_main} {bt} {mp_value}".strip() if mp_value else f"{cat_main} {bt}".strip()
                primary_brand_queries.append(q)
            primary_brand_queries = _dedupe_keep_order(primary_brand_queries)

            # Add exactly 12 fallback queries (previous pattern), excluding size/weight/dimension-related ones.
            filtered_fallback = [q for q in fallback_queries if not _is_size_weight_query(q)]
            seen = {q.lower() for q in primary_brand_queries}
            add_12: List[str] = []
            for q in filtered_fallback:
                k = q.lower()
                if k in seen:
                    continue
                seen.add(k)
                add_12.append(q)
                if len(add_12) >= 12:
                    break

            plan.search_queries = (primary_brand_queries + add_12)[:20]
            if len(add_12) < 12:
                plan.extraction_warnings = list(
                    dict.fromkeys((plan.extraction_warnings or []) + [f"v0.2 only {len(add_12)}/12 fallback queries available after excluding size/weight/dimension terms."])
                )
            if not llm_queries:
                plan.extraction_warnings = list(
                    dict.fromkeys((plan.extraction_warnings or []) + ["v0.2 step-3 LLM search_queries empty/invalid; fallback query builder used."])
                )
            if plan.comparison_dimensions and plan.search_queries:
                plan.extraction_warnings = list(dict.fromkeys((plan.extraction_warnings or []) + ["Generated via staged LLM plan v0.2 (4-step merge)."] ))
                return plan
        except Exception as exc:
            warnings.append(f"staged output validation failed; heuristic plan used ({exc}).")

    fallback = _heuristic_plan(profile=profile, provider=provider, warnings=warnings)
    fallback.extraction_warnings = list(dict.fromkeys((fallback.extraction_warnings or []) + ["Staged LLM plan v0.2 fell back to heuristic plan."]))
    return fallback
