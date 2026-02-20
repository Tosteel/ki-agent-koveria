from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from server.services.llm_ionos import IonosLLM
from server.services.llm_openai import LlmOpenai
from server.services.llm_perplexity import LlmPerplexity

from .models import AnalysisPlan, AnalysisScope, ComparisonDimension, SearchTerm


def _parse_json_strictish(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    t = text.strip()

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

    start = t.find("{")
    end = t.rfind("}")
    if start >= 0 and end > start:
        snippet = t[start : end + 1]
        try:
            obj = json.loads(snippet)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _safe_list_str(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
    for v in values:
        s = str(v or "").strip()
        if s:
            out.append(s)
    return out


def _norm_text(v: Any) -> str:
    return re.sub(r"\s+", " ", str(v or "")).strip()


def _clean_category(raw: Any) -> str:
    c = _norm_text(raw)
    if not c:
        return "unknown"
    generic = {
        "unknown",
        "produkt",
        "product",
        "artikel",
        "item",
        "fahrzeug",
        "vehicle",
        "system",
        "lösung",
        "loesung",
    }
    if c.lower() in generic:
        return "unknown"
    return c


def _is_noisy_feature_term(term: str) -> bool:
    t = _norm_text(term)
    if not t:
        return True
    if len(t) < 3 or len(t) > 64:
        return True
    if t.startswith(("•", "-", "–", "*")):
        return True
    if any(ch in t for ch in ["\u0000", "\uFFFD"]):
        return True
    # Avoid fragments/clauses and highly specific sentence pieces.
    banned_fragments = [
        "inklusive",
        "bis zu",
        "datenblatt",
        "konfigurationsspezifisch",
        "mit ",
        " fuer ",
        " für ",
        ",",
    ]
    tl = f" {t.lower()} "
    if any(frag in tl for frag in banned_fragments):
        return True
    # Mostly numeric/units is not a stable search schema feature name.
    if re.fullmatch(r"[\d\W]+", t):
        return True
    return False


def _clean_feature_terms(values: List[str], max_items: int = 20) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for raw in values:
        t = _norm_text(raw)
        if _is_noisy_feature_term(t):
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
        if len(out) >= max_items:
            break
    return out


def _build_competitor_queries(
    *,
    product_name: str,
    manufacturer: str,
    product_category: str,
    min_queries: int = 6,
) -> List[str]:
    name = _norm_text(product_name)
    brand = _norm_text(manufacturer)
    category = _clean_category(product_category)

    anchor = name or brand
    if not anchor and category != "unknown":
        anchor = category
    if not anchor:
        anchor = "produkt"

    # Keep queries competitor-focused on the main product/market class.
    queries: List[str] = [
        f"Alternativen zu {anchor}",
        f"Wettbewerber von {anchor}",
        f"{anchor} Konkurrenz Vergleich",
        f"{anchor} vs Konkurrenzmodelle",
        f"{anchor} competitor alternatives",
        f"{anchor} competing products",
    ]

    if category != "unknown":
        queries.extend(
            [
                f"{category} Wettbewerber Vergleich",
                f"{category} ähnliche Produkte zu {anchor}",
                f"{category} Konkurrenzmodelle Datenblatt",
            ]
        )

    if brand and name:
        queries.extend(
            [
                f"{brand} {name} competitors",
                f"{brand} {name} alternatives",
            ]
        )

    deduped = list(dict.fromkeys([_norm_text(q) for q in queries if _norm_text(q)]))
    if len(deduped) < min_queries and category != "unknown":
        deduped.append(f"{category} Marktvergleich")
    return deduped[:12]


def _sanitize_plan(
    *,
    profile: Dict[str, Any],
    provider: str,
    warnings: List[str],
    base_dimensions: List[ComparisonDimension],
    llm_data: Optional[Dict[str, Any]] = None,
) -> AnalysisPlan:
    p = str(provider or "ionos").strip().lower()
    if p not in {"ionos", "openai", "perplexity"}:
        p = "ionos"

    metadata = profile.get("metadata") if isinstance(profile.get("metadata"), dict) else {}
    product_name = _norm_text(metadata.get("product_name"))
    manufacturer = _norm_text(metadata.get("manufacturer"))

    profile_category = _clean_category(profile.get("product_category"))
    llm_category = _clean_category((llm_data or {}).get("product_category"))
    product_category = llm_category if llm_category != "unknown" else profile_category
    if product_category == "unknown":
        # Product-independent fallback without domain-specific assumptions.
        product_category = _norm_text(profile.get("product_category")) or "Produkt"

    raw_features = []
    for f in (profile.get("normalized_features") or []):
        if isinstance(f, dict):
            raw_features.append(f.get("name"))
    if llm_data:
        raw_features.extend(_safe_list_str(llm_data.get("extended_feature_schema")))
    clean_features = _clean_feature_terms([_norm_text(x) for x in raw_features if _norm_text(x)], max_items=20)

    if not clean_features:
        clean_features = [
            "Leistung",
            "Kapazität",
            "Effizienz",
            "Abmessungen",
            "Gewicht",
            "Betriebstemperatur",
            "Compliance",
            "Service & Wartung",
        ]

    llm_terms = [t for t in ((llm_data or {}).get("search_terms") or []) if isinstance(t, dict)]
    search_terms: List[SearchTerm] = []
    if product_name:
        search_terms.append(SearchTerm(term=product_name, intent="exact_product"))
    if manufacturer:
        search_terms.append(SearchTerm(term=manufacturer, intent="brand"))
    if product_category and product_category.lower() != "unknown":
        search_terms.append(SearchTerm(term=product_category, intent="category"))

    # Keep segment/use-case terms; avoid noisy feature terms as search anchors.
    for s in _safe_list_str(profile.get("target_segments"))[:4]:
        search_terms.append(SearchTerm(term=s, intent="segment"))
    for u in _safe_list_str(profile.get("use_cases"))[:4]:
        search_terms.append(SearchTerm(term=u, intent="use_case"))
    for t in llm_terms[:8]:
        intent = _norm_text(t.get("intent")).lower() or "generic"
        term = _norm_text(t.get("term"))
        if not term:
            continue
        if intent == "feature" and _is_noisy_feature_term(term):
            continue
        if intent in {"exact_product", "brand", "category", "segment", "use_case"}:
            search_terms.append(SearchTerm(term=term, intent=intent))

    # Deduplicate terms.
    search_terms = list({(st.term.lower(), st.intent): st for st in search_terms if st.term}.values())

    queries = _build_competitor_queries(
        product_name=product_name,
        manufacturer=manufacturer,
        product_category=product_category,
    )

    comp_dims = []
    if llm_data:
        comp_dims = [ComparisonDimension(**d) for d in (llm_data.get("comparison_dimensions") or []) if isinstance(d, dict)]
    if not comp_dims:
        comp_dims = base_dimensions

    relevance_criteria = _safe_list_str((llm_data or {}).get("relevance_criteria")) if llm_data else []
    if not relevance_criteria:
        relevance_criteria = [
            "Gleiche oder sehr ähnliche Zielgruppe",
            "Ähnliches Preisniveau (±20%) oder klar dokumentierte Preissegmente",
            "Vergleichbare Kernleistung (z. B. Output, Kapazität, Effizienz)",
            "Mindestens 60% Feature-Overlap mit dem Zielprodukt",
            "Gleiche Region oder nachweisbar bediente Zielmärkte",
        ]

    scope = AnalysisScope(depth="medium", breadth="medium", include_regional=True, include_global=True, max_results_per_query=20)
    if llm_data and isinstance(llm_data.get("analysis_scope"), dict):
        try:
            scope = AnalysisScope(**llm_data.get("analysis_scope"))
        except Exception:
            warnings.append("analysis_scope from LLM invalid; fallback used.")

    notes = _safe_list_str((llm_data or {}).get("notes")) if llm_data else []
    notes.append("Search queries werden auf Wettbewerber zum Hauptprodukt fokussiert; Feature-Noise wird unterdrückt.")
    notes = list(dict.fromkeys([n for n in notes if _norm_text(n)]))

    min_competitors = int((llm_data or {}).get("min_competitors") or 6)
    min_competitors = max(6, min(50, min_competitors))

    return AnalysisPlan(
        provider=p,
        product_category=product_category,
        comparison_dimensions=comp_dims,
        extended_feature_schema=clean_features,
        search_terms=search_terms,
        search_queries=queries,
        min_competitors=min_competitors,
        relevance_criteria=relevance_criteria,
        analysis_scope=scope,
        notes=notes,
        extraction_warnings=warnings,
    )


def _resolve_input_path(path: str, user_root: Path, work_root: Path) -> Path:
    raw = str(path or "").strip().lstrip("/")
    p = Path(raw)
    if not raw or p.is_absolute() or ".." in p.parts:
        raise HTTPException(status_code=400, detail=f"Invalid path: {path}")

    uploads_root = (user_root / "uploads").resolve()
    uploads_root.mkdir(parents=True, exist_ok=True)

    candidates: list[Path] = []
    parts = list(p.parts)
    if parts and parts[0] == "uploads":
        candidates.append((uploads_root / Path(*parts[1:])).resolve())
    elif parts and parts[0] == "work":
        candidates.append((work_root / Path(*parts[1:])).resolve())
    else:
        candidates.append((work_root / p).resolve())
        candidates.append((uploads_root / p).resolve())

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            if user_root in candidate.parents or candidate == user_root:
                return candidate

    raise HTTPException(status_code=404, detail=f"File not found: {path}")


def _load_product_profile(
    *,
    product_profile: Optional[Dict[str, Any]],
    product_profile_path: Optional[str],
    user_root: Path,
    work_root: Path,
) -> Dict[str, Any]:
    payload: Dict[str, Any]
    if isinstance(product_profile, dict) and product_profile:
        payload = product_profile
    else:
        p = _resolve_input_path(str(product_profile_path or ""), user_root=user_root.resolve(), work_root=work_root.resolve())
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON in product_profile_path: {product_profile_path}") from exc

    if "product_profile" in payload and isinstance(payload.get("product_profile"), dict):
        payload = payload["product_profile"]

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid product profile payload")
    return payload


def _semantic_schema() -> Dict[str, Any]:
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
            "extended_feature_schema": {"type": "array", "items": {"type": "string"}},
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
            "search_queries": {"type": "array", "items": {"type": "string"}},
            "min_competitors": {"type": "integer"},
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
                "required": [
                    "depth",
                    "breadth",
                    "include_regional",
                    "include_global",
                    "max_results_per_query",
                ],
            },
            "notes": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "product_category",
            "comparison_dimensions",
            "extended_feature_schema",
            "search_terms",
            "search_queries",
            "min_competitors",
            "relevance_criteria",
            "analysis_scope",
            "notes",
        ],
    }


def _openai_extract_output_text(resp: Dict[str, Any]) -> str:
    out = ""
    for item in resp.get("output", []):
        for c in item.get("content", []):
            if c.get("type") == "output_text":
                out += str(c.get("text") or "")
    return out.strip()


def _llm_plan(provider: str, context: str, warnings: List[str]) -> Dict[str, Any]:
    schema = _semantic_schema()
    system = (
        "Du erzeugst einen adaptiven Wettbewerbsanalyse-Plan für ein Produktprofil. "
        "Antworte strikt als JSON gemäß Schema. "
        "Definiere Vergleichsdimensionen, Suchstrategie, Relevanzkriterien und Analyseumfang."
    )
    user = (
        "Erstelle einen pragmatischen Analyseplan basierend auf folgendem product_profile. "
        "Gewichte Dimensionen sinnvoll und gib konkrete Suchbegriffe/Queries aus.\n\n"
        f"product_profile:\n{context}"
    )

    p = str(provider or "ionos").strip().lower()
    if p not in {"ionos", "openai", "perplexity"}:
        p = "ionos"

    if p in {"openai", "perplexity"}:
        client = LlmOpenai() if p == "openai" else LlmPerplexity()
        if not client.enabled():
            warnings.append(f"{p} not configured; heuristic analysis plan used.")
            return {}
        fmt = {
            "type": "json_schema",
            "name": "analysis_plan",
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
            warnings.append(f"{p} analysis plan generation failed: {exc}")
            return {}

    client_i = IonosLLM()
    if not client_i.enabled():
        warnings.append("IONOS not configured; heuristic analysis plan used.")
        return {}

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "analysis_plan",
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
        warnings.append(f"IONOS analysis plan generation failed: {exc}")
        return {}


def _heuristic_plan(profile: Dict[str, Any], provider: str, warnings: List[str]) -> AnalysisPlan:
    product_category = str(profile.get("product_category") or "unknown").strip() or "unknown"
    metadata = profile.get("metadata") if isinstance(profile.get("metadata"), dict) else {}

    feature_names = []
    for f in profile.get("normalized_features") or []:
        if isinstance(f, dict):
            name = str(f.get("name") or "").strip()
            if name:
                feature_names.append(name)

    claim_terms = []
    for c in profile.get("claims") or []:
        if isinstance(c, dict):
            txt = str(c.get("text") or "").strip()
            if txt:
                claim_terms.append(txt)

    base_dimensions = [
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

    product_name = str(metadata.get("product_name") or "").strip()
    manufacturer = str(metadata.get("manufacturer") or "").strip()

    llm_seed = {
        "product_category": product_category,
        "comparison_dimensions": [d.model_dump() for d in base_dimensions],
        "extended_feature_schema": feature_names,
        "notes": ["Claims aus product_profile für Positionierungsvergleich berücksichtigen."] if claim_terms else [],
    }
    return _sanitize_plan(
        profile=profile,
        provider=provider,
        warnings=warnings,
        base_dimensions=base_dimensions,
        llm_data=llm_seed,
    )


def generate_adaptive_analysis_plan(
    *,
    product_profile: Optional[Dict[str, Any]],
    product_profile_path: Optional[str],
    provider: str = "ionos",
    max_context_chars: int = 14000,
    user_root: Path,
    work_root: Path,
) -> AnalysisPlan:
    profile = _load_product_profile(
        product_profile=product_profile,
        product_profile_path=product_profile_path,
        user_root=user_root,
        work_root=work_root,
    )

    warnings = _safe_list_str(profile.get("extraction_warnings"))

    compact_profile = {
        "product_category": profile.get("product_category"),
        "metadata": profile.get("metadata"),
        "normalized_features": profile.get("normalized_features"),
        "performance_parameters": profile.get("performance_parameters"),
        "price_indicators": profile.get("price_indicators"),
        "claims": profile.get("claims"),
        "differentiators": profile.get("differentiators"),
        "target_segments": profile.get("target_segments"),
        "use_cases": profile.get("use_cases"),
    }

    context = json.dumps(compact_profile, ensure_ascii=False)[:max_context_chars]
    llm_data = _llm_plan(provider=provider, context=context, warnings=warnings)

    if llm_data:
        try:
            base_dimensions = [
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
            plan = _sanitize_plan(
                profile=profile,
                provider=provider,
                warnings=warnings,
                base_dimensions=base_dimensions,
                llm_data=llm_data,
            )
            if plan.comparison_dimensions and plan.search_queries:
                return plan
        except Exception as exc:
            warnings.append(f"Structured LLM output validation failed; heuristic plan used ({exc}).")

    return _heuristic_plan(profile=profile, provider=provider, warnings=warnings)
