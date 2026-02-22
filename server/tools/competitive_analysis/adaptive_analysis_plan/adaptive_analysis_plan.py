from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from server.services.llm_ionos import IonosLLM
from server.services.llm_openai import LlmOpenai
from server.services.llm_perplexity import LlmPerplexity

from .models import AnalysisPlan, AnalysisScope, ComparisonDimension, SearchTerm

_TARGET_SEARCH_QUERIES = 16


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


def _compact_category_phrase(raw: str) -> str:
    c = _norm_text(raw)
    if not c:
        return "Produkt"
    # Keep category query-friendly and short.
    c = re.sub(r"\s+", " ", c).strip(" -_,.;:")
    words = c.split()
    if len(words) > 8:
        c = " ".join(words[:8])
    return c


def _strip_parenthetical(text: str) -> str:
    s = _norm_text(text)
    if not s:
        return ""
    # Remove technical/noisy parenthetical qualifiers in category anchor,
    # e.g. "(0.25-0.72 kW)".
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"\s{2,}", " ", s).strip(" -_,.;:/")
    return s


def _split_category_variants(category_phrase: str, max_items: int = 3) -> List[str]:
    """
    Split broad category phrases into concise alternatives to avoid
    repeated identical query prefixes.
    Example: "Faltbares E-Bike / E-Klapprad" -> ["Faltbares E-Bike", "E-Klapprad"].
    """
    raw = _strip_parenthetical(category_phrase) or _norm_text(category_phrase)
    if not raw:
        return ["Produkt"]

    normalized = re.sub(r"\s*/\s*", " | ", raw)
    normalized = re.sub(r"\s+\|\s+", " | ", normalized)
    parts = [p.strip(" -_,.;:") for p in normalized.split("|")]

    # Also split very broad conjunction forms.
    expanded: List[str] = []
    for p in parts:
        if not p:
            continue
        # Keep short compounds, split only if clearly two category alternatives.
        if re.search(r"\b(?:und|or|oder)\b", p, flags=re.IGNORECASE) and len(p.split()) >= 4:
            sub = re.split(r"\b(?:und|or|oder)\b", p, flags=re.IGNORECASE)
            for x in sub:
                xx = _compact_category_phrase(_norm_text(x))
                if xx:
                    expanded.append(xx)
        else:
            expanded.append(_compact_category_phrase(p))

    out: List[str] = []
    seen: set[str] = set()
    for p in expanded:
        s = _norm_text(p)
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= max_items:
            break
    return out or [_compact_category_phrase(raw)]


def _extract_power_hints(profile: Dict[str, Any], product_name: str) -> List[str]:
    hints: List[str] = []
    text_blobs: List[str] = [_norm_text(product_name)]

    for k in ("performance_parameters", "normalized_features"):
        vals = profile.get(k)
        if isinstance(vals, list):
            for item in vals:
                if not isinstance(item, dict):
                    continue
                n = _norm_text(item.get("name"))
                v = _norm_text(item.get("value"))
                u = _norm_text(item.get("unit"))
                if n:
                    text_blobs.append(n)
                if v:
                    text_blobs.append(f"{v} {u}".strip())

    raw = " | ".join([x for x in text_blobs if x])

    # Pattern like 5/6/8/10K or 5-10kW.
    slash_nums = re.findall(r"(?<!\d)(\d{1,2}(?:[.,]\d)?)\s*/\s*(\d{1,2}(?:[.,]\d)?)(?:\s*/\s*(\d{1,2}(?:[.,]\d)?))?(?:\s*/\s*(\d{1,2}(?:[.,]\d)?))?\s*[kK]\b", raw)
    for grp in slash_nums:
        nums = [x for x in grp if x]
        try:
            vals = sorted({float(x.replace(",", ".")) for x in nums})
            if vals:
                hints.append(f"{vals[0]:g}-{vals[-1]:g} kW")
        except Exception:
            continue

    range_kw = re.findall(r"(\d{1,2}(?:[.,]\d)?)\s*[-–]\s*(\d{1,2}(?:[.,]\d)?)\s*[kK][wW]?", raw)
    for lo, hi in range_kw:
        try:
            l = float(lo.replace(",", "."))
            h = float(hi.replace(",", "."))
            if l <= h:
                hints.append(f"{l:g}-{h:g} kW")
        except Exception:
            continue

    # Derive coarse band from W values in performance fields.
    w_values: List[float] = []
    for entry in (profile.get("performance_parameters") or []):
        if not isinstance(entry, dict):
            continue
        name = _norm_text(entry.get("name")).lower()
        unit = _norm_text(entry.get("normalized_unit") or entry.get("unit")).lower()
        val = entry.get("normalized_value")
        if val is None:
            val = entry.get("value")
        if val is None:
            continue
        try:
            fv = float(val)
        except Exception:
            continue
        if unit in {"w", "kw", "va", "kva"} and any(tok in name for tok in ("leistung", "power", "output", "nenn", "rated", "nominal", "max", "pv", "ac")):
            # Convert to kW-like value for rough query anchor.
            if unit in {"w", "va"}:
                fv = fv / 1000.0
            w_values.append(fv)

    # Generic fallback: also scan normalized_features for electrical power units.
    for entry in (profile.get("normalized_features") or []):
        if not isinstance(entry, dict):
            continue
        unit = _norm_text(entry.get("normalized_unit") or entry.get("unit")).lower()
        val = entry.get("normalized_value")
        if val is None:
            val = entry.get("value")
        if val is None or unit not in {"w", "kw", "va", "kva"}:
            continue
        try:
            fv = float(val)
        except Exception:
            continue
        if unit in {"w", "va"}:
            fv = fv / 1000.0
        if fv > 0:
            w_values.append(fv)

    if w_values:
        lo = min(w_values)
        hi = max(w_values)
        if lo > 0 and hi > 0:
            hints.append(f"{lo:g}-{hi:g} kW")

    deduped = list(dict.fromkeys([_norm_text(h) for h in hints if _norm_text(h)]))
    return deduped[:3]


def _pick_primary_power_hint(hints: List[str]) -> str:
    """
    Prefer compact ranges (e.g. 5-10 kW) and avoid overly broad/noisy ranges.
    """
    best = ""
    best_span = 9999.0
    for h in hints:
        m = re.search(r"(\d+(?:[.,]\d+)?)\s*-\s*(\d+(?:[.,]\d+)?)\s*kW", str(h), flags=re.IGNORECASE)
        if not m:
            continue
        lo = float(m.group(1).replace(",", "."))
        hi = float(m.group(2).replace(",", "."))
        if lo <= 0 or hi <= 0 or hi < lo:
            continue
        span = hi - lo
        # Generic guard: keep only realistic, focused residential/commercial bands.
        if hi > 30 or span > 12:
            continue
        if span < best_span:
            best_span = span
            best = f"{lo:g}-{hi:g} kW"
    return best


def _extract_functionality_query_terms(profile: Dict[str, Any], max_terms: int = 4) -> List[str]:
    raw_terms: List[str] = []
    for f in (profile.get("normalized_features") or []):
        if isinstance(f, dict):
            n = _norm_text(f.get("name"))
            if n:
                raw_terms.append(n)
    raw_terms.extend(_safe_list_str(profile.get("differentiators")))
    raw_terms.extend(_safe_list_str(profile.get("use_cases")))

    out: List[str] = []
    seen: set[str] = set()
    for t in raw_terms:
        s = _norm_text(t)
        if not s:
            continue
        # Prefer trailing functional phrase where present.
        m = re.search(r"\b(?:zur|für|for)\s+([A-Za-zÄÖÜäöüß][A-Za-zÄÖÜäöüß\- ]{2,})$", s, flags=re.IGNORECASE)
        if m:
            s = _norm_text(m.group(1))
        s = re.sub(r"[™®©]", "", s)
        s = re.sub(r"\s+", " ", s).strip(" -_,.;:")
        if len(s) < 5 or len(s) > 40:
            continue
        if re.search(r"\d", s):
            continue
        if len(re.findall(r"[A-Za-zÄÖÜäöüß]", s)) < 4:
            continue
        # Drop mostly spec-ish terms for this specific query expansion.
        if any(k in s.lower() for k in ("gewicht", "maß", "abmess", "spannung", "strom", "leistung", "kapaz")):
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
        if len(out) >= max_terms:
            break
    return out


def _extract_functional_signature(profile: Dict[str, Any], max_parts: int = 4) -> List[str]:
    """
    Build compact, high-signal functional signature parts from normalized features:
    product-type-neutral but useful for search recall/precision.
    Example outcome: ["lithium-ion", "6000 mAh", "rechargeable", "usb-c"].
    """
    parts: List[str] = []
    seen: set[str] = set()

    def _push(v: str) -> None:
        s = _norm_text(v).lower()
        if not s or s in seen:
            return
        seen.add(s)
        parts.append(v.strip())

    for f in (profile.get("normalized_features") or []):
        if not isinstance(f, dict):
            continue
        name = _norm_text(f.get("name")).lower()
        value_txt = _norm_text(f.get("value"))
        unit_txt = _norm_text(f.get("unit"))

        # Chemistry / cell type
        if any(k in name for k in ("chemistry", "chemie", "cell chemistry")):
            v = value_txt.lower()
            if any(k in v for k in ("li-ion", "lithium-ion", "lithium ion", "li ion")):
                _push("lithium-ion")
            elif "lifepo4" in v:
                _push("lifepo4")
            elif v:
                _push(value_txt)

        # Rechargeable signal
        if any(k in name for k in ("recharge", "auflad", "ladbar", "charging cycles")):
            _push("rechargeable")

        # Capacity signal
        if any(k in name for k in ("capacity", "kapazit")):
            if value_txt:
                if unit_txt:
                    _push(f"{value_txt} {unit_txt}")
                else:
                    _push(value_txt)

        # Port / connector signal
        if any(k in name for k in ("port", "anschluss", "connector")):
            v = value_txt.lower()
            if "usb-c" in v or "usb c" in v or "type-c" in v or "type c" in v:
                _push("usb-c")
            elif v:
                _push(value_txt)

        if len(parts) >= max_parts:
            break

    # Fallback from claims/differentiators for functional terms.
    text_blob = " | ".join(
        _safe_list_str(profile.get("differentiators"))[:8]
        + _safe_list_str(profile.get("use_cases"))[:8]
    ).lower()
    if text_blob:
        if "recharge" in text_blob or "auflad" in text_blob:
            _push("rechargeable")
        if "usb-c" in text_blob or "usb c" in text_blob:
            _push("usb-c")
        if "lithium-ion" in text_blob or "li-ion" in text_blob or "lithium ion" in text_blob:
            _push("lithium-ion")

    return parts[:max_parts]


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
    # Truncated/fractured endings are typically OCR/table-join artifacts.
    if t.endswith((" -", " –", "/", ":", "(", "[")):
        return True
    # Heavily concatenated measurement chains are usually low-signal features.
    num_tokens = re.findall(r"\d+(?:[.,]\d+)?", t)
    if len(num_tokens) >= 3:
        return True
    if re.search(r"(?:\d+(?:[.,]\d+)?\s*[a-zA-Z%]+.*){2,}", t):
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


def _infer_category_phrase_when_unknown(
    *,
    product_name: str,
    segments: List[str],
    use_cases: List[str],
    functionality_terms: List[str],
) -> str:
    blob = " | ".join(
        [str(product_name or "")]
        + [str(x or "") for x in (segments or [])[:8]]
        + [str(x or "") for x in (use_cases or [])[:8]]
        + [str(x or "") for x in (functionality_terms or [])[:8]]
    ).lower()

    keyword_map = [
        (("saug", "wisch", "mopp", "vacuum", "robot"), "Saug- und Wischroboter"),
        (("kaffeevollautomat", "espresso", "cappuccino", "coffee machine"), "Kaffeevollautomat"),
        (("wechselrichter", "inverter", "mppt", "hybrid-wr"), "Wechselrichter"),
        (("li-ion", "lithium-ion", "rechargeable battery", "akku", "usb-c"), "Lithium-Ionen Akku"),
        (("lüftung", "ventilation", "air handling"), "Lüftungssystem"),
    ]
    for keys, label in keyword_map:
        if any(k in blob for k in keys):
            return label
    return "Produktkategorie"


def _build_competitor_queries(
    *,
    profile: Dict[str, Any],
    product_name: str,
    manufacturer: str,
    product_category: str,
    segments: List[str],
    use_cases: List[str],
    dimension_names: List[str],
    functionality_terms: List[str],
    functional_signature: List[str],
    min_queries: int = 10,
) -> List[str]:
    name = _norm_text(product_name)
    brand = _norm_text(manufacturer)
    category = _clean_category(product_category)
    if category != "unknown":
        category_phrase = _compact_category_phrase(category)
    else:
        category_phrase = _infer_category_phrase_when_unknown(
            product_name=name,
            segments=segments,
            use_cases=use_cases,
            functionality_terms=functionality_terms,
        )
    power_hints = _extract_power_hints(profile, name)

    anchor = name or brand
    if not anchor and category != "unknown":
        anchor = category
    if not anchor:
        anchor = "produkt"

    # Anchor/model queries (limited to avoid overfitting to vendor/model naming).
    anchor_queries: List[str] = [
        f"Alternativen zu {anchor} Datenblatt",
        f"Wettbewerber von {anchor} technische Daten",
        f"{anchor} datasheet specs",
    ]

    primary_power = _pick_primary_power_hint(power_hints)
    category_anchor = category_phrase
    category_variants = _split_category_variants(category_phrase, max_items=3)
    base_terms = category_variants if category_variants else [category_anchor]

    # Category queries: concise and product-page oriented.
    category_templates = [
        "{cat} Wettbewerber Vergleich",
        "{cat} ähnliche Produkte zu {anchor}",
        "{cat} Produktserie technische Daten",
        "{cat} product page specs",
    ]
    category_queries: List[str] = []
    for idx, tmpl in enumerate(category_templates):
        cat = base_terms[idx % len(base_terms)]
        category_queries.append(tmpl.format(cat=cat, anchor=anchor).strip())

    # Technical range hint only once (avoid same technical prefix everywhere).
    if primary_power:
        category_queries.append(f"{base_terms[0]} {primary_power} technische Daten".strip())

    # Use an intent rotation so we don't generate near-duplicate suffix variants
    # for the same base term (e.g. "... Nominal voltage", "... datasheet", "... specs").
    feature_intents = [
        "{cat} {term}",
        "{cat} {term} datasheet",
        "{cat} {term} product page",
        "{cat} {term} technical specifications",
    ]
    for idx, ft in enumerate(functionality_terms[:6]):
        term = _compact_category_phrase(ft)
        if not term:
            continue
        tmpl = feature_intents[idx % len(feature_intents)]
        cat = base_terms[idx % len(base_terms)]
        category_queries.append(tmpl.format(cat=cat, term=term).strip())

    if brand and name:
        # Avoid duplicated brand prefix when product_name already includes brand.
        brand_in_name = brand.lower() in name.lower()
        brand_name_anchor = name if brand_in_name else f"{brand} {name}"
        anchor_queries.extend(
            [
                f"{brand_name_anchor} competitors",
                f"{brand_name_anchor} alternatives",
            ]
        )

    if functional_signature:
        sig = " ".join([_compact_category_phrase(x) for x in functional_signature if _compact_category_phrase(x)]).strip()
        if sig:
            category_queries.extend(
                [
                    f"{base_terms[0]} {sig}",
                    f"{base_terms[min(1, len(base_terms)-1)]} {sig} competitors",
                    f"{base_terms[0]} {sig} datasheet",
                ]
            )

    # Balanced order with guaranteed anchor reservation:
    # keep category/function focus, but always retain 2-3 brand/product anchor queries.
    max_queries = 16
    clean_category = list(dict.fromkeys([_norm_text(q) for q in category_queries if _norm_text(q)]))
    clean_anchor = list(dict.fromkeys([_norm_text(q) for q in anchor_queries if _norm_text(q)]))

    anchor_keep = 3 if len(clean_anchor) >= 3 else (2 if len(clean_anchor) >= 2 else len(clean_anchor))
    category_budget = max(0, max_queries - anchor_keep)

    # Ensure at least one explicit brand-bearing anchor query survives truncation,
    # otherwise brand-specific recall can disappear in short query sets.
    selected_anchor: List[str] = []
    if anchor_keep > 0:
        brand_q: List[str] = []
        non_brand_q: List[str] = clean_anchor[:]
        if brand:
            brand_q = [q for q in clean_anchor if brand.lower() in q.lower()]
            non_brand_q = [q for q in clean_anchor if q not in brand_q]

        if non_brand_q:
            selected_anchor.append(non_brand_q[0])
        if brand_q and len(selected_anchor) < anchor_keep:
            selected_anchor.append(brand_q[0])
        if len(brand_q) > 1 and len(selected_anchor) < anchor_keep:
            selected_anchor.append(brand_q[1])

        for q in (non_brand_q[1:] + brand_q[1:] + clean_anchor):
            if len(selected_anchor) >= anchor_keep:
                break
            if q not in selected_anchor:
                selected_anchor.append(q)

    mixed: List[str] = []
    mixed.extend(clean_category[:category_budget])
    mixed.extend(selected_anchor[:anchor_keep])
    if len(mixed) < max_queries:
        mixed.extend(clean_category[category_budget : category_budget + (max_queries - len(mixed))])

    deduped = list(dict.fromkeys([q for q in mixed if _norm_text(q)]))
    if len(deduped) < min_queries and category != "unknown":
        deduped.append(f"{category_anchor} technische Datenblatt")
    if len(deduped) < min_queries:
        deduped.append("Produkt technische Daten")
    return deduped[:max_queries]


def _compact_use_case_phrase(raw: str) -> str:
    u = _compact_category_phrase(raw)
    if not u:
        return ""
    # Remove repetitive heads to keep queries concise and reusable.
    u = re.sub(r"^(für|for)\s+", "", u, flags=re.IGNORECASE)
    u = re.sub(r"\bvon\s+\w+\b$", "", u, flags=re.IGNORECASE).strip()
    # Drop repeated category nouns in use-case phrase to avoid tautologies.
    u = re.sub(r"\b(wechselrichter|inverter|produkt|systeme?)\b", "", u, flags=re.IGNORECASE)
    u = re.sub(r"\s{2,}", " ", u).strip(" -_,.;:")
    return u or _compact_category_phrase(raw)


_QUERY_BANNED_PATTERNS = (
    r"\bsite:",
    r"\bvs\b",
    r"https?://",
)


def _sanitize_query_term(text: str) -> str:
    q = _norm_text(text)
    if not q:
        return ""
    q = q.replace("_", " ")
    # Strip internal/version suffixes like _v2, v2, final, draft.
    q = re.sub(r"\b(v\d+|ver(sion)?\s*\d+|final|draft|tmp)\b", "", q, flags=re.IGNORECASE)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def _is_generic_query_allowed(text: str) -> bool:
    q = _norm_text(text)
    if not q:
        return False
    if len(q) < 12 or len(q) > 140:
        return False
    if len(q.split()) < 3:
        return False
    ql = q.lower()
    if any(re.search(p, ql, flags=re.IGNORECASE) for p in _QUERY_BANNED_PATTERNS):
        return False
    if ql.count("?") > 1:
        return False
    years = [int(y) for y in re.findall(r"\b(20\d{2})\b", q)]
    if years:
        current_year = datetime.utcnow().year
        # Prevent stale temporal anchors (e.g., "2023") while allowing current/near years.
        if any(y < current_year - 1 for y in years):
            return False
    return True


def _sanitize_llm_queries(raw_queries: List[str], *, min_queries: int, max_queries: int = _TARGET_SEARCH_QUERIES) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for q in raw_queries:
        qq = _sanitize_query_term(q)
        if not _is_generic_query_allowed(qq):
            continue
        k = qq.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(qq)
        if len(out) >= max_queries:
            break
    return out if len(out) >= min_queries else out


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

    comp_dims = []
    if llm_data:
        comp_dims = [ComparisonDimension(**d) for d in (llm_data.get("comparison_dimensions") or []) if isinstance(d, dict)]
    if not comp_dims:
        comp_dims = base_dimensions
    comp_dims = _normalize_comparison_dimensions_required_fields(comp_dims)

    min_competitors = int((llm_data or {}).get("min_competitors") or 6)
    min_competitors = max(6, min(50, min_competitors))

    queries = _build_competitor_queries(
        profile=profile,
        product_name=product_name,
        manufacturer=manufacturer,
        product_category=product_category,
        segments=_safe_list_str(profile.get("target_segments")),
        use_cases=_safe_list_str(profile.get("use_cases")),
        dimension_names=[d.name for d in comp_dims if _norm_text(d.name)],
        functionality_terms=_extract_functionality_query_terms(profile, max_terms=4),
        functional_signature=_extract_functional_signature(profile, max_parts=4),
    )

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
    notes.append("Query-Mix priorisiert Produkt-/Serienseiten mit Datenblatt/Spezifikations-Intent.")
    notes.append("Zusätzliche Query-Varianten für funktionale Kernmerkmale werden berücksichtigt.")
    notes.append("Funktionale Signatur aus Produkttyp+Kernspezifikationen (z. B. Chemie/Kapazität/Port) wird in Queries priorisiert.")
    notes = list(dict.fromkeys([n for n in notes if _norm_text(n)]))

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


_GENERIC_REQUIRED_FIELDS = {
    "performance_parameters",
    "normalized_features",
    "extended_feature_schema",
    "price_indicators",
    "relevance_criteria",
    "notes",
    "target_segments",
    "use_cases",
    "claims",
    "differentiators",
}


def _generic_required_fields_by_dimension_name(dim_name: str) -> List[str]:
    n = _norm_text(dim_name).lower()
    if any(k in n for k in ("preis", "price", "kosten", "cost", "tco", "wirtschaft")):
        return ["price_indicators"]
    if any(k in n for k in ("compliance", "norm", "zert", "regulator", "sicherheit", "safety")):
        return ["relevance_criteria", "notes"]
    if any(k in n for k in ("zielgruppe", "segment", "use-case", "use case", "anwendung", "fit")):
        return ["target_segments", "use_cases"]
    if any(k in n for k in ("feature", "funktion", "ausstattung", "capabilit")):
        return ["normalized_features", "extended_feature_schema"]
    # Default for technical/competitive dimensions:
    return ["performance_parameters", "normalized_features"]


def _normalize_comparison_dimensions_required_fields(dimensions: List[ComparisonDimension]) -> List[ComparisonDimension]:
    out: List[ComparisonDimension] = []
    for d in dimensions:
        raw = [str(x).strip() for x in (d.required_fields or []) if str(x).strip()]
        kept = [x for x in raw if x in _GENERIC_REQUIRED_FIELDS]
        if not kept:
            kept = _generic_required_fields_by_dimension_name(d.name)
        # ensure deterministic order + dedupe
        seen: set[str] = set()
        normalized: List[str] = []
        for k in kept:
            if k in seen:
                continue
            seen.add(k)
            normalized.append(k)
        out.append(
            ComparisonDimension(
                name=d.name,
                weight=d.weight,
                rationale=d.rationale,
                required_fields=normalized,
            )
        )
    return out


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
    current_year = datetime.utcnow().year
    system = (
        "Du erzeugst einen adaptiven Wettbewerbsanalyse-Plan für ein Produktprofil. "
        "Antworte strikt als JSON gemäß Schema. "
        "Definiere Vergleichsdimensionen, Suchstrategie, Relevanzkriterien und Analyseumfang. "
        "Wichtig für search_queries: nur generische Wettbewerber-Suchanfragen, keine 'vs'-Vergleiche, "
        "keine site:-Filter, keine URLs. "
        "Keine markenspezifischen Pflichtbegriffe in required_fields; nur generische Felder. "
        "Jede Query muss mindestens 3 Wörter enthalten und konkrete Suchintention tragen. "
        "Jahreszahlen nicht in jeder Query verwenden."
    )
    user = (
        "Erstelle einen pragmatischen Analyseplan basierend auf folgendem product_profile. "
        "Gewichte Dimensionen sinnvoll und gib konkrete Suchbegriffe/Queries aus.\n"
        f"Für search_queries (genau {_TARGET_SEARCH_QUERIES} Stück) nutze diesen Mix:\n"
        "- 5-8 Markt-/Alternativen-Queries (Konkurrenz, Alternativen, Top-Modelle)\n"
        "- 5-8 technische Capability-Queries auf Basis comparison_dimensions\n"
        "- 3-5 Segment/Use-Case-Queries\n"
        "- 2-4 Kaufkriterien-Queries (Preis-Leistung, Zuverlässigkeit, Wartung)\n"
        f"- Jahresbezug nur aktuell/nah (z. B. {current_year}); keine alten Jahre.\n"
        "- Nur 30-50% der Queries dürfen eine Jahreszahl tragen.\n"
        "- Technische Capability-Queries und Use-Case-Queries bevorzugt ohne Jahreszahl formulieren.\n"
        "- Wenn ein Jahr verwendet wird, nicht standardmäßig als letztes Token anhängen; "
        "Formulierungen variieren (z. B. 'im Jahr 2026', '2026 Vergleich').\n\n"
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
