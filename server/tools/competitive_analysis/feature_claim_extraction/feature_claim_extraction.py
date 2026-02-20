from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from fastapi import HTTPException

from server.services.llm_ionos import IonosLLM
from server.services.llm_openai import LlmOpenai
from server.services.llm_perplexity import LlmPerplexity
from server.tools.competitive_analysis.document_import.models import ParsedDocument

from .models import ClaimItem, NormalizedFeature, PriceIndicator, ProductProfile


_PRICE_RE = re.compile(
    r"(?P<raw>(?:(?:EUR|USD|CHF|GBP)\s*)?[0-9]{1,3}(?:[.,][0-9]{3})*(?:[.,][0-9]{1,2})?\s*(?:EUR|USD|CHF|GBP|€|\$|£))",
    re.IGNORECASE,
)

_CURRENCY_MAP = {
    "€": "EUR",
    "$": "USD",
    "£": "GBP",
}

_UNIT_NORMALIZATION = {
    "kw": (1000.0, "W"),
    "w": (1.0, "W"),
    "mw": (0.001, "W"),
    "kg": (1.0, "kg"),
    "g": (0.001, "kg"),
    "mg": (0.000001, "kg"),
    "l": (1.0, "L"),
    "ml": (0.001, "L"),
    "m": (1.0, "m"),
    "cm": (0.01, "m"),
    "mm": (0.001, "m"),
    "kv": (1000.0, "V"),
    "v": (1.0, "V"),
    "ma": (0.001, "A"),
    "a": (1.0, "A"),
    "khz": (1000.0, "Hz"),
    "mhz": (1000000.0, "Hz"),
    "ghz": (1000000000.0, "Hz"),
    "hz": (1.0, "Hz"),
}

_PERFORMANCE_HINTS = {
    "leistung",
    "power",
    "capacity",
    "kapazität",
    "accuracy",
    "genauigkeit",
    "efficiency",
    "wirkungsgrad",
    "throughput",
    "druck",
    "pressure",
    "spannung",
    "strom",
    "frequency",
    "frequenz",
}

_CATEGORY_HINTS = {
    "pump": "industrial_pump",
    "pumpe": "industrial_pump",
    "sensor": "sensor",
    "inverter": "inverter",
    "antrieb": "drive_system",
    "heating": "hvac",
    "kühl": "hvac",
    "compressor": "compressor",
}


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


def _load_parsed_doc(
    *,
    parsed_doc: Optional[Dict[str, Any]],
    parsed_doc_path: Optional[str],
    user_root: Path,
    work_root: Path,
) -> ParsedDocument:
    payload: Dict[str, Any]
    if isinstance(parsed_doc, dict) and parsed_doc:
        payload = parsed_doc
    else:
        p = _resolve_input_path(str(parsed_doc_path or ""), user_root=user_root.resolve(), work_root=work_root.resolve())
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON in parsed_doc_path: {parsed_doc_path}") from exc

    if "parsed_doc" in payload and isinstance(payload.get("parsed_doc"), dict):
        payload = payload["parsed_doc"]

    return ParsedDocument(**payload)


def _normalize_feature(name: str, value: Any, unit: str, source: str = "") -> NormalizedFeature:
    unit_raw = str(unit or "").strip()
    unit_key = unit_raw.lower()
    normalized_value: float | int | str | None = None
    normalized_unit = unit_raw

    numeric_value: float | int | None = None
    if isinstance(value, (int, float)):
        numeric_value = value
    else:
        try:
            numeric_value = float(str(value).replace(",", "."))
        except Exception:
            numeric_value = None

    if numeric_value is not None and unit_key in _UNIT_NORMALIZATION:
        factor, target_unit = _UNIT_NORMALIZATION[unit_key]
        norm = float(numeric_value) * factor
        normalized_value = int(norm) if norm.is_integer() else round(norm, 6)
        normalized_unit = target_unit
    else:
        normalized_value = numeric_value if numeric_value is not None else str(value)

    return NormalizedFeature(
        name=str(name or "").strip() or "unknown",
        value=value,
        unit=unit_raw,
        normalized_value=normalized_value,
        normalized_unit=normalized_unit,
        source=source,
    )


def _extract_price_indicators(text: str) -> List[PriceIndicator]:
    out: List[PriceIndicator] = []
    for m in _PRICE_RE.finditer(text or ""):
        raw = m.group("raw")
        lower = raw.lower()

        currency = ""
        for symbol, code in _CURRENCY_MAP.items():
            if symbol in raw:
                currency = code
                break
        if not currency:
            for code in ("EUR", "USD", "CHF", "GBP"):
                if code.lower() in lower:
                    currency = code
                    break

        num_match = re.search(r"[0-9]{1,3}(?:[.,][0-9]{3})*(?:[.,][0-9]{1,2})?|[0-9]+(?:[.,][0-9]{1,2})?", raw)
        num = None
        if num_match:
            val = num_match.group(0).replace(".", "").replace(",", ".")
            try:
                f = float(val)
                num = int(f) if f.is_integer() else f
            except Exception:
                num = None

        start = max(0, m.start() - 60)
        end = min(len(text), m.end() + 60)
        context = (text[start:end] or "").replace("\n", " ").strip()

        period = ""
        window = context.lower()
        if "monat" in window or "month" in window:
            period = "monthly"
        elif "jahr" in window or "year" in window or "jähr" in window:
            period = "yearly"
        elif "einmal" in window or "one-time" in window:
            period = "one_time"

        out.append(PriceIndicator(raw=raw, value=num, currency=currency, period=period, context=context))
    return out


def _dedupe_by_key(items: Iterable[Any], key_fn) -> List[Any]:
    out: List[Any] = []
    seen: set[str] = set()
    for item in items:
        key = key_fn(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _heuristic_category(text: str) -> str:
    low = (text or "").lower()
    for hint, cat in _CATEGORY_HINTS.items():
        if hint in low:
            return cat
    return "unknown"


def _build_semantic_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "product_category": {"type": "string"},
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "text": {"type": "string"},
                        "claim_type": {"type": "string", "enum": ["benefit", "differentiation", "value"]},
                        "evidence": {"type": "string"},
                    },
                    "required": ["text", "claim_type", "evidence"],
                },
            },
            "differentiators": {"type": "array", "items": {"type": "string"}},
            "target_segments": {"type": "array", "items": {"type": "string"}},
            "use_cases": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["product_category", "claims", "differentiators", "target_segments", "use_cases"],
    }


def _openai_extract_output_text(resp: Dict[str, Any]) -> str:
    out = ""
    for item in resp.get("output", []):
        for c in item.get("content", []):
            if c.get("type") == "output_text":
                out += str(c.get("text") or "")
    return out.strip()


def _llm_semantics(provider: str, context: str, warnings: List[str]) -> Dict[str, Any]:
    schema = _build_semantic_schema()
    system = (
        "Extrahiere ein normiertes Produktprofil aus dem Dokumentkontext. "
        "Liefere nur JSON gemäß Schema. Keine Halluzinationen. "
        "Claims müssen im Text belegbar sein. Redundanzen entfernen."
    )
    user = (
        "Analysiere den folgenden Dokumentkontext und extrahiere: "
        "Produktkategorie, Claims, Differenzierungsmerkmale, Zielsegmente und Use-Cases.\n\n"
        f"Kontext:\n{context}"
    )

    p = str(provider or "ionos").strip().lower()
    if p not in {"ionos", "openai", "perplexity"}:
        p = "ionos"

    if p in {"openai", "perplexity"}:
        client = LlmOpenai() if p == "openai" else LlmPerplexity()
        if not client.enabled():
            warnings.append(f"{p} not configured; semantic extraction fallback used.")
            return {}
        fmt = {"type": "json_schema", "name": "product_profile_semantics", "schema": schema, "strict": False}
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
            warnings.append(f"{p} semantic extraction failed: {exc}")
            return {}

    client_i = IonosLLM()
    if not client_i.enabled():
        warnings.append("IONOS not configured; semantic extraction fallback used.")
        return {}

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "product_profile_semantics",
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
        warnings.append(f"IONOS semantic extraction failed: {exc}")
        return {}


def extract_feature_claim_profile(
    *,
    parsed_doc: Optional[Dict[str, Any]],
    parsed_doc_path: Optional[str],
    provider: str = "ionos",
    max_context_chars: int = 18000,
    user_root: Path,
    work_root: Path,
) -> ProductProfile:
    doc = _load_parsed_doc(
        parsed_doc=parsed_doc,
        parsed_doc_path=parsed_doc_path,
        user_root=user_root,
        work_root=work_root,
    )

    warnings: List[str] = list(doc.extraction_warnings or [])

    features = [
        _normalize_feature(
            name=m.property_name or "measurement",
            value=m.value,
            unit=m.unit,
            source=m.context or "measurement",
        )
        for m in doc.measurements
    ]
    features = _dedupe_by_key(features, lambda f: f"{f.name.lower()}|{f.normalized_value}|{f.normalized_unit.lower()}")

    performance = [
        f
        for f in features
        if any(h in f"{f.name} {f.source}".lower() for h in _PERFORMANCE_HINTS)
    ]
    performance = _dedupe_by_key(performance, lambda f: f"{f.name.lower()}|{f.normalized_value}|{f.normalized_unit.lower()}")

    section_text = "\n\n".join(
        [f"[{s.name}]\n{s.text}" for s in doc.sections if (s.text or "").strip()]
    )
    raw_text = (doc.raw_text or "").strip()
    combined_text = (section_text or raw_text)[:max_context_chars]

    price_indicators = _extract_price_indicators(raw_text or section_text)
    price_indicators = _dedupe_by_key(
        price_indicators,
        lambda p: f"{p.raw.lower()}|{p.value}|{p.currency}|{p.period}",
    )

    compact_feature_lines = [
        f"- {f.name}: {f.value} {f.unit} (normalized={f.normalized_value} {f.normalized_unit})"
        for f in features[:80]
    ]
    metadata = doc.metadata.model_dump()
    llm_context = (
        f"Metadata: {json.dumps(metadata, ensure_ascii=False)}\n\n"
        f"Normalized features:\n{chr(10).join(compact_feature_lines)}\n\n"
        f"Sections/raw text:\n{combined_text}"
    )[:max_context_chars]

    semantic = _llm_semantics(provider=provider, context=llm_context, warnings=warnings)

    claims = [ClaimItem(**c) for c in (semantic.get("claims") or []) if isinstance(c, dict)]
    claims = _dedupe_by_key(claims, lambda c: f"{c.text.lower()}|{c.claim_type.lower()}")

    product_category = str(semantic.get("product_category") or "").strip() or _heuristic_category(raw_text or section_text)
    target_segments = _dedupe_by_key(_safe_list_str(semantic.get("target_segments")), lambda s: s.lower())
    use_cases = _dedupe_by_key(_safe_list_str(semantic.get("use_cases")), lambda s: s.lower())
    differentiators = _dedupe_by_key(_safe_list_str(semantic.get("differentiators")), lambda s: s.lower())

    p = str(provider or "ionos").strip().lower()
    if p not in {"ionos", "openai", "perplexity"}:
        p = "ionos"

    return ProductProfile(
        provider=p,
        product_category=product_category,
        metadata=metadata,
        normalized_features=features,
        performance_parameters=performance,
        price_indicators=price_indicators,
        claims=claims,
        differentiators=differentiators,
        target_segments=target_segments,
        use_cases=use_cases,
        extraction_warnings=warnings,
    )
