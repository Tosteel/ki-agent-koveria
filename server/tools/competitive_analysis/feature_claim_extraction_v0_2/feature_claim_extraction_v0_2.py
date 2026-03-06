from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from server.services.llm_ionos import IonosLLM
from server.services.llm_openai import LlmOpenai
from server.services.llm_perplexity import LlmPerplexity
from server.tools.competitive_analysis.document_import.models import ParsedDocument
from server.tools.competitive_analysis.backup.feature_claim_extraction.feature_claim_extraction import (
    _dedupe_by_key,
    _load_parsed_doc,
    _normalize_feature,
    _openai_extract_output_text,
    _parse_json_strictish,
    _safe_list_str,
)

from .models import (
    ClaimItem,
    ExtractionQualityReport,
    NormalizedFeature,
    PriceIndicator,
    ProductProfileV2,
    SoftFeature,
)


def _feature_obj_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string"},
            "value": {"oneOf": [{"type": "number"}, {"type": "string"}, {"type": "integer"}]},
            "unit": {"type": "string"},
            "normalized_value": {
                "oneOf": [{"type": "number"}, {"type": "string"}, {"type": "integer"}, {"type": "null"}]
            },
            "normalized_unit": {"type": "string"},
            "source": {"type": "string"},
        },
        "required": ["name", "value", "unit", "normalized_value", "normalized_unit", "source"],
    }


def _price_obj_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "raw": {"type": "string"},
            "value": {"oneOf": [{"type": "number"}, {"type": "integer"}, {"type": "null"}]},
            "currency": {"type": "string"},
            "period": {"type": "string"},
            "context": {"type": "string"},
        },
        "required": ["raw", "value", "currency", "period", "context"],
    }


def _claim_obj_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "text": {"type": "string"},
            "claim_type": {"type": "string", "enum": ["benefit", "differentiation", "value"]},
            "evidence": {"type": "string"},
        },
        "required": ["text", "claim_type", "evidence"],
    }


def _soft_feature_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string"},
            "available": {"type": "boolean"},
            "source": {"type": "string"},
        },
        "required": ["name", "available", "source"],
    }


def _llm_step_schema(step: str) -> Dict[str, Any]:
    if step == "normalized_features":
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "normalized_features": {"type": "array", "items": _feature_obj_schema()},
                "metadata_patch": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "product_name": {"type": ["string", "null"]},
                        "manufacturer": {"type": ["string", "null"]},
                    },
                    "required": ["product_name", "manufacturer"],
                },
            },
            "required": ["normalized_features", "metadata_patch"],
        }
    if step == "performance_parameters":
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "performance_parameters": {"type": "array", "items": _feature_obj_schema()},
            },
            "required": ["performance_parameters"],
        }
    if step == "price_indicators":
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "price_indicators": {"type": "array", "items": _price_obj_schema()},
            },
            "required": ["price_indicators"],
        }
    if step == "soft_features":
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "soft_features": {"type": "array", "items": _soft_feature_schema()},
            },
            "required": ["soft_features"],
        }
    if step == "final_semantics":
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "claims": {"type": "array", "items": _claim_obj_schema()},
                "differentiators": {"type": "array", "items": {"type": "string"}},
                "target_segments": {"type": "array", "items": {"type": "string"}},
                "use_cases": {"type": "array", "items": {"type": "string"}},
                "product_category": {"type": "string"},
            },
            "required": ["claims", "differentiators", "target_segments", "use_cases", "product_category"],
        }
    return {"type": "object", "additionalProperties": True, "properties": {}}


def _llm_step(
    *,
    provider: str,
    step: str,
    context: str,
    staged: Dict[str, Any],
    warnings: List[str],
) -> Dict[str, Any]:
    schema = _llm_step_schema(step)
    system = "Du extrahierst Produktprofil-Daten schrittweise als JSON. Nur Fakten aus Kontext verwenden. Nichts halluzinieren."
    if step == "normalized_features":
        system = (
            system
            + " Extrahiere ALLE relevanten Produktmerkmale aus dem Parsed-Doc in normalisierter Form."
            + " Bewahre bei ähnlichen Messwerten den fachlichen Kontext im Namen (z. B. Staubbehälter Roboter vs. Staubbehälter Basisstation)."
            + " Verwende keine generischen Namen wie 'measurement' oder 'value'."
        )
    if step == "performance_parameters":
        system = (
            system
            + " Extrahiere aus normalized_features ALLE messbaren Leistungs-/Spezifikationsmerkmale in einheitlicher Form."
            + " Messbar bedeutet: numerischer Wert und/oder explizite Einheit im Featurekontext."
            + " Solche Merkmale muessen immer in performance_parameters erscheinen, niemals nur in soft_features."
            + " Beispiele (generisch): Leistung, Laufzeit, Kapazitaet, Volumen, Temperatur, Abmessungen, Gewicht, Reichweite, Durchsatz, Geschwindigkeit."
            + " Leistungsangaben (z. B. Saugleistung, Motorleistung, Durchsatz) immer als performance_parameters ausgeben."
            + " Wenn ein Merkmal semantisch funktional wirkt, aber einen messbaren Wert/Einheit hat, trotzdem als performance_parameter aufnehmen."
            + " Bei mehrdimensionalen Maßangaben (z. B. 350 x 350 x 88 mm) splitte in eigene Parameter mit Achsenname: Breite, Tiefe, Hoehe."
            + " Bei 2D-Maßen nutze Breite und Hoehe."
        )
    if step == "soft_features":
        system = (
            system
            + " Extrahiere aus normalized_features und performance_parameters NUR soft_features (nicht-messbare Features wie spezielle Funktionen, App, Technologien)."
            + " Nimm NICHTS auf, was messbar ist oder bereits als performance_parameter enthalten ist."
            + " Merkmale mit Zahl/Einheit sind ausgeschlossen und gehoeren in performance_parameters."
            + " Keine Claims, keine Segmente, keine Use-Cases."
            + " Gib ein leeres Array nur zurück, wenn wirklich keine nicht-messbaren Merkmale vorkommen."
        )
    if step == "price_indicators":
        system = (
            system
            + " Extrahiere ALLE Preisangaben als price_indicators."
            + " Erkenne explizit Muster wie 'Preis', 'UVP', 'MSRP', 'statt', 'ab', 'sale'."
            + " Wenn Text wie 'Preis: 849,00 € (UVP 999,00 €)' vorkommt, gib zwei Einträge zurück:"
            + " einmal Kaufpreis (context='Preis') und einmal UVP (context='UVP')."
            + " currency als ISO-Code (z. B. EUR), period leer lassen falls nicht vorhanden."
            + " Keine Halluzinationen, nur Werte mit klarer Evidenz."
        )
    if step == "final_semantics":
        system = (
            system
            + " Erzeuge am Ende NUR claims, differentiators, target_segments und use_cases sowie product_category."
            + " Leite diese aus dem Parsed-Doc-Kontext und den bereits extrahierten Features ab."
            + " Wenn ausreichend technische/marketingbezogene Evidenz vorhanden ist, liefere nicht-leere Listen."
            + " Für Claims immer konkrete evidence-Textausschnitte angeben."
        )
    user = (
        f"Step: {step}\n"
        f"Bereits extrahiert: {json.dumps(staged, ensure_ascii=False)[:8000]}\n\n"
        f"Kontext:\n{context[:15000]}"
    )

    p = str(provider or "ionos").strip().lower()
    if p not in {"ionos", "openai", "perplexity"}:
        p = "ionos"

    if p in {"openai", "perplexity"}:
        client = LlmOpenai() if p == "openai" else LlmPerplexity()
        if not client.enabled():
            warnings.append(f"{p} step '{step}' skipped (provider not configured).")
            return {}
        try:
            resp = client._call(
                input_messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                text_format={"type": "json_schema", "name": f"feature_claim_{step}", "schema": schema, "strict": False},
            )
            return _parse_json_strictish(_openai_extract_output_text(resp))
        except Exception as exc:
            warnings.append(f"{p} step '{step}' failed: {exc}")
            return {}

    client_i = IonosLLM()
    if not client_i.enabled():
        warnings.append(f"IONOS step '{step}' skipped (provider not configured).")
        return {}
    try:
        comp = client_i.chat_completions(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": f"feature_claim_{step}", "schema": schema, "strict": True},
            },
        )
        return _parse_json_strictish(client_i.extract_text(comp))
    except Exception as exc:
        warnings.append(f"IONOS step '{step}' failed: {exc}")
        return {}


def _derive_feature_name_from_source(source: str, fallback: str = "measurement") -> str:
    src = re.sub(r"\s+", " ", str(source or "")).strip()
    if not src:
        return fallback
    m = re.search(r"([A-Za-zÄÖÜäöüß0-9™®©/()+\- ]{3,80})\s*:\s*[-+0-9]", src)
    if m:
        cand = m.group(1).strip(" -•\t")
        if cand and cand.lower() not in {"measurement", "value", "wert"}:
            return cand
    parts = re.split(r"[|•]", src)
    for p in parts:
        p = p.strip(" -•\t")
        if len(p) >= 3 and not re.match(r"^[0-9.,\\s]+$", p):
            return p[:80]
    return fallback


def _repair_base_feature_names(features: List[NormalizedFeature]) -> List[NormalizedFeature]:
    out: List[NormalizedFeature] = []
    for f in features or []:
        n = (f.name or "").strip()
        if n.lower() in {"measurement", "unknown", "value", "wert"} or len(n) < 3:
            f.name = _derive_feature_name_from_source(f.source, n or "measurement")
        out.append(f)
    return out


def _to_float_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip().replace(",", ".")
        m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
        if m:
            try:
                return float(m.group(0))
            except Exception:
                return None
    return None


def _looks_measurable_feature(f: NormalizedFeature) -> bool:
    if _to_float_or_none(f.normalized_value) is not None:
        return True
    if _to_float_or_none(f.value) is not None:
        return True
    unit = str(f.normalized_unit or f.unit or "").strip()
    if unit:
        return True
    src = str(f.source or "")
    if re.search(r"[-+]?\d+(?:[.,]\d+)?\s*[A-Za-z%°/]+", src):
        return True
    return False


def _merge_missing_measurables_into_performance(
    *,
    normalized_features: List[NormalizedFeature],
    performance_parameters: List[NormalizedFeature],
) -> List[NormalizedFeature]:
    existing_keys = {
        f"{(str(f.name or '')).strip().lower()}|{str(f.normalized_unit or f.unit or '').strip().lower()}"
        for f in (performance_parameters or [])
    }
    merged = list(performance_parameters or [])
    for f in normalized_features or []:
        if not _looks_measurable_feature(f):
            continue
        key = f"{(str(f.name or '')).strip().lower()}|{str(f.normalized_unit or f.unit or '').strip().lower()}"
        if key in existing_keys:
            continue
        merged.append(f)
        existing_keys.add(key)
    return _dedupe_by_key(
        merged,
        lambda x: f"{x.name.lower()}|{x.normalized_value}|{x.normalized_unit.lower()}",
    )


def _extract_dim_values(value: Any) -> List[float]:
    txt = str(value or "")
    if not txt:
        return []
    # Support common separators for dimensions: x, X, ×
    parts = re.split(r"\s*[xX×]\s*", txt)
    if len(parts) < 2:
        return []
    out: List[float] = []
    for p in parts:
        m = re.search(r"[-+]?\d+(?:[.,]\d+)?", p)
        if not m:
            continue
        try:
            out.append(float(m.group(0).replace(",", ".")))
        except Exception:
            continue
    return out if len(out) >= 2 else []


def _axis_labels_for_dims(dim_count: int) -> List[str]:
    if dim_count >= 3:
        return ["Breite", "Tiefe", "Hoehe"][:dim_count]
    if dim_count == 2:
        return ["Breite", "Hoehe"]
    return [f"Dimension {i+1}" for i in range(dim_count)]


def _base_measure_name(name: str) -> str:
    n = str(name or "").strip()
    if not n:
        return "Masse"
    # Remove common brackets/tails to produce stable base
    n = re.sub(r"\(.*?\)", "", n).strip()
    n = re.sub(r"\s+", " ", n).strip(" -:_")
    return n or "Masse"


def _fix_multidim_features(features: List[NormalizedFeature]) -> List[NormalizedFeature]:
    out: List[NormalizedFeature] = []
    for f in features or []:
        dims = _extract_dim_values(f.value)
        if dims:
            # Keep original dimensional string as normalized value to avoid collapsing to one axis.
            f.normalized_value = str(f.value)
            if not (f.normalized_unit or "").strip():
                f.normalized_unit = str(f.unit or "").strip()
            out.append(f)
            unit = str(f.unit or f.normalized_unit or "").strip()
            labels = _axis_labels_for_dims(len(dims))
            base_name = _base_measure_name(f.name)
            for i, v in enumerate(dims):
                axis = labels[i] if i < len(labels) else f"Dimension {i+1}"
                out.append(
                    NormalizedFeature(
                        name=f"{base_name} - {axis}",
                        value=v,
                        unit=unit,
                        normalized_value=v,
                        normalized_unit=unit,
                        source=f.source,
                    )
                )
            continue
        out.append(f)
    return _dedupe_by_key(
        out,
        lambda x: f"{x.name.lower()}|{x.normalized_value}|{x.normalized_unit.lower()}",
    )


def _to_feature_list(raw: Any) -> List[NormalizedFeature]:
    out: List[NormalizedFeature] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            out.append(NormalizedFeature(**item))
        except Exception:
            continue
    return out


def _coerce_feature_models(items: List[Any]) -> List[NormalizedFeature]:
    out: List[NormalizedFeature] = []
    for item in items or []:
        if isinstance(item, NormalizedFeature):
            out.append(item)
            continue
        data: Dict[str, Any] | None = None
        if isinstance(item, dict):
            data = item
        elif hasattr(item, "model_dump"):
            try:
                data = dict(item.model_dump())
            except Exception:
                data = None
        if not data:
            continue
        try:
            out.append(NormalizedFeature(**data))
        except Exception:
            continue
    return out


def _to_price_list(raw: Any) -> List[PriceIndicator]:
    out: List[PriceIndicator] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            out.append(PriceIndicator(**item))
        except Exception:
            continue
    return out


def _coerce_price_models(items: List[Any]) -> List[PriceIndicator]:
    out: List[PriceIndicator] = []
    for item in items or []:
        if isinstance(item, PriceIndicator):
            out.append(item)
            continue
        data: Dict[str, Any] | None = None
        if isinstance(item, dict):
            data = item
        elif hasattr(item, "model_dump"):
            try:
                data = dict(item.model_dump())
            except Exception:
                data = None
        if not data:
            continue
        try:
            out.append(PriceIndicator(**data))
        except Exception:
            continue
    return out


def _to_claims(raw: Any) -> List[ClaimItem]:
    out: List[ClaimItem] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            out.append(ClaimItem(**item))
        except Exception:
            continue
    return out


def _to_soft_features(raw: Any) -> List[SoftFeature]:
    out: List[SoftFeature] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            out.append(SoftFeature(**item))
        except Exception:
            continue
    return out


def extract_feature_claim_profile_v0_2(
    *,
    parsed_doc: Optional[Dict[str, Any]],
    parsed_doc_path: Optional[str],
    provider: str = "ionos",
    max_context_chars: int = 18000,
    user_root: Path,
    work_root: Path,
) -> ProductProfileV2:
    doc: ParsedDocument = _load_parsed_doc(
        parsed_doc=parsed_doc,
        parsed_doc_path=parsed_doc_path,
        user_root=user_root,
        work_root=work_root,
    )
    warnings: List[str] = list(doc.extraction_warnings or [])

    base_features = [
        _normalize_feature(
            name=m.property_name or "measurement",
            value=m.value,
            unit=m.unit,
            source=m.context or "measurement",
        )
        for m in doc.measurements
    ]
    base_features = _dedupe_by_key(
        base_features,
        lambda f: f"{f.name.lower()}|{f.normalized_value}|{f.normalized_unit.lower()}",
    )
    base_features = _repair_base_feature_names(base_features)

    section_text = "\n\n".join([f"[{s.name}]\n{s.text}" for s in doc.sections if (s.text or "").strip()])
    raw_text = (doc.raw_text or "").strip()
    combined_text = (section_text or raw_text)[:max_context_chars]
    metadata = doc.metadata.model_dump()

    staged: Dict[str, Any] = {}

    # Step 1: normalized_features
    part = _llm_step(
        provider=provider,
        step="normalized_features",
        context=json.dumps({"metadata": metadata, "measurements": [f.model_dump() for f in base_features[:220]], "text": combined_text}, ensure_ascii=False),
        staged=staged,
        warnings=warnings,
    )
    staged.update(part if isinstance(part, dict) else {})
    normalized_features = _to_feature_list(staged.get("normalized_features"))
    normalized_features = _coerce_feature_models(normalized_features)
    normalized_features = _dedupe_by_key(
        normalized_features,
        lambda f: f"{f.name.lower()}|{f.normalized_value}|{f.normalized_unit.lower()}",
    )
    normalized_features = _fix_multidim_features(normalized_features)
    mp = staged.get("metadata_patch") if isinstance(staged.get("metadata_patch"), dict) else {}
    if str(mp.get("product_name") or "").strip():
        metadata["product_name"] = str(mp.get("product_name")).strip()
    if str(mp.get("manufacturer") or "").strip():
        metadata["manufacturer"] = str(mp.get("manufacturer")).strip()

    # Step 2: performance_parameters (LLM only, derived from normalized_features)
    part = _llm_step(
        provider=provider,
        step="performance_parameters",
        context=json.dumps({"normalized_features": [f.model_dump() for f in normalized_features[:220]], "text": combined_text}, ensure_ascii=False),
        staged=staged,
        warnings=warnings,
    )
    staged.update(part if isinstance(part, dict) else {})
    performance_parameters = _to_feature_list(staged.get("performance_parameters"))
    if not performance_parameters:
        retry_context = (
            "WICHTIG: Extrahiere aus normalized_features ALLE messbaren Produktmerkmale "
            "als performance_parameters (keine Teilmenge nur auf Core-KPIs begrenzen).\n\n"
            + json.dumps(
                {"normalized_features": [f.model_dump() for f in normalized_features[:220]], "text": combined_text[:14000]},
                ensure_ascii=False,
            )
        )
        retry = _llm_step(
            provider=provider,
            step="performance_parameters",
            context=retry_context,
            staged=staged,
            warnings=warnings,
        )
        if isinstance(retry, dict):
            performance_parameters = _to_feature_list(retry.get("performance_parameters"))
    performance_parameters = _coerce_feature_models(performance_parameters)
    performance_parameters = _dedupe_by_key(
        performance_parameters,
        lambda f: f"{f.name.lower()}|{f.normalized_value}|{f.normalized_unit.lower()}",
    )
    performance_parameters = _fix_multidim_features(performance_parameters)
    performance_parameters = _merge_missing_measurables_into_performance(
        normalized_features=normalized_features,
        performance_parameters=performance_parameters,
    )

    # Step 3: price_indicators
    price_context = json.dumps(
        {
            "metadata": metadata,
            "sections": [
                {"name": s.name, "text": s.text}
                for s in doc.sections[:40]
                if (s.text or "").strip()
            ],
            "raw_text_excerpt": (raw_text or section_text)[:14000],
        },
        ensure_ascii=False,
    )
    part = _llm_step(
        provider=provider,
        step="price_indicators",
        context=price_context,
        staged=staged,
        warnings=warnings,
    )
    staged.update(part if isinstance(part, dict) else {})
    price_indicators = _to_price_list(staged.get("price_indicators"))
    if not price_indicators:
        retry_context = (
            "WICHTIG: Extrahiere NUR konkrete Preisangaben (raw, value, currency, period, context). "
            "Wenn im Kontext keine Preise vorhanden sind, gib leeres Array zurück.\n\n"
            + price_context
        )
        retry = _llm_step(
            provider=provider,
            step="price_indicators",
            context=retry_context,
            staged=staged,
            warnings=warnings,
        )
        if isinstance(retry, dict):
            price_indicators = _to_price_list(retry.get("price_indicators"))
    price_indicators = _coerce_price_models(price_indicators)
    price_indicators = _dedupe_by_key(
        price_indicators,
        lambda p: f"{p.raw.lower()}|{p.value}|{p.currency}|{p.period}",
    )

    # Step 4: soft_features (own LLM pass from normalized_features + performance_parameters; no claims in this step)
    part = _llm_step(
        provider=provider,
        step="soft_features",
        context=json.dumps(
            {
                "metadata": metadata,
                "normalized_features": [f.model_dump() for f in normalized_features[:160]],
                "performance_parameters": [f.model_dump() for f in performance_parameters[:120]],
                "text": combined_text,
            },
            ensure_ascii=False,
        ),
        staged=staged,
        warnings=warnings,
    )
    staged.update(part if isinstance(part, dict) else {})
    soft_features = _to_soft_features(staged.get("soft_features"))
    if not soft_features:
        retry_context = (
            "WICHTIG: Extrahiere aus normalized_features und performance_parameters explizit alle nicht-messbaren Merkmale als soft_features. "
            "Beispiele: Anti-Haarverhedderung, App-Steuerung, Navigationstechnologie, spezielle Bürstensysteme. "
            "Schließe alles aus, was messbar ist (mit Zahlen/Einheiten) oder bereits in performance_parameters vorkommt. "
            "Gib bei vorhandener Evidenz mindestens 3 soft_features zurück.\n\n"
            + json.dumps(
                {
                    "normalized_features": [f.model_dump() for f in normalized_features[:160]],
                    "performance_parameters": [f.model_dump() for f in performance_parameters[:120]],
                },
                ensure_ascii=False,
            )
        )
        retry = _llm_step(
            provider=provider,
            step="soft_features",
            context=retry_context,
            staged=staged,
            warnings=warnings,
        )
        if isinstance(retry, dict):
            soft_features = _to_soft_features(retry.get("soft_features"))
    soft_features = _dedupe_by_key(soft_features, lambda s: f"{s.name.lower()}|{s.available}")

    # Step 5: one final LLM pass for claims + differentiators + target_segments + use_cases.
    provisional_category = str(staged.get("product_category") or "").strip()
    final_context = json.dumps(
        {
            "parsed_doc_metadata": metadata,
            "parsed_doc_sections_excerpt": section_text[:12000],
            "parsed_doc_raw_excerpt": raw_text[:12000],
            "product_category": provisional_category,
            "soft_features": [s.model_dump() for s in soft_features[:40]],
            "normalized_features": [f.model_dump() for f in normalized_features[:120]],
            "performance_parameters": [f.model_dump() for f in performance_parameters[:100]],
            "price_indicators": [p.model_dump() for p in price_indicators[:40]],
        },
        ensure_ascii=False,
    )
    part = _llm_step(
        provider=provider,
        step="final_semantics",
        context=final_context,
        staged=staged,
        warnings=warnings,
    )
    staged.update(part if isinstance(part, dict) else {})
    if not staged.get("claims") or not staged.get("differentiators") or not staged.get("target_segments") or not staged.get("use_cases"):
        retry_context = (
            "WICHTIG: Fülle claims, differentiators, target_segments und use_cases anhand des Parsed-Doc-Kontexts. "
            "Wenn Funktionen/Spezifikationen vorhanden sind, gib nicht-leere Listen aus. "
            "Claims brauchen konkrete evidence-Auszüge.\n\n"
            + final_context
        )
        retry = _llm_step(
            provider=provider,
            step="final_semantics",
            context=retry_context,
            staged=staged,
            warnings=warnings,
        )
        if isinstance(retry, dict):
            staged.update({k: v for k, v in retry.items() if v})

    claims = _to_claims(staged.get("claims"))
    claims = _dedupe_by_key(claims, lambda c: f"{c.text.lower()}|{c.claim_type.lower()}")

    product_category = str(staged.get("product_category") or "").strip()

    differentiators = _dedupe_by_key(_safe_list_str(staged.get("differentiators")), lambda s: s.lower())
    target_segments = _dedupe_by_key(_safe_list_str(staged.get("target_segments")), lambda s: s.lower())
    use_cases = _dedupe_by_key(_safe_list_str(staged.get("use_cases")), lambda s: s.lower())

    p = str(provider or "ionos").strip().lower()
    if p not in {"ionos", "openai", "perplexity"}:
        p = "ionos"

    quality_report = ExtractionQualityReport(
        normalized_features_count=len(normalized_features),
        performance_parameters_count=len(performance_parameters),
        price_indicators_count=len(price_indicators),
        claims_count=len(claims),
        soft_features_count=len(soft_features),
        staged_llm_steps=["normalized_features", "performance_parameters", "price_indicators", "soft_features", "final_semantics"],
        notes=["v0.2 sequential extraction pipeline executed."],
    )

    return ProductProfileV2(
        provider=p,
        product_category=product_category,
        metadata=metadata,
        normalized_features=normalized_features,
        performance_parameters=performance_parameters,
        price_indicators=price_indicators,
        claims=claims,
        soft_features=soft_features,
        differentiators=differentiators,
        target_segments=target_segments,
        use_cases=use_cases,
        extraction_warnings=warnings,
        quality_report=quality_report,
    )
