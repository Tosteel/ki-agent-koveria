from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException

from server.services.llm_ionos import IonosLLM
from server.services.llm_openai import LlmOpenai
from server.services.llm_perplexity import LlmPerplexity
from server.tools.competitive_analysis.backup.feature_claim_extraction.models import (
    NormalizedFeature,
    ProductProfile,
)

from .models import FeatureClaimQualityReport


_DIM_FRAGMENT_RE = re.compile(
    r"^\d+(?:[.,]\d+)?(?:\s*[x×]\s*\d+(?:[.,]\d+)?){1,4}\s*[x×]?$",
    re.IGNORECASE,
)
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_SPACE_RE = re.compile(r"\s+")
_GENERIC_BAD_NAMES = {
    "measurement",
    "unknown",
    "n/a",
    "na",
    "-",
    "x",
    "xx",
}
_PSEUDO_NAME_RES = [
    re.compile(r"^vdc\s*=", re.IGNORECASE),
    re.compile(r"^idc\s*=", re.IGNORECASE),
    re.compile(r"^udc\s*=", re.IGNORECASE),
    re.compile(r"^[x×/\-.,\s\d]+$"),
]


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

    for c in candidates:
        if c.exists() and c.is_file() and (user_root in c.parents or c == user_root):
            return c

    raise HTTPException(status_code=404, detail=f"File not found: {path}")


def _load_profile(
    *,
    product_profile: Optional[Dict[str, Any]],
    product_profile_path: Optional[str],
    user_root: Path,
    work_root: Path,
) -> ProductProfile:
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

    try:
        return ProductProfile(**payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid product_profile payload: {exc}") from exc


def _clean_text(text: str) -> str:
    t = _CONTROL_RE.sub(" ", str(text or ""))
    return _SPACE_RE.sub(" ", t).strip()


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
        try:
            obj = json.loads(t[start : end + 1])
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _llm_clean_schema() -> Dict[str, Any]:
    feature_obj = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string"},
            "value": {"oneOf": [{"type": "number"}, {"type": "string"}, {"type": "integer"}]},
            "unit": {"type": "string"},
            "normalized_value": {"oneOf": [{"type": "number"}, {"type": "string"}, {"type": "integer"}, {"type": "null"}]},
            "normalized_unit": {"type": "string"},
            "source": {"type": "string"},
        },
        "required": ["name", "value", "unit", "normalized_value", "normalized_unit", "source"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "normalized_features": {"type": "array", "items": feature_obj},
            "performance_parameters": {"type": "array", "items": feature_obj},
            "quality_notes": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["normalized_features", "performance_parameters", "quality_notes"],
    }


def _openai_extract_output_text(resp: Dict[str, Any]) -> str:
    out = ""
    for item in resp.get("output", []):
        for c in item.get("content", []):
            if c.get("type") == "output_text":
                out += str(c.get("text") or "")
    return out.strip()


def _extract_any_output_text(resp: Dict[str, Any]) -> str:
    txt = _openai_extract_output_text(resp)
    if txt:
        return txt
    try:
        return LlmPerplexity._extract_text(resp)  # type: ignore[attr-defined]
    except Exception:
        return ""


def _feature_to_prompt_dict(f: NormalizedFeature, *, include_source: bool) -> Dict[str, Any]:
    src = _clean_text(f.source)
    if include_source and len(src) > 180:
        src = src[:180].rstrip() + "..."
    return {
        "name": _clean_text(f.name),
        "value": f.value,
        "unit": _clean_text(f.unit),
        "normalized_value": f.normalized_value,
        "normalized_unit": _clean_text(f.normalized_unit),
        "source": src if include_source else "",
    }


def _build_llm_user_payload(
    *,
    profile: ProductProfile,
    max_context_chars: int,
    include_source: bool,
) -> str:
    nf = [_feature_to_prompt_dict(f, include_source=include_source) for f in (profile.normalized_features or [])]
    pf = [_feature_to_prompt_dict(f, include_source=include_source) for f in (profile.performance_parameters or [])]
    payload = {
        "product_category": profile.product_category,
        "normalized_features": nf,
        "performance_parameters": pf,
    }
    # Keep prompt JSON syntactically valid; shrink by dropping tail items instead of hard string cutting.
    while len(json.dumps(payload, ensure_ascii=False)) > max_context_chars and (payload["normalized_features"] or payload["performance_parameters"]):
        if len(payload["normalized_features"]) >= len(payload["performance_parameters"]) and payload["normalized_features"]:
            payload["normalized_features"].pop()
        elif payload["performance_parameters"]:
            payload["performance_parameters"].pop()
        else:
            break
    return "Eingabeprofil:\n" + json.dumps(payload, ensure_ascii=False)


def _llm_clean_features(
    *,
    provider: str,
    profile: ProductProfile,
    max_context_chars: int,
) -> Dict[str, Any]:
    p = str(provider or "openai").strip().lower()
    if p not in {"openai", "ionos", "perplexity"}:
        p = "openai"

    schema = _llm_clean_schema()
    system = (
        "Du bist ein Quality-Gate für Produkt-Feature-Extraktion. "
        "Korrigiere und bereinige NUR normalized_features/performance_parameters. "
        "Regeln: 1) Unsinnige, unvollstaendige oder offensichtlich generische Feature-Namen entfernen. "
        "2) Abgeschnittene Namen sinnvoll reparieren. 3) Keine neuen Fakten erfinden. "
        "4) Werte/Einheiten nur aus vorhandenen Daten übernehmen. "
        "5) Nur gültiges JSON gemäß Schema."
    )
    users = [
        _build_llm_user_payload(profile=profile, max_context_chars=max_context_chars, include_source=True),
        _build_llm_user_payload(profile=profile, max_context_chars=min(max_context_chars, 14000), include_source=False),
        _build_llm_user_payload(profile=profile, max_context_chars=min(max_context_chars, 9000), include_source=False),
    ]

    if p in {"openai", "perplexity"}:
        client = LlmOpenai() if p == "openai" else LlmPerplexity()
        if not client.enabled():
            raise HTTPException(status_code=400, detail=f"{p} not configured for feature_claim_extraction_quality_gate.")
        rf = {
            "type": "json_schema",
            "name": "feature_claim_quality_gate",
            "schema": schema,
            "strict": True,
        }
        errors: List[str] = []
        for idx, user in enumerate(users, start=1):
            try:
                resp = client._call(
                    input_messages=[
                        {
                            "role": "system",
                            "content": system
                            + " WICHTIG: Gib ausschließlich ein einzelnes JSON-Objekt zurück. Kein Markdown, kein Fließtext.",
                        },
                        {"role": "user", "content": user},
                    ],
                    text_format=rf,
                )
            except Exception as exc:
                errors.append(f"attempt_{idx}: {exc}")
                continue
            parsed = _parse_json_strictish(_extract_any_output_text(resp))
            if parsed and isinstance(parsed.get("normalized_features"), list) and isinstance(parsed.get("performance_parameters"), list):
                return parsed
            errors.append(f"attempt_{idx}: invalid_json_or_schema")
        raise HTTPException(status_code=502, detail=f"{p} quality gate returned invalid JSON. {' | '.join(errors)}")

    client_i = IonosLLM()
    if not client_i.enabled():
        raise HTTPException(status_code=400, detail="IONOS not configured for feature_claim_extraction_quality_gate.")
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "feature_claim_quality_gate",
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
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"IONOS quality gate failed: {exc}") from exc
    if not parsed:
        raise HTTPException(status_code=502, detail="IONOS quality gate returned invalid JSON.")
    return parsed


def _count_alpha_chars(text: str) -> int:
    return sum(1 for ch in str(text or "") if ch.isalpha())


def _repair_feature_name(name: str) -> str:
    out = _clean_text(name)
    out = out.rstrip(":;,.- ")
    out = re.sub(r"\s*\(\s*$", "", out)
    out = re.sub(r"\s*\(bis\s*$", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+bis\s*$", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\s*[x×]\s*$", "", out)
    out = _SPACE_RE.sub(" ", out).strip()
    return out


def _feature_quality_reason(
    *,
    feature_name: str,
    min_alpha_chars: int,
    max_feature_name_length: int,
) -> Optional[str]:
    n = str(feature_name or "").strip()
    n_low = n.lower()

    if not n:
        return "empty_name"
    if n_low in _GENERIC_BAD_NAMES:
        return "generic_name"
    if len(n) > max_feature_name_length:
        return "name_too_long"
    if _DIM_FRAGMENT_RE.match(n):
        return "dimension_fragment_name"
    if _count_alpha_chars(n) < min_alpha_chars:
        return "insufficient_alpha_chars"
    if any(rx.search(n) for rx in _PSEUDO_NAME_RES):
        return "pseudo_feature_name"
    if n.count("/") >= 3 and len(re.findall(r"\d+(?:[.,]\d+)?", n)) >= 4:
        return "concatenated_measurement_chain"
    return None


def _dedupe_features(features: List[NormalizedFeature]) -> List[NormalizedFeature]:
    out: List[NormalizedFeature] = []
    seen: set[str] = set()
    for f in features:
        key = (
            f"{str(f.name).strip().lower()}|{str(f.value).strip()}|{str(f.unit).strip().lower()}|"
            f"{str(f.normalized_value)}|{str(f.normalized_unit).strip().lower()}"
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


def _normalize_product_name(raw_name: Any) -> Any:
    n = _clean_text(str(raw_name or ""))
    if not n:
        return raw_name
    n = n.replace("_", " ")
    n = re.sub(r"^(?:[A-Z]{2}\s+)?(?:DS\s+)?", "", n, flags=re.IGNORECASE)
    n = re.sub(r"\b(?:datenblatt|datasheet|data\s*sheet|spec(?:ification)?s?|pdf)\b", "", n, flags=re.IGNORECASE)
    n = re.sub(r"\s{2,}", " ", n).strip(" -_,.;")
    return n or raw_name


def _filter_and_repair_features(
    *,
    features: List[NormalizedFeature],
    remove_nonsensical_features: bool,
    repair_feature_names: bool,
    min_alpha_chars: int,
    max_feature_name_length: int,
) -> Tuple[List[NormalizedFeature], int, Dict[str, int]]:
    repaired = 0
    reason_counts: Dict[str, int] = {}
    out: List[NormalizedFeature] = []

    for f in features:
        name_before = str(f.name or "")
        name_after = _repair_feature_name(name_before) if repair_feature_names else _clean_text(name_before)
        if name_after != name_before:
            repaired += 1

        reason = _feature_quality_reason(
            feature_name=name_after,
            min_alpha_chars=min_alpha_chars,
            max_feature_name_length=max_feature_name_length,
        )
        if reason and remove_nonsensical_features:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            continue

        source_clean = _clean_text(f.source)
        out.append(
            NormalizedFeature(
                name=name_after or f.name,
                value=f.value,
                unit=f.unit,
                normalized_value=f.normalized_value,
                normalized_unit=f.normalized_unit,
                source=source_clean,
            )
        )

    return out, repaired, reason_counts


def run_feature_claim_extraction_quality_gate(
    *,
    product_profile: Optional[Dict[str, Any]],
    product_profile_path: Optional[str],
    provider: str,
    max_context_chars: int,
    remove_nonsensical_features: bool,
    repair_feature_names: bool,
    min_alpha_chars: int,
    max_feature_name_length: int,
    allow_llm_fallback: bool,
    user_root: Path,
    work_root: Path,
) -> tuple[ProductProfile, FeatureClaimQualityReport]:
    profile = _load_profile(
        product_profile=product_profile,
        product_profile_path=product_profile_path,
        user_root=user_root,
        work_root=work_root,
    )

    input_feature_count = len(profile.normalized_features or [])
    llm_failed = False
    llm_error_msg = ""
    try:
        llm_result = _llm_clean_features(
            provider=provider,
            profile=profile,
            max_context_chars=max_context_chars,
        )
    except HTTPException as exc:
        llm_failed = True
        llm_error_msg = str(getattr(exc, "detail", exc))
        if not allow_llm_fallback:
            raise HTTPException(status_code=502, detail=f"quality_gate_without_fallback_failed: {llm_error_msg}") from exc
        llm_result = {}

    llm_features_raw = llm_result.get("normalized_features") if isinstance(llm_result.get("normalized_features"), list) else []
    llm_perf_raw = llm_result.get("performance_parameters") if isinstance(llm_result.get("performance_parameters"), list) else []
    quality_notes = [str(x).strip() for x in (llm_result.get("quality_notes") or []) if str(x or "").strip()]

    cleaned_features_in = [NormalizedFeature(**x) for x in llm_features_raw if isinstance(x, dict)]
    source_features_in = list(profile.normalized_features or [])
    llm_performance_in = [NormalizedFeature(**x) for x in llm_perf_raw if isinstance(x, dict)]
    source_performance_in = list(profile.performance_parameters or [])

    repaired = 0
    reason_counts: Dict[str, int] = {}

    cleaned_features, repaired_a, reasons_a = _filter_and_repair_features(
        features=cleaned_features_in,
        remove_nonsensical_features=remove_nonsensical_features,
        repair_feature_names=repair_feature_names,
        min_alpha_chars=min_alpha_chars,
        max_feature_name_length=max_feature_name_length,
    )
    cleaned_features = _dedupe_features(cleaned_features)
    repaired += repaired_a
    for k, v in reasons_a.items():
        reason_counts[k] = reason_counts.get(k, 0) + v

    if not cleaned_features:
        fallback_features, repaired_f, reasons_f = _filter_and_repair_features(
            features=source_features_in,
            remove_nonsensical_features=remove_nonsensical_features,
            repair_feature_names=repair_feature_names,
            min_alpha_chars=min_alpha_chars,
            max_feature_name_length=max_feature_name_length,
        )
        cleaned_features = _dedupe_features(fallback_features)
        repaired += repaired_f
        for k, v in reasons_f.items():
            reason_counts[k] = reason_counts.get(k, 0) + v

    filtered_performance, repaired_b, reasons_b = _filter_and_repair_features(
        features=llm_performance_in,
        remove_nonsensical_features=remove_nonsensical_features,
        repair_feature_names=repair_feature_names,
        min_alpha_chars=min_alpha_chars,
        max_feature_name_length=max_feature_name_length,
    )
    filtered_performance = _dedupe_features(filtered_performance)
    repaired += repaired_b
    for k, v in reasons_b.items():
        reason_counts[k] = reason_counts.get(k, 0) + v

    if not filtered_performance:
        fallback_perf, repaired_c, reasons_c = _filter_and_repair_features(
            features=source_performance_in,
            remove_nonsensical_features=remove_nonsensical_features,
            repair_feature_names=repair_feature_names,
            min_alpha_chars=min_alpha_chars,
            max_feature_name_length=max_feature_name_length,
        )
        fallback_perf = _dedupe_features(fallback_perf)
        filtered_performance = fallback_perf
        repaired += repaired_c
        for k, v in reasons_c.items():
            reason_counts[k] = reason_counts.get(k, 0) + v

    if not filtered_performance:
        inferred_perf = [f for f in cleaned_features if f.normalized_value is not None and str(f.unit or "").strip()]
        filtered_performance = _dedupe_features(inferred_perf[:24])

    warnings = list(profile.extraction_warnings or [])
    dropped = input_feature_count - len(cleaned_features)
    warnings.append(f"Quality gate corrected by LLM provider={provider}.")
    if llm_failed:
        warnings.append(f"Quality gate LLM fallback used: {llm_error_msg}")
    if dropped > 0:
        warnings.append(f"Quality gate removed {dropped} nonsensical normalized_features.")
    if repaired > 0:
        warnings.append(f"Quality gate repaired {repaired} feature names.")
    for note in quality_notes:
        warnings.append(f"Quality gate note: {note}")

    report = FeatureClaimQualityReport(
        total_input_features=input_feature_count,
        total_output_features=len(cleaned_features),
        dropped_features=max(0, dropped),
        repaired_features=repaired,
        drop_reasons=reason_counts,
        notes=[],
    )
    report.notes.append("post_llm_guard=enabled")
    report.notes.append(f"llm_provider={provider}")
    report.notes.append(f"llm_fallback_used={str(llm_failed).lower()}")
    if not llm_performance_in and filtered_performance:
        report.notes.append("performance_parameters_fallback=source_or_inferred")

    metadata = dict(profile.metadata or {})
    metadata["product_name"] = _normalize_product_name(metadata.get("product_name"))

    cleaned_profile = ProductProfile(
        schema_version=profile.schema_version,
        provider=profile.provider,
        product_category=profile.product_category,
        metadata=metadata,
        normalized_features=cleaned_features,
        performance_parameters=filtered_performance,
        price_indicators=profile.price_indicators,
        claims=profile.claims,
        differentiators=profile.differentiators,
        target_segments=profile.target_segments,
        use_cases=profile.use_cases,
        extraction_warnings=warnings,
    )
    return cleaned_profile, report
