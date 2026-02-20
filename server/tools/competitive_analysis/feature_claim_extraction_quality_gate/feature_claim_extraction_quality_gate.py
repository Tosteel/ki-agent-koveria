from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException

from server.services.llm_ionos import IonosLLM
from server.services.llm_openai import LlmOpenai
from server.services.llm_perplexity import LlmPerplexity
from server.tools.competitive_analysis.feature_claim_extraction.models import (
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
        "Regeln: 1) Unsinnige Feature-Namen entfernen (z.B. reine Dimensionsfragmente wie '457 × 350 ×', generische Namen wie 'measurement'). "
        "2) Abgeschnittene Namen sinnvoll reparieren. 3) Keine neuen Fakten erfinden. "
        "4) Werte/Einheiten nur aus vorhandenen Daten übernehmen. "
        "5) Nur gültiges JSON gemäß Schema."
    )
    user = (
        "Eingabeprofil:\n"
        + json.dumps(
            {
                "product_category": profile.product_category,
                "normalized_features": [f.model_dump() for f in (profile.normalized_features or [])],
                "performance_parameters": [f.model_dump() for f in (profile.performance_parameters or [])],
            },
            ensure_ascii=False,
        )[:max_context_chars]
    )

    if p in {"openai", "perplexity"}:
        client = LlmOpenai() if p == "openai" else LlmPerplexity()
        if not client.enabled():
            raise HTTPException(status_code=400, detail=f"{p} not configured for feature_claim_extraction_quality_gate.")
        fmt = {"type": "json_schema", "name": "feature_claim_quality_gate", "schema": schema, "strict": False}
        try:
            resp = client._call(
                input_messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                text_format=fmt,
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"{p} quality gate failed: {exc}") from exc
        parsed = _parse_json_strictish(_openai_extract_output_text(resp))
        if not parsed:
            raise HTTPException(status_code=502, detail=f"{p} quality gate returned invalid JSON.")
        return parsed

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
    return None


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
    llm_result = _llm_clean_features(
        provider=provider,
        profile=profile,
        max_context_chars=max_context_chars,
    )

    llm_features_raw = llm_result.get("normalized_features") if isinstance(llm_result.get("normalized_features"), list) else []
    llm_perf_raw = llm_result.get("performance_parameters") if isinstance(llm_result.get("performance_parameters"), list) else []
    quality_notes = [str(x).strip() for x in (llm_result.get("quality_notes") or []) if str(x or "").strip()]

    cleaned_features = [NormalizedFeature(**x) for x in llm_features_raw if isinstance(x, dict)]
    filtered_performance = [NormalizedFeature(**x) for x in llm_perf_raw if isinstance(x, dict)]

    repaired = 0
    reason_counts: Dict[str, int] = {}

    warnings = list(profile.extraction_warnings or [])
    dropped = input_feature_count - len(cleaned_features)
    warnings.append(f"Quality gate corrected by LLM provider={provider}.")
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
    report.notes.append("post_llm_guard=disabled")
    report.notes.append(f"llm_provider={provider}")

    cleaned_profile = ProductProfile(
        schema_version=profile.schema_version,
        provider=profile.provider,
        product_category=profile.product_category,
        metadata=profile.metadata,
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
