from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException

from server.services.llm_ionos import IonosLLM
from server.services.llm_openai import LlmOpenai
from server.services.llm_perplexity import LlmPerplexity
from server.workflows.competitive_analysis.backup.competitor_profile_extraction.competitor_profile_extraction import (
    _fetch_page,
    _openai_web_search_urls,
    _perplexity_web_search_urls,
)

from .models import FeatureMatrixGapAnalysisQualityReport


def _resolve_input_path(path: str, user_root: Path, work_root: Path) -> Path:
    raw = str(path or "").strip().lstrip("/")
    p = Path(raw)
    if not raw or p.is_absolute() or ".." in p.parts:
        raise HTTPException(status_code=400, detail=f"Invalid path: {path}")

    uploads_root = (user_root / "uploads").resolve()
    uploads_root.mkdir(parents=True, exist_ok=True)

    candidates: List[Path] = []
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


def _load_feature_matrix_gap(
    *,
    feature_matrix_gap: Optional[Dict[str, Any]],
    feature_matrix_gap_path: Optional[str],
    user_root: Path,
    work_root: Path,
) -> Dict[str, Any]:
    payload: Dict[str, Any]
    if isinstance(feature_matrix_gap, dict) and feature_matrix_gap:
        payload = feature_matrix_gap
    else:
        p = _resolve_input_path(str(feature_matrix_gap_path or ""), user_root=user_root.resolve(), work_root=work_root.resolve())
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON in feature_matrix_gap_path: {feature_matrix_gap_path}") from exc

    if "feature_matrix_gap" in payload and isinstance(payload.get("feature_matrix_gap"), dict):
        payload = payload["feature_matrix_gap"]

    if "comparison_matrix" not in payload:
        raise HTTPException(status_code=400, detail="Invalid feature_matrix_gap payload: missing comparison_matrix")
    return payload


def _search_urls(provider: str, query: str, max_results: int, warnings: List[str]) -> List[str]:
    p = str(provider or "perplexity").strip().lower()
    if p not in {"openai", "perplexity"}:
        p = "perplexity"

    if p == "openai":
        key = os.getenv("OPENAI_API_KEY", "").strip()
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
        if not key:
            warnings.append("OPENAI_API_KEY missing; web search disabled for gap QG.")
            return []
        try:
            return _openai_web_search_urls(query, api_key=key, model=model, max_results=max_results)
        except Exception as exc:
            warnings.append(f"OpenAI web search failed for '{query}': {exc}")
            return []

    key = os.getenv("PERPLEXITY_API_KEY", "").strip()
    model = os.getenv("PERPLEXITY_MODEL", "sonar-pro").strip() or "sonar-pro"
    if not key:
        warnings.append("PERPLEXITY_API_KEY missing; web search disabled for gap QG.")
        return []
    try:
        return _perplexity_web_search_urls(query, api_key=key, model=model, max_results=max_results)
    except Exception as exc:
        warnings.append(f"Perplexity web search failed for '{query}': {exc}")
        return []


def _feature_extract_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "found": {"type": "boolean"},
            "value": {"type": "string"},
            "unit": {"type": "string"},
            "normalized_value": {"oneOf": [{"type": "number"}, {"type": "null"}]},
            "evidence": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["found", "value", "unit", "normalized_value", "evidence", "confidence"],
    }


def _llm_extract_feature_from_text(
    *,
    provider: str,
    competitor: str,
    feature: str,
    url: str,
    text: str,
    warnings: List[str],
) -> Dict[str, Any]:
    p = str(provider or "perplexity").strip().lower()
    if p not in {"openai", "perplexity", "ionos"}:
        p = "perplexity"

    schema = _feature_extract_schema()
    context = (text or "")[:10000]

    system = (
        "Extrahiere für das angegebene Produkt exakt den angefragten Feature-Wert aus dem Text. "
        "Wenn nicht klar vorhanden, found=false. Keine Schätzung. Antworte nur JSON gemäß Schema."
    )
    user = json.dumps(
        {
            "competitor": competitor,
            "target_feature": feature,
            "source_url": url,
            "page_excerpt": context,
        },
        ensure_ascii=False,
    )

    if p in {"openai", "perplexity"}:
        client = LlmOpenai() if p == "openai" else LlmPerplexity()
        if not client.enabled():
            warnings.append(f"{p} not configured; skipped LLM extraction for '{competitor} / {feature}'.")
            return {}
        rf = {
            "type": "json_schema",
            "name": "feature_extract",
            "schema": schema,
            "strict": False,
        }
        try:
            resp = client._call(  # type: ignore[attr-defined]
                input_messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                text_format=rf,
            )
            return _parse_json_strictish(_extract_any_output_text(resp))
        except Exception as exc:
            warnings.append(f"{p} LLM extraction failed for '{competitor} / {feature}': {exc}")
            return {}

    c2 = IonosLLM()
    if not c2.enabled():
        warnings.append(f"IONOS not configured; skipped LLM extraction for '{competitor} / {feature}'.")
        return {}
    try:
        completion = c2.chat_completions(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "feature_extract",
                    "schema": schema,
                    "strict": True,
                },
            },
        )
        return _parse_json_strictish(c2.extract_text(completion))
    except Exception as exc:
        warnings.append(f"IONOS LLM extraction failed for '{competitor} / {feature}': {exc}")
        return {}


def _to_float(v: Any) -> float | None:
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v or "").strip().replace(",", ".")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def _norm_unit(u: str) -> str:
    x = str(u or "").strip().lower().replace(" ", "")
    table = {
        "mm": "mm",
        "millimeter": "mm",
        "millimetre": "mm",
        "cm": "cm",
        "m": "m",
        "meter": "m",
        "metre": "m",
        "kg": "kg",
        "g": "g",
        "l": "L",
        "lt": "L",
        "liter": "L",
        "litre": "L",
        "ml": "mL",
        "w": "W",
        "kw": "kW",
        "bar": "bar",
        "mpa": "MPa",
        "°c": "°C",
        "c": "°C",
    }
    return table.get(x, str(u or "").strip())


def _feature_kind(name: str) -> str:
    n = str(name or "").lower()
    if any(k in n for k in ["abmess", "breite", "höhe", "hoehe", "tiefe", "länge", "laenge", "auslauf"]):
        return "length"
    if "gewicht" in n:
        return "mass"
    if any(k in n for k in ["wassertank", "milchbehälter", "milchbehaelter", "kapazität", "kapazitaet"]):
        return "volume"
    if "leistung" in n:
        return "power"
    if "druck" in n:
        return "pressure"
    if "temperatur" in n:
        return "temperature"
    return "generic"


def _normalize_feature_value(feature: str, value: str, unit: str, llm_norm: Any) -> Tuple[float | None, str]:
    """
    Return (normalized_value, normalized_unit) with deterministic unit handling.
    This prevents mismatches like 387 mm -> 38.7 with unit still mm.
    """
    raw = _to_float(value)
    llm_num = _to_float(llm_norm)
    u = _norm_unit(unit)
    kind = _feature_kind(feature)

    if raw is None and llm_num is not None:
        return llm_num, u
    if raw is None:
        return None, u

    # Length-like features -> canonical meters
    if kind == "length":
        if u == "mm":
            return raw / 1000.0, "m"
        if u == "cm":
            return raw / 100.0, "m"
        if u == "m":
            return raw, "m"
        return raw, u

    # Mass -> canonical kg
    if kind == "mass":
        if u == "g":
            return raw / 1000.0, "kg"
        if u == "kg":
            return raw, "kg"
        return raw, u

    # Volumes -> canonical L
    if kind == "volume":
        if u == "mL":
            return raw / 1000.0, "L"
        if u == "L":
            return raw, "L"
        return raw, u

    # Power -> canonical W
    if kind == "power":
        if u == "kW":
            return raw * 1000.0, "W"
        if u == "W":
            return raw, "W"
        return raw, u

    # Pressure -> canonical bar
    if kind == "pressure":
        if u == "MPa":
            return raw * 10.0, "bar"
        if u == "bar":
            return raw, "bar"
        return raw, u

    # Temperature -> canonical °C
    if kind == "temperature":
        if u in {"°C", "C"}:
            return raw, "°C"
        return raw, u

    # Generic fallback: keep parsed value in declared unit.
    return raw, u


def _norm_feature_name(name: str) -> str:
    t = re.sub(r"[^a-z0-9]+", " ", str(name or "").lower()).strip()
    return re.sub(r"\s+", " ", t)


def _presence_ratio(rows: List[Dict[str, Any]], feature: str) -> float:
    valid_rows = [r for r in rows if isinstance(r, dict)]
    if not valid_rows:
        return 0.0
    key = _norm_feature_name(feature)
    present = 0
    for r in valid_rows:
        features = r.get("features") if isinstance(r.get("features"), list) else []
        hit = None
        for c in features:
            if not isinstance(c, dict):
                continue
            if _norm_feature_name(str(c.get("feature") or "")) == key:
                hit = c
                break
        if hit and bool(hit.get("present")):
            present += 1
    return present / len(valid_rows)


def _recompute_gap_ratios(payload: Dict[str, Any], warnings: List[str]) -> int:
    cm = payload.get("comparison_matrix") if isinstance(payload.get("comparison_matrix"), dict) else {}
    rows = cm.get("competitor_rows") if isinstance(cm.get("competitor_rows"), list) else []
    if not rows:
        return 0

    g = payload.get("gaps_and_usps") if isinstance(payload.get("gaps_and_usps"), dict) else {}
    changed = 0

    for key in ("gaps", "prioritized_gaps"):
        items = g.get(key)
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, dict):
                continue
            feat = str(it.get("feature") or "").strip()
            if not feat:
                continue
            ratio = round(_presence_ratio(rows, feat), 4)
            prev = it.get("market_presence_ratio")
            try:
                prev_f = float(prev)
            except Exception:
                prev_f = None
            if prev_f is None or abs(prev_f - ratio) > 1e-9:
                it["market_presence_ratio"] = ratio
                changed += 1

    if changed > 0:
        warnings.append(
            f"Recomputed market_presence_ratio after feature_matrix_gap_qg backfill (updated_items={changed})."
        )
    return changed


def run_feature_matrix_gap_analysis_quality_gate(
    *,
    feature_matrix_gap: Optional[Dict[str, Any]],
    feature_matrix_gap_path: Optional[str],
    provider: str,
    max_missing_features_per_competitor: int,
    max_urls_per_feature: int,
    max_llm_calls: int,
    min_confidence: float,
    verbose_progress: bool,
    user_root: Path,
    work_root: Path,
) -> Tuple[Dict[str, Any], FeatureMatrixGapAnalysisQualityReport]:
    payload = _load_feature_matrix_gap(
        feature_matrix_gap=feature_matrix_gap,
        feature_matrix_gap_path=feature_matrix_gap_path,
        user_root=user_root,
        work_root=work_root,
    )

    cm = payload.get("comparison_matrix") if isinstance(payload.get("comparison_matrix"), dict) else {}
    rows = cm.get("competitor_rows") if isinstance(cm.get("competitor_rows"), list) else []

    warnings = list(payload.get("extraction_warnings") or [])

    total_missing = 0
    llm_calls = 0
    filled = 0
    skipped = 0
    total_rows = len([r for r in rows if isinstance(r, dict)])
    processed_rows = 0

    for row in rows:
        if not isinstance(row, dict):
            continue
        processed_rows += 1
        cname = str(row.get("competitor") or "").strip()
        features = row.get("features") if isinstance(row.get("features"), list) else []
        missing_cells: List[Dict[str, Any]] = []
        for cell in features:
            if not isinstance(cell, dict):
                continue
            if bool(cell.get("present")):
                continue
            missing_cells.append(cell)

        if not missing_cells:
            continue

        total_missing += len(missing_cells)
        scoped_missing = missing_cells[: max(1, int(max_missing_features_per_competitor))]
        for idx, cell in enumerate(scoped_missing, start=1):
            if llm_calls >= max_llm_calls:
                skipped += 1
                continue

            feat = str(cell.get("feature") or "").strip()
            if not feat:
                skipped += 1
                continue

            if verbose_progress:
                print(
                    (
                        f"[feature_matrix_gap_qg] row {processed_rows}/{total_rows} "
                        f"competitor='{cname}' feature {idx}/{len(scoped_missing)} '{feat}' "
                        f"(llm_calls={llm_calls}/{max_llm_calls})"
                    ),
                    flush=True,
                )

            query = f"{cname} {feat}".strip()
            urls = _search_urls(provider=provider, query=query, max_results=max_urls_per_feature, warnings=warnings)
            if not urls:
                skipped += 1
                continue

            best: Dict[str, Any] = {}
            best_url = ""
            best_conf = -1.0

            for u in urls[: max(1, int(max_urls_per_feature))]:
                if llm_calls >= max_llm_calls:
                    break
                try:
                    _title, text, _html, _ctype = _fetch_page(u)
                except Exception:
                    continue
                if not text or len(text) < 120:
                    continue
                llm_calls += 1
                ext = _llm_extract_feature_from_text(
                    provider=provider,
                    competitor=cname,
                    feature=feat,
                    url=u,
                    text=text,
                    warnings=warnings,
                )
                if not ext:
                    continue
                conf = float(ext.get("confidence") or 0.0)
                if bool(ext.get("found")) and conf > best_conf:
                    best = ext
                    best_conf = conf
                    best_url = u

            if not best:
                skipped += 1
                continue

            if best_conf < float(min_confidence):
                skipped += 1
                continue

            val = str(best.get("value") or "").strip()
            unit = str(best.get("unit") or "").strip()
            evidence = str(best.get("evidence") or "").strip()
            norm_val, norm_unit = _normalize_feature_value(
                feature=feat,
                value=val,
                unit=unit,
                llm_norm=best.get("normalized_value"),
            )

            cell["value"] = f"{val} {unit}".strip()
            cell["normalized_value"] = norm_val
            cell["unit"] = norm_unit
            cell["present"] = True
            cell["evidence_url"] = best_url
            cell["present_note"] = "added_by_feature_matrix_gap_qg"
            if evidence:
                cell["evidence"] = evidence

            warnings.append(f"Feature backfill: '{cname}' -> '{feat}' from {best_url} (conf={best_conf:.2f}).")
            filled += 1
            if verbose_progress:
                print(
                    f"[feature_matrix_gap_qg] backfilled competitor='{cname}' feature='{feat}' from {best_url} (conf={best_conf:.2f})",
                    flush=True,
                )

    payload["comparison_matrix"] = cm
    _recompute_gap_ratios(payload, warnings)
    payload["extraction_warnings"] = warnings

    report = FeatureMatrixGapAnalysisQualityReport(
        total_missing_features_scanned=total_missing,
        llm_calls=llm_calls,
        filled_features=filled,
        skipped_features=skipped,
        notes=[
            "feature_matrix_gap_analysis_qg=enabled",
            f"provider={provider}",
            f"min_confidence={min_confidence}",
            f"max_llm_calls={max_llm_calls}",
        ],
    )

    return payload, report
