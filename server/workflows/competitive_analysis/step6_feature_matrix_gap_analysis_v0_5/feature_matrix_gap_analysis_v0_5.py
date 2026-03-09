from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional

from fastapi import HTTPException

from server.services.llm_ionos import IonosLLM
from server.services.llm_openai import LlmOpenai
from server.services.llm_perplexity import LlmPerplexity

from .models import (
    ClusterAssignmentV05,
    ComparisonMatrixV05,
    CompetitorRowV05,
    FeatureMatrixGapAnalysisV05Result,
    GapItemV05,
    GapsAndUspsV05,
    PerformanceCellV05,
    PriceCellV05,
    SoftFeatureCellV05,
    UspItemV05,
)


def _safe_list_str(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
    for v in values:
        s = str(v or "").strip()
        if s:
            out.append(s)
    return out


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
    start = t.find("{")
    end = t.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(t[start : end + 1])
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


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


def _load_json_obj(
    *,
    inline_obj: Optional[Dict[str, Any]],
    path: Optional[str],
    root_keys: List[str],
    user_root: Path,
    work_root: Path,
) -> Dict[str, Any]:
    payload: Dict[str, Any]
    if isinstance(inline_obj, dict) and inline_obj:
        payload = inline_obj
    else:
        p = _resolve_input_path(str(path or ""), user_root=user_root.resolve(), work_root=work_root.resolve())
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON in path: {path}") from exc

    for key in root_keys:
        if key in payload and isinstance(payload.get(key), dict):
            payload = payload[key]
            break
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail=f"Invalid payload for keys: {root_keys}")
    return payload


def _norm(s: str) -> str:
    return " ".join(str(s or "").strip().lower().replace("_", " ").split())


def _is_price_context_label(label: str) -> bool:
    k = _norm(label)
    if not k:
        return False
    tokens = ["preis", "price", "uvp", "msrp", "rrp", "list price", "retail price"]
    return any(t in k for t in tokens)


def _canonical_price_context(label: str) -> str:
    k = _norm(label)
    if any(t in k for t in ["uvp", "msrp", "rrp", "list price", "retail price"]):
        return "UVP"
    return "Preis"


def _feature_name(item: Any, *, fallback: str = "") -> str:
    if isinstance(item, dict):
        return str(item.get("name") or item.get("feature") or item.get("context") or fallback or "").strip()
    return str(item or fallback or "").strip()


def _collect_performance_dims(product: Dict[str, Any], competitors: List[Dict[str, Any]]) -> List[str]:
    dims = set()
    metric_dims_norm = set(_norm(x) for x in _collect_metric_dims(product, competitors))
    for x in (product.get("performance_parameters") or []):
        n = _feature_name(x)
        if n and _norm(n) not in metric_dims_norm:
            dims.add(n)
    for c in competitors:
        for x in (c.get("performance_parameters") or []):
            n = _feature_name(x)
            if n and _norm(n) not in metric_dims_norm:
                dims.add(n)
    return sorted(dims, key=lambda s: s.lower())


_LOWER_BETTER_NAME_HINTS = (
    "width",
    "height",
    "depth",
    "length",
    "thickness",
    "diameter",
    "size",
    "footprint",
    "weight",
    "mass",
)

_METRIC_UNIT_HINTS = {
    "mm",
    "cm",
    "m",
    "km",
    "in",
    "inch",
    "ft",
    "g",
    "kg",
    "lb",
    "oz",
    "ml",
    "l",
    "bar",
    "psi",
    "pa",
    "kpa",
    "mpa",
    "w",
    "kw",
    "hz",
    "khz",
    "mhz",
}


def _is_metric_feature_name_unit(name: str, unit: str) -> bool:
    n = _norm(name)
    u = _norm(unit)
    if not n:
        return False
    if any(tok in n for tok in _LOWER_BETTER_NAME_HINTS):
        return True
    return u in _METRIC_UNIT_HINTS


def _collect_metric_dims(product: Dict[str, Any], competitors: List[Dict[str, Any]]) -> List[str]:
    dims = set()

    def _from_item_list(items: List[Dict[str, Any]]) -> None:
        for x in items:
            if not isinstance(x, dict):
                continue
            n = _feature_name(x)
            if not n:
                continue
            if _is_metric_feature_name_unit(n, str(x.get("unit") or "")):
                dims.add(n)

    # Primary source: explicit metric_features from step 2+/5+.
    _from_item_list(product.get("metric_features") or [])
    for c in competitors:
        _from_item_list(c.get("metric_features") or [])

    # Backward compatible fallback: derive from performance_parameters if needed.
    if not dims:
        _from_item_list(product.get("performance_parameters") or [])
        for c in competitors:
            _from_item_list(c.get("performance_parameters") or [])

    return sorted(dims, key=lambda s: s.lower())


def _collect_price_dims(product: Dict[str, Any], competitors: List[Dict[str, Any]]) -> List[str]:
    dims = set()
    items = list(product.get("price_indicators") or [])
    for c in competitors:
        items.extend(list(c.get("price_indicators") or []))
    for x in items:
        if not isinstance(x, dict):
            continue
        raw_ctx = _feature_name(x, fallback="")
        has_value = _to_float(x.get("value")) is not None or bool(str(x.get("raw") or "").strip())
        if raw_ctx and _is_price_context_label(raw_ctx):
            dims.add(_canonical_price_context(raw_ctx))
        elif has_value:
            dims.add("Preis")
    if not dims:
        dims = {"Preis", "UVP"}
    return sorted(dims, key=lambda s: s.lower())


def _collect_soft_dims(product: Dict[str, Any], competitors: List[Dict[str, Any]]) -> List[str]:
    dims = set()
    for x in (product.get("soft_features") or []):
        n = _feature_name(x)
        if n:
            dims.add(n)
    for c in competitors:
        for x in (c.get("soft_features") or []):
            n = _feature_name(x)
            if n:
                dims.add(n)
    return sorted(dims, key=lambda s: s.lower())


def _to_float(v: Any) -> float | None:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().replace(",", ".")
        try:
            return float(s)
        except Exception:
            return None
    return None


def _avg_price_from_indicators(items: List[Dict[str, Any]]) -> float | None:
    vals: List[float] = []
    for x in items:
        if not isinstance(x, dict):
            continue
        fv = _to_float(x.get("value"))
        if fv is not None:
            vals.append(fv)
    return round(mean(vals), 4) if vals else None


def _perf_map(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for x in items:
        if not isinstance(x, dict):
            continue
        n = _feature_name(x)
        if not n:
            continue
        out[_norm(n)] = x
    return out


def _metric_map(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for x in items:
        if not isinstance(x, dict):
            continue
        n = _feature_name(x)
        if not n:
            continue
        out[_norm(n)] = x
    return out


def _price_map(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for x in items:
        if not isinstance(x, dict):
            continue
        c = _feature_name(x, fallback="")
        has_value = _to_float(x.get("value")) is not None or bool(str(x.get("raw") or "").strip())
        if c and _is_price_context_label(c):
            canonical = _canonical_price_context(c)
        elif has_value:
            canonical = "Preis"
        else:
            continue
        out[_norm(canonical)] = x
    return out


def _soft_map(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for x in items:
        if not isinstance(x, dict):
            continue
        n = _feature_name(x)
        if not n:
            continue
        out[_norm(n)] = x
    return out


_UNIT_FACTOR: Dict[str, tuple[str, float]] = {
    # length -> m
    "mm": ("length_m", 0.001),
    "cm": ("length_m", 0.01),
    "m": ("length_m", 1.0),
    "km": ("length_m", 1000.0),
    "in": ("length_m", 0.0254),
    "inch": ("length_m", 0.0254),
    "ft": ("length_m", 0.3048),
    # mass -> kg
    "mg": ("mass_kg", 0.000001),
    "g": ("mass_kg", 0.001),
    "kg": ("mass_kg", 1.0),
    "lb": ("mass_kg", 0.45359237),
    "oz": ("mass_kg", 0.028349523125),
    # volume -> l
    "ml": ("volume_l", 0.001),
    "cl": ("volume_l", 0.01),
    "dl": ("volume_l", 0.1),
    "l": ("volume_l", 1.0),
    "lt": ("volume_l", 1.0),
    # pressure -> bar
    "bar": ("pressure_bar", 1.0),
    "kpa": ("pressure_bar", 0.01),
    "mpa": ("pressure_bar", 10.0),
    "pa": ("pressure_bar", 0.00001),
    "psi": ("pressure_bar", 0.0689475729),
    # power -> w
    "w": ("power_w", 1.0),
    "kw": ("power_w", 1000.0),
    # frequency -> hz
    "hz": ("frequency_hz", 1.0),
    "khz": ("frequency_hz", 1000.0),
    "mhz": ("frequency_hz", 1000000.0),
}


def _to_metric_value(v: Any, unit: str) -> tuple[Optional[float], Optional[str]]:
    fv = _to_float(v)
    if fv is None:
        return None, None
    u = _norm(unit)
    if not u:
        return fv, None
    mapped = _UNIT_FACTOR.get(u)
    if not mapped:
        return fv, u
    family, factor = mapped
    return fv * factor, family


def _preferred_direction_for_metric(feature_name: str) -> Optional[str]:
    n = _norm(feature_name)
    if any(tok in n for tok in _LOWER_BETTER_NAME_HINTS):
        return "lower"
    return None


def _build_row(
    *,
    competitor: str,
    cluster: str,
    perf_items: List[Dict[str, Any]],
    metric_items: List[Dict[str, Any]],
    price_items: List[Dict[str, Any]],
    soft_items: List[Dict[str, Any]],
    perf_dims: List[str],
    metric_dims: List[str],
    price_dims: List[str],
    soft_dims: List[str],
) -> CompetitorRowV05:
    pmap = _perf_map(perf_items)
    mmap = _metric_map(metric_items)
    prmap = _price_map(price_items)
    smap = _soft_map(soft_items)

    perf_cells: List[PerformanceCellV05] = []
    perf_present = 0
    for d in perf_dims:
        m = pmap.get(_norm(d))
        if not m:
            perf_cells.append(PerformanceCellV05(name=d, value=None, unit="", present=False))
            continue
        v = m.get("value")
        present = v not in (None, "")
        if present:
            perf_present += 1
        perf_cells.append(
            PerformanceCellV05(
                name=d,
                value=v,
                unit=str(m.get("unit") or "").strip(),
                present=present,
            )
        )

    metric_cells: List[PerformanceCellV05] = []
    metric_present = 0
    for d in metric_dims:
        m = mmap.get(_norm(d)) or pmap.get(_norm(d))
        if not m:
            metric_cells.append(PerformanceCellV05(name=d, value=None, unit="", present=False))
            continue
        v = m.get("value")
        present = v not in (None, "")
        if present:
            metric_present += 1
        metric_cells.append(
            PerformanceCellV05(
                name=d,
                value=v,
                unit=str(m.get("unit") or "").strip(),
                present=present,
            )
        )

    price_cells: List[PriceCellV05] = []
    price_present = 0
    for d in price_dims:
        m = prmap.get(_norm(d))
        if not m:
            price_cells.append(PriceCellV05(context=d, present=False))
            continue
        v = m.get("value")
        present = (v not in (None, "")) or bool(str(m.get("raw") or "").strip())
        if present:
            price_present += 1
        price_cells.append(
            PriceCellV05(
                context=d,
                raw=str(m.get("raw") or "").strip(),
                value=v,
                currency=str(m.get("currency") or "").strip(),
                period=str(m.get("period") or "").strip(),
                present=present,
            )
        )

    soft_cells: List[SoftFeatureCellV05] = []
    soft_present = 0
    for d in soft_dims:
        m = smap.get(_norm(d))
        available = bool(m.get("available")) if isinstance(m, dict) else False
        if available:
            soft_present += 1
        soft_cells.append(SoftFeatureCellV05(name=d, available=available))

    cov_perf = (perf_present / max(1, len(perf_dims))) if perf_dims else 0.0
    cov_metric = (metric_present / max(1, len(metric_dims))) if metric_dims else 0.0
    cov_price = (price_present / max(1, len(price_dims))) if price_dims else 0.0
    cov_soft = (soft_present / max(1, len(soft_dims))) if soft_dims else 0.0
    coverage_components = [cov_perf, cov_price, cov_soft]
    if metric_dims:
        coverage_components.append(cov_metric)
    coverage = sum(coverage_components) / max(1, len(coverage_components))

    avg_price = _avg_price_from_indicators(price_items)
    value_score = coverage
    if avg_price is not None and avg_price > 0:
        value_score = coverage / (1.0 + (avg_price / 10000.0))

    return CompetitorRowV05(
        competitor=competitor,
        cluster=cluster,
        performance_parameters=perf_cells,
        metric_features=metric_cells,
        price_indicators=price_cells,
        soft_features=soft_cells,
        avg_price=avg_price,
        value_score=round(value_score, 4),
    )


def _presence_ratio_perf(rows: List[CompetitorRowV05], dim: str) -> float:
    if not rows:
        return 0.0
    k = _norm(dim)
    present = 0
    for r in rows:
        c = next((x for x in r.performance_parameters if _norm(x.name) == k), None)
        if c and c.present:
            present += 1
    return present / len(rows)


def _presence_ratio_soft(rows: List[CompetitorRowV05], dim: str) -> float:
    if not rows:
        return 0.0
    k = _norm(dim)
    present = 0
    for r in rows:
        c = next((x for x in r.soft_features if _norm(x.name) == k), None)
        if c and c.available:
            present += 1
    return present / len(rows)


def _presence_ratio_metric(rows: List[CompetitorRowV05], dim: str) -> float:
    if not rows:
        return 0.0
    k = _norm(dim)
    present = 0
    for r in rows:
        c = next((x for x in r.metric_features if _norm(x.name) == k), None)
        if c and c.present:
            present += 1
    return present / len(rows)


def _presence_ratio_price(rows: List[CompetitorRowV05], dim: str) -> float:
    if not rows:
        return 0.0
    k = _norm(dim)
    present = 0
    for r in rows:
        c = next((x for x in r.price_indicators if _norm(x.context) == k), None)
        if c and c.present:
            present += 1
    return present / len(rows)


def _presence_ratio_for_group(
    group: str,
    feature: str,
    perf_dims: List[str],
    metric_dims: List[str],
    soft_dims: List[str],
    price_dims: List[str],
    comp_rows: List[CompetitorRowV05],
) -> float:
    k = _norm(feature)
    if group == "performance_parameters":
        d = next((x for x in perf_dims if _norm(x) == k), feature)
        return _presence_ratio_perf(comp_rows, d)
    if group == "metric_features":
        d = next((x for x in metric_dims if _norm(x) == k), feature)
        return _presence_ratio_metric(comp_rows, d)
    if group == "soft_features":
        d = next((x for x in soft_dims if _norm(x) == k), feature)
        return _presence_ratio_soft(comp_rows, d)
    d = next((x for x in price_dims if _norm(x) == k), feature)
    return _presence_ratio_price(comp_rows, d)


def _find_metric_cell(row: CompetitorRowV05, dim: str) -> Optional[PerformanceCellV05]:
    k = _norm(dim)
    return next((x for x in row.metric_features if _norm(x.name) == k), None)


def _median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    n = len(xs)
    mid = n // 2
    if n % 2 == 1:
        return xs[mid]
    return (xs[mid - 1] + xs[mid]) / 2.0


def _llm_refine_gaps_v05(provider: str, payload: Dict[str, Any], warnings: List[str]) -> Dict[str, Any]:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "gaps_text": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "feature_group": {"type": "string", "enum": ["performance_parameters", "metric_features", "price_indicators", "soft_features"]},
                        "feature": {"type": "string"},
                        "recommendation": {"type": "string"},
                    },
                    "required": ["feature_group", "feature", "recommendation"],
                },
            },
            "usps_text": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "feature_group": {"type": "string", "enum": ["performance_parameters", "metric_features", "price_indicators", "soft_features"]},
                        "feature": {"type": "string"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["feature_group", "feature", "rationale"],
                },
            },
        },
        "required": ["gaps_text", "usps_text"],
    }
    p = str(provider or "openai").strip().lower()
    if p not in {"ionos", "openai", "perplexity"}:
        p = "openai"

    system = (
        "You receive deterministic gap/usp candidates from a feature matrix. "
        "Do not invent new features and do not change status/ratios. "
        "Return only concise recommendation/rationale text for each provided candidate in JSON."
    )
    user = "Input:\n" + json.dumps(payload, ensure_ascii=False)

    if p in {"openai", "perplexity"}:
        c = LlmOpenai() if p == "openai" else LlmPerplexity()
        if not c.enabled():
            warnings.append(f"{p} not configured; heuristic gap analysis used.")
            return {}
        fmt = {"type": "json_schema", "name": "gaps_usps_v05_text", "schema": schema, "strict": False}
        try:
            resp = c._call(
                input_messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                text_format=fmt,
            )
            text = ""
            for item in resp.get("output", []):
                for cc in item.get("content", []):
                    if cc.get("type") == "output_text":
                        text += str(cc.get("text") or "")
            return _parse_json_strictish(text)
        except Exception as exc:
            warnings.append(f"{p} gap analysis failed: {exc}")
            return {}

    c2 = IonosLLM()
    if not c2.enabled():
        warnings.append("IONOS not configured; heuristic gap analysis used.")
        return {}
    try:
        completion = c2.chat_completions(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                    "json_schema": {"name": "gaps_usps_v05_text", "schema": schema, "strict": True},
                },
            )
        parsed = _parse_json_strictish(c2.extract_text(completion))
        if parsed:
            return parsed
    except Exception as exc:
        warnings.append(f"IONOS gap analysis failed: {exc}")
    return {}


def run_feature_matrix_gap_analysis_v0_5(
    *,
    product_profile: Optional[Dict[str, Any]],
    product_profile_path: Optional[str],
    competitor_profile_results: Optional[Dict[str, Any]],
    competitor_profile_results_path: Optional[str],
    provider: str = "openai",
    user_root: Path,
    work_root: Path,
) -> FeatureMatrixGapAnalysisV05Result:
    product = _load_json_obj(
        inline_obj=product_profile,
        path=product_profile_path,
        root_keys=["product_profile"],
        user_root=user_root,
        work_root=work_root,
    )
    comp_payload = _load_json_obj(
        inline_obj=competitor_profile_results,
        path=competitor_profile_results_path,
        root_keys=["competitor_profile_results", "competitor_profiles", "competitor_search_results"],
        user_root=user_root,
        work_root=work_root,
    )

    warnings = _safe_list_str(product.get("extraction_warnings")) + _safe_list_str(comp_payload.get("extraction_warnings"))

    competitors = comp_payload.get("competitors") if isinstance(comp_payload.get("competitors"), list) else []
    competitors = [c for c in competitors if isinstance(c, dict)]

    metric_dims = _collect_metric_dims(product, competitors)
    perf_dims = _collect_performance_dims(product, competitors)
    price_dims = _collect_price_dims(product, competitors)
    soft_dims = _collect_soft_dims(product, competitors)

    baseline_name = str((product.get("metadata") or {}).get("product_name") or product.get("name") or "target_product").strip()
    baseline_row = _build_row(
        competitor=baseline_name,
        cluster="target",
        perf_items=product.get("performance_parameters") if isinstance(product.get("performance_parameters"), list) else [],
        metric_items=product.get("metric_features") if isinstance(product.get("metric_features"), list) else [],
        price_items=product.get("price_indicators") if isinstance(product.get("price_indicators"), list) else [],
        soft_items=product.get("soft_features") if isinstance(product.get("soft_features"), list) else [],
        perf_dims=perf_dims,
        metric_dims=metric_dims,
        price_dims=price_dims,
        soft_dims=soft_dims,
    )

    comp_rows: List[CompetitorRowV05] = []
    for c in competitors:
        name = str(c.get("product_name") or c.get("name") or "unknown_competitor").strip()
        cluster = str(c.get("cluster") or "unknown").strip()
        row = _build_row(
            competitor=name,
            cluster=cluster,
            perf_items=c.get("performance_parameters") if isinstance(c.get("performance_parameters"), list) else [],
            metric_items=c.get("metric_features") if isinstance(c.get("metric_features"), list) else [],
            price_items=c.get("price_indicators") if isinstance(c.get("price_indicators"), list) else [],
            soft_items=c.get("soft_features") if isinstance(c.get("soft_features"), list) else [],
            perf_dims=perf_dims,
            metric_dims=metric_dims,
            price_dims=price_dims,
            soft_dims=soft_dims,
        )
        comp_rows.append(row)

    matrix = ComparisonMatrixV05(
        provider=str(provider or "openai").strip().lower(),
        baseline_product=baseline_name,
        performance_dimensions=perf_dims,
        metric_dimensions=metric_dims,
        price_dimensions=price_dims,
        soft_feature_dimensions=soft_dims,
        baseline_row=baseline_row,
        competitor_rows=comp_rows,
    )

    gaps: List[GapItemV05] = []
    usps: List[UspItemV05] = []
    market_standards: List[str] = []
    differentiators: List[str] = []

    # Deterministic baseline presence maps.
    base_perf_present = {_norm(x.name): bool(x.present) for x in baseline_row.performance_parameters}
    base_metric_present = {_norm(x.name): bool(x.present) for x in baseline_row.metric_features}
    base_soft_present = {_norm(x.name): bool(x.available) for x in baseline_row.soft_features}
    base_price_present = {_norm(x.context): bool(x.present) for x in baseline_row.price_indicators}

    # Deterministic market standards and candidate gaps/usps.
    for d in perf_dims:
        ratio = round(_presence_ratio_perf(comp_rows, d), 4)
        if ratio >= 0.5:
            market_standards.append(d)
        k = _norm(d)
        if not base_perf_present.get(k, False) and ratio >= 0.5:
            gaps.append(
                GapItemV05(
                    feature_group="performance_parameters",
                    feature=d,
                    status="absent",
                    market_presence_ratio=ratio,
                    recommendation="Feature zur Wettbewerbsparität evaluieren.",
                )
            )
        elif base_perf_present.get(k, False) and ratio < 0.4:
            usps.append(
                UspItemV05(
                    feature_group="performance_parameters",
                    feature=d,
                    market_presence_ratio=ratio,
                    rarity_score=round(1.0 - ratio, 4),
                    rationale="Beim Zielprodukt vorhanden, in Wettbewerbern selten.",
                )
            )
            differentiators.append(d)

    # Metric features: no binary gap/USP from mention alone.
    # Only evaluate where a clear optimization direction exists and numeric values are comparable.
    metric_margin = 0.05  # 5% difference required to avoid noise-driven classifications.
    for d in metric_dims:
        ratio = round(_presence_ratio_metric(comp_rows, d), 4)
        if ratio >= 0.5:
            market_standards.append(d)

        direction = _preferred_direction_for_metric(d)
        if direction is None:
            continue

        base_cell = _find_metric_cell(baseline_row, d)
        if not base_cell or not base_cell.present:
            continue

        base_value, base_family = _to_metric_value(base_cell.value, base_cell.unit)
        if base_value is None:
            continue

        comp_values: List[float] = []
        for r in comp_rows:
            c = _find_metric_cell(r, d)
            if not c or not c.present:
                continue
            v, fam = _to_metric_value(c.value, c.unit)
            if v is None:
                continue
            # If both sides have known physical families, require same family.
            if base_family and fam and base_family != fam:
                continue
            comp_values.append(v)

        if len(comp_values) < 2:
            continue

        comp_median = _median(comp_values)
        if comp_median is None or comp_median <= 0:
            continue

        if direction == "lower":
            if base_value <= comp_median * (1.0 - metric_margin):
                usps.append(
                    UspItemV05(
                        feature_group="metric_features",
                        feature=d,
                        market_presence_ratio=ratio,
                        rarity_score=round(1.0 - ratio, 4),
                        rationale="Beim Zielprodukt messbar vorteilhafter als der Wettbewerbsmedian.",
                    )
                )
                differentiators.append(d)
            elif base_value >= comp_median * (1.0 + metric_margin):
                gaps.append(
                    GapItemV05(
                        feature_group="metric_features",
                        feature=d,
                        status="value_gap",
                        market_presence_ratio=ratio,
                        recommendation="Metrikwert im Vergleich zum Wettbewerbsniveau verbessern.",
                    )
                )

    for d in price_dims:
        ratio = round(_presence_ratio_price(comp_rows, d), 4)
        if ratio >= 0.5:
            market_standards.append(d)
        k = _norm(d)
        if not base_price_present.get(k, False) and ratio >= 0.5:
            status = "missing_data" if _norm(d) in {"preis", "uvp"} else "absent"
            gaps.append(
                GapItemV05(
                    feature_group="price_indicators",
                    feature=d,
                    status=status,
                    market_presence_ratio=ratio,
                    recommendation="Preisindikator konsistent erfassen.",
                )
            )

    for d in soft_dims:
        ratio = round(_presence_ratio_soft(comp_rows, d), 4)
        if ratio >= 0.5:
            market_standards.append(d)
        k = _norm(d)
        if not base_soft_present.get(k, False) and ratio >= 0.5:
            gaps.append(
                GapItemV05(
                    feature_group="soft_features",
                    feature=d,
                    status="absent",
                    market_presence_ratio=ratio,
                    recommendation="Feature zur Wettbewerbsparität evaluieren.",
                )
            )
        elif base_soft_present.get(k, False) and ratio < 0.4:
            usps.append(
                UspItemV05(
                    feature_group="soft_features",
                    feature=d,
                    market_presence_ratio=ratio,
                    rarity_score=round(1.0 - ratio, 4),
                    rationale="Beim Zielprodukt vorhanden, in Wettbewerbern selten.",
                )
            )
            differentiators.append(d)

    llm_payload = {
        "gaps_candidates": [g.model_dump() for g in gaps],
        "usps_candidates": [u.model_dump() for u in usps],
    }
    llm_data = _llm_refine_gaps_v05(provider=provider, payload=llm_payload, warnings=warnings)
    if llm_data:
        try:
            gap_text_map: Dict[tuple[str, str], str] = {}
            for g in llm_data.get("gaps_text") or []:
                if not isinstance(g, dict):
                    continue
                fg = str(g.get("feature_group") or "").strip()
                ft = str(g.get("feature") or "").strip()
                rc = str(g.get("recommendation") or "").strip()
                if fg and ft and rc:
                    gap_text_map[(fg, _norm(ft))] = rc
            for i, g in enumerate(gaps):
                repl = gap_text_map.get((g.feature_group, _norm(g.feature)))
                if repl:
                    gaps[i] = GapItemV05(
                        feature_group=g.feature_group,
                        feature=g.feature,
                        status=g.status,
                        market_presence_ratio=g.market_presence_ratio,
                        recommendation=repl,
                    )

            usp_text_map: Dict[tuple[str, str], str] = {}
            for u in llm_data.get("usps_text") or []:
                if not isinstance(u, dict):
                    continue
                fg = str(u.get("feature_group") or "").strip()
                ft = str(u.get("feature") or "").strip()
                ra = str(u.get("rationale") or "").strip()
                if fg and ft and ra:
                    usp_text_map[(fg, _norm(ft))] = ra
            for i, u in enumerate(usps):
                repl = usp_text_map.get((u.feature_group, _norm(u.feature)))
                if repl:
                    usps[i] = UspItemV05(
                        feature_group=u.feature_group,
                        feature=u.feature,
                        market_presence_ratio=u.market_presence_ratio,
                        rarity_score=u.rarity_score,
                        rationale=repl,
                    )
        except Exception as exc:
            warnings.append(f"LLM structured output invalid ({exc}).")

    # Prioritize and cap USPs to avoid inflated long lists.
    usp_ranked: List[tuple[float, UspItemV05]] = []
    for u in usps:
        ratio = u.market_presence_ratio
        if ratio is None:
            ratio = _presence_ratio_for_group(u.feature_group, u.feature, perf_dims, metric_dims, soft_dims, price_dims, comp_rows)
        rarity = 1.0 - ratio
        group_weight = 1.0
        if u.feature_group == "soft_features":
            group_weight = 0.9
        elif u.feature_group == "metric_features":
            group_weight = 0.95
        elif u.feature_group == "price_indicators":
            group_weight = 0.8
        usp_ranked.append((rarity * group_weight, u))
    usp_ranked.sort(key=lambda x: x[0], reverse=True)
    usps = [u for _, u in usp_ranked[:10]]
    usp_keys = {(_norm(u.feature_group), _norm(u.feature)) for u in usps}
    differentiators = [
        d
        for d in differentiators
        if (_norm("soft_features"), _norm(d)) in usp_keys
        or (_norm("performance_parameters"), _norm(d)) in usp_keys
        or (_norm("metric_features"), _norm(d)) in usp_keys
    ]

    cluster_assignment: List[ClusterAssignmentV05] = []
    priced = [r for r in comp_rows if isinstance(r.avg_price, (int, float))]
    median_price = sorted([float(r.avg_price) for r in priced])[len(priced) // 2] if priced else None
    for r in comp_rows:
        label = r.cluster or "unknown"
        if median_price is not None and isinstance(r.avg_price, (int, float)) and isinstance(r.value_score, (int, float)):
            if r.avg_price <= median_price and r.value_score >= 0.4:
                label = "value_leader"
            elif r.avg_price > median_price and r.value_score >= 0.4:
                label = "premium_performer"
            elif r.avg_price <= median_price and r.value_score < 0.4:
                label = "budget_basic"
            else:
                label = "premium_basic"
        elif isinstance(r.value_score, (int, float)):
            # Value-only fallback when price is missing.
            if r.value_score >= 0.7:
                label = "performance_focused"
            elif r.value_score >= 0.4:
                label = "mainstream"
            elif r.value_score > 0:
                label = "feature_limited"
            else:
                label = "data_gap"
        cluster_assignment.append(
            ClusterAssignmentV05(
                competitor=r.competitor,
                cluster=label,
                avg_price=r.avg_price,
                value_score=r.value_score,
            )
        )

    # Post-validation: remove invalid contradictions and dedupe.
    valid_gaps: List[GapItemV05] = []
    for g in gaps:
        k = _norm(g.feature)
        if g.feature_group == "performance_parameters" and base_perf_present.get(k, False):
            warnings.append(f"Removed contradictory gap '{g.feature}' (present in baseline performance).")
            continue
        if g.feature_group == "metric_features" and not base_metric_present.get(k, False):
            warnings.append(f"Removed invalid metric gap '{g.feature}' (missing in baseline metrics).")
            continue
        if g.feature_group == "soft_features" and base_soft_present.get(k, False):
            warnings.append(f"Removed contradictory gap '{g.feature}' (present in baseline soft features).")
            continue
        if g.feature_group == "price_indicators" and base_price_present.get(k, False):
            warnings.append(f"Removed contradictory gap '{g.feature}' (present in baseline price indicators).")
            continue
        valid_gaps.append(g)
    gaps = valid_gaps

    valid_usps: List[UspItemV05] = []
    for u in usps:
        k = _norm(u.feature)
        if u.feature_group == "performance_parameters" and not base_perf_present.get(k, False):
            warnings.append(f"Removed invalid USP '{u.feature}' (missing in baseline performance).")
            continue
        if u.feature_group == "metric_features" and not base_metric_present.get(k, False):
            warnings.append(f"Removed invalid USP '{u.feature}' (missing in baseline metrics).")
            continue
        if u.feature_group == "soft_features" and not base_soft_present.get(k, False):
            warnings.append(f"Removed invalid USP '{u.feature}' (missing in baseline soft features).")
            continue
        if u.feature_group == "price_indicators" and not base_price_present.get(k, False):
            warnings.append(f"Removed invalid USP '{u.feature}' (missing in baseline price indicators).")
            continue
        valid_usps.append(u)
    usps = valid_usps

    def _dedupe(xs: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for x in xs:
            k = x.lower().strip()
            if not k or k in seen:
                continue
            seen.add(k)
            out.append(x)
        return out

    gaps_usps = GapsAndUspsV05(
        gaps=[GapItemV05(**g.model_dump()) for g in {(_norm(x.feature_group), _norm(x.feature)): x for x in gaps}.values()],
        usps=[UspItemV05(**u.model_dump()) for u in {(_norm(x.feature_group), _norm(x.feature)): x for x in usps}.values()],
        market_standards=_dedupe(market_standards),
        differentiators=_dedupe(differentiators),
    )

    return FeatureMatrixGapAnalysisV05Result(
        comparison_matrix=matrix,
        gaps_and_usps=gaps_usps,
        cluster_assignment=cluster_assignment,
    )


# Backward-compatible alias for older imports.
def run_feature_matrx_gap_analysis_v0_5(
    *,
    product_profile: Optional[Dict[str, Any]],
    product_profile_path: Optional[str],
    competitor_profile_results: Optional[Dict[str, Any]],
    competitor_profile_results_path: Optional[str],
    provider: str = "openai",
    user_root: Path,
    work_root: Path,
) -> FeatureMatrixGapAnalysisV05Result:
    return run_feature_matrix_gap_analysis_v0_5(
        product_profile=product_profile,
        product_profile_path=product_profile_path,
        competitor_profile_results=competitor_profile_results,
        competitor_profile_results_path=competitor_profile_results_path,
        provider=provider,
        user_root=user_root,
        work_root=work_root,
    )
