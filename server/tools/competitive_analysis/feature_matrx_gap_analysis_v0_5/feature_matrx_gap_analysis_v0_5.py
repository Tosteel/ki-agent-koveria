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
    FeatureMatrxGapAnalysisV05Result,
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


def _feature_name(item: Any, *, fallback: str = "") -> str:
    if isinstance(item, dict):
        return str(item.get("name") or item.get("feature") or item.get("context") or fallback or "").strip()
    return str(item or fallback or "").strip()


def _collect_performance_dims(product: Dict[str, Any], competitors: List[Dict[str, Any]]) -> List[str]:
    dims = set()
    for x in (product.get("performance_parameters") or []):
        n = _feature_name(x)
        if n:
            dims.add(n)
    for c in competitors:
        for x in (c.get("performance_parameters") or []):
            n = _feature_name(x)
            if n:
                dims.add(n)
    return sorted(dims, key=lambda s: s.lower())


def _collect_price_dims(product: Dict[str, Any], competitors: List[Dict[str, Any]]) -> List[str]:
    dims = set()
    for x in (product.get("price_indicators") or []):
        n = _feature_name(x, fallback="Preis")
        if n:
            dims.add(n)
    for c in competitors:
        for x in (c.get("price_indicators") or []):
            n = _feature_name(x, fallback="Preis")
            if n:
                dims.add(n)
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


def _price_map(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for x in items:
        if not isinstance(x, dict):
            continue
        c = _feature_name(x, fallback="Preis")
        if not c:
            continue
        out[_norm(c)] = x
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


def _build_row(
    *,
    competitor: str,
    cluster: str,
    perf_items: List[Dict[str, Any]],
    price_items: List[Dict[str, Any]],
    soft_items: List[Dict[str, Any]],
    perf_dims: List[str],
    price_dims: List[str],
    soft_dims: List[str],
) -> CompetitorRowV05:
    pmap = _perf_map(perf_items)
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
    cov_price = (price_present / max(1, len(price_dims))) if price_dims else 0.0
    cov_soft = (soft_present / max(1, len(soft_dims))) if soft_dims else 0.0
    coverage = (cov_perf + cov_price + cov_soft) / 3.0

    avg_price = _avg_price_from_indicators(price_items)
    value_score = coverage
    if avg_price is not None and avg_price > 0:
        value_score = coverage / (1.0 + (avg_price / 10000.0))

    return CompetitorRowV05(
        competitor=competitor,
        cluster=cluster,
        performance_parameters=perf_cells,
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


def _llm_refine_gaps_v05(provider: str, payload: Dict[str, Any], warnings: List[str]) -> Dict[str, Any]:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "market_standards": {"type": "array", "items": {"type": "string"}},
            "differentiators": {"type": "array", "items": {"type": "string"}},
            "gaps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "feature_group": {"type": "string", "enum": ["performance_parameters", "price_indicators", "soft_features"]},
                        "feature": {"type": "string"},
                        "status": {"type": "string"},
                        "market_presence_ratio": {"type": "number"},
                        "recommendation": {"type": "string"},
                    },
                    "required": ["feature_group", "feature", "status", "market_presence_ratio", "recommendation"],
                },
            },
            "usps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "feature_group": {"type": "string", "enum": ["performance_parameters", "price_indicators", "soft_features"]},
                        "feature": {"type": "string"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["feature_group", "feature", "rationale"],
                },
            },
        },
        "required": ["market_standards", "differentiators", "gaps", "usps"],
    }
    p = str(provider or "openai").strip().lower()
    if p not in {"ionos", "openai", "perplexity"}:
        p = "openai"

    system = (
        "You analyze a feature matrix and derive market standards, gaps, and USPs. "
        "Respond strictly as JSON using the provided schema."
    )
    user = "Input:\n" + json.dumps(payload, ensure_ascii=False)

    if p in {"openai", "perplexity"}:
        c = LlmOpenai() if p == "openai" else LlmPerplexity()
        if not c.enabled():
            warnings.append(f"{p} not configured; heuristic gap analysis used.")
            return {}
        fmt = {"type": "json_schema", "name": "gaps_usps_v05", "schema": schema, "strict": False}
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
                "json_schema": {"name": "gaps_usps_v05", "schema": schema, "strict": True},
            },
        )
        parsed = _parse_json_strictish(c2.extract_text(completion))
        if parsed:
            return parsed
    except Exception as exc:
        warnings.append(f"IONOS gap analysis failed: {exc}")
    return {}


def run_feature_matrx_gap_analysis_v0_5(
    *,
    product_profile: Optional[Dict[str, Any]],
    product_profile_path: Optional[str],
    competitor_profile_results: Optional[Dict[str, Any]],
    competitor_profile_results_path: Optional[str],
    provider: str = "openai",
    user_root: Path,
    work_root: Path,
) -> FeatureMatrxGapAnalysisV05Result:
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

    perf_dims = _collect_performance_dims(product, competitors)
    price_dims = _collect_price_dims(product, competitors)
    soft_dims = _collect_soft_dims(product, competitors)

    baseline_name = str((product.get("metadata") or {}).get("product_name") or product.get("name") or "target_product").strip()
    baseline_row = _build_row(
        competitor=baseline_name,
        cluster="target",
        perf_items=product.get("performance_parameters") if isinstance(product.get("performance_parameters"), list) else [],
        price_items=product.get("price_indicators") if isinstance(product.get("price_indicators"), list) else [],
        soft_items=product.get("soft_features") if isinstance(product.get("soft_features"), list) else [],
        perf_dims=perf_dims,
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
            price_items=c.get("price_indicators") if isinstance(c.get("price_indicators"), list) else [],
            soft_items=c.get("soft_features") if isinstance(c.get("soft_features"), list) else [],
            perf_dims=perf_dims,
            price_dims=price_dims,
            soft_dims=soft_dims,
        )
        comp_rows.append(row)

    matrix = ComparisonMatrixV05(
        provider=str(provider or "openai").strip().lower(),
        baseline_product=baseline_name,
        performance_dimensions=perf_dims,
        price_dimensions=price_dims,
        soft_feature_dimensions=soft_dims,
        baseline_row=baseline_row,
        competitor_rows=comp_rows,
    )

    gaps: List[GapItemV05] = []
    usps: List[UspItemV05] = []
    market_standards: List[str] = []
    differentiators: List[str] = []

    llm_payload = {
        "performance_dimensions": perf_dims,
        "price_dimensions": price_dims,
        "soft_feature_dimensions": soft_dims,
        "baseline_present_performance": [x.name for x in baseline_row.performance_parameters if x.present],
        "baseline_present_soft_features": [x.name for x in baseline_row.soft_features if x.available],
        "competitor_presence_ratios": {
            "performance_parameters": {d: round(_presence_ratio_perf(comp_rows, d), 4) for d in perf_dims},
            "soft_features": {d: round(_presence_ratio_soft(comp_rows, d), 4) for d in soft_dims},
        },
        "price_value": [
            {"competitor": r.competitor, "avg_price": r.avg_price, "value_score": r.value_score}
            for r in comp_rows
        ],
    }
    llm_data = _llm_refine_gaps_v05(provider=provider, payload=llm_payload, warnings=warnings)
    if llm_data:
        try:
            market_standards = _safe_list_str(llm_data.get("market_standards"))
            differentiators = _safe_list_str(llm_data.get("differentiators"))
            if isinstance(llm_data.get("gaps"), list):
                gaps = [GapItemV05(**g) for g in llm_data.get("gaps") if isinstance(g, dict)]
            if isinstance(llm_data.get("usps"), list):
                usps = [UspItemV05(**u) for u in llm_data.get("usps") if isinstance(u, dict)]
        except Exception as exc:
            warnings.append(f"LLM structured output invalid ({exc}).")

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
        cluster_assignment.append(
            ClusterAssignmentV05(
                competitor=r.competitor,
                cluster=label,
                avg_price=r.avg_price,
                value_score=r.value_score,
            )
        )

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
        gaps=gaps,
        usps=usps,
        market_standards=_dedupe(market_standards),
        differentiators=_dedupe(differentiators),
    )

    if warnings:
        # keep linter happy + preserve current signature behavior
        _ = warnings

    return FeatureMatrxGapAnalysisV05Result(
        comparison_matrix=matrix,
        gaps_and_usps=gaps_usps,
        cluster_assignment=cluster_assignment,
    )
