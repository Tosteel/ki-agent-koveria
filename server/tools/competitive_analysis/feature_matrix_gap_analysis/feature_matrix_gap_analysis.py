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
    ClusterAssignment,
    ComparisonMatrix,
    CompetitorRow,
    FeatureCell,
    FeatureMatrixGapAnalysisResult,
    GapItem,
    GapsAndUsps,
    UspItem,
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
    root_key: str,
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

    if root_key in payload and isinstance(payload.get(root_key), dict):
        payload = payload[root_key]
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail=f"Invalid payload for {root_key}")
    return payload


def _norm_feature_name(name: str) -> str:
    return " ".join(str(name or "").strip().lower().replace("_", " ").split())


def _feature_dict_from_list(features: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for f in features:
        if not isinstance(f, dict):
            continue
        name = str(f.get("schema_feature") or f.get("name") or "").strip()
        if not name:
            continue
        key = _norm_feature_name(name)
        out[key] = f
    return out


def _feature_value_str(f: Dict[str, Any]) -> str:
    value = f.get("value")
    if value in (None, ""):
        value = f.get("normalized_value")
    unit = str(f.get("normalized_unit") or f.get("unit") or "").strip()
    if value in (None, ""):
        return ""
    return f"{value} {unit}".strip()


def _feature_numeric(f: Dict[str, Any]) -> float | None:
    v = f.get("normalized_value")
    if isinstance(v, (int, float)):
        return float(v)
    v = f.get("value")
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _avg_price(profile: Dict[str, Any]) -> float | None:
    prices = profile.get("prices") if isinstance(profile.get("prices"), list) else []
    vals = []
    for p in prices:
        if not isinstance(p, dict):
            continue
        v = p.get("value")
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return round(mean(vals), 4) if vals else None


def _build_dimensions(product: Dict[str, Any], competitors: List[Dict[str, Any]]) -> List[str]:
    dims = set()
    for f in (product.get("normalized_features") or []):
        if isinstance(f, dict):
            n = str(f.get("name") or f.get("schema_feature") or "").strip()
            if n:
                dims.add(n)
    for c in competitors:
        for f in (c.get("mapped_features") or []):
            if isinstance(f, dict):
                n = str(f.get("schema_feature") or f.get("name") or "").strip()
                if n and n.lower() != "other":
                    dims.add(n)
    return sorted(dims, key=lambda s: s.lower())


def _row_from_feature_map(*, competitor: str, cluster: str, fmap: Dict[str, Dict[str, Any]], dims: List[str], avg_price: float | None) -> CompetitorRow:
    cells: List[FeatureCell] = []
    present_count = 0
    for d in dims:
        key = _norm_feature_name(d)
        f = fmap.get(key)
        if f is None:
            cells.append(FeatureCell(feature=d, present=False))
            continue
        present_count += 1
        cells.append(
            FeatureCell(
                feature=d,
                value=_feature_value_str(f),
                normalized_value=f.get("normalized_value"),
                unit=str(f.get("normalized_unit") or f.get("unit") or ""),
                present=True,
            )
        )

    feature_coverage = present_count / max(1, len(dims))
    value_score = feature_coverage
    if avg_price is not None and avg_price > 0:
        value_score = feature_coverage / (1.0 + (avg_price / 10000.0))
    return CompetitorRow(
        competitor=competitor,
        cluster=cluster,
        features=cells,
        avg_price=avg_price,
        value_score=round(value_score, 4),
    )


def _presence_ratio(rows: List[CompetitorRow], feature: str) -> float:
    if not rows:
        return 0.0
    key = _norm_feature_name(feature)
    present = 0
    for r in rows:
        m = next((c for c in r.features if _norm_feature_name(c.feature) == key), None)
        if m and m.present:
            present += 1
    return present / len(rows)


def _llm_refine_gaps(provider: str, payload: Dict[str, Any], warnings: List[str]) -> Dict[str, Any]:
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
                        "feature": {"type": "string"},
                        "status": {"type": "string"},
                        "market_presence_ratio": {"type": "number"},
                        "recommendation": {"type": "string"},
                    },
                    "required": ["feature", "status", "market_presence_ratio", "recommendation"],
                },
            },
            "usps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "feature": {"type": "string"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["feature", "rationale"],
                },
            },
        },
        "required": ["market_standards", "differentiators", "gaps", "usps"],
    }
    p = str(provider or "ionos").strip().lower()
    if p not in {"ionos", "openai", "perplexity"}:
        p = "ionos"

    system = (
        "Du analysierst Feature-Matrix-Daten und leitest Marktstandards, Gaps und USPs ab. "
        "Antworte strikt als JSON gemäß Schema."
    )
    user = "Input:\n" + json.dumps(payload, ensure_ascii=False)

    if p in {"openai", "perplexity"}:
        c = LlmOpenai() if p == "openai" else LlmPerplexity()
        if not c.enabled():
            warnings.append(f"{p} not configured; heuristic gap analysis used.")
            return {}
        fmt = {"type": "json_schema", "name": "gaps_usps", "schema": schema, "strict": False}
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
                "json_schema": {"name": "gaps_usps", "schema": schema, "strict": True},
            },
        )
        parsed = _parse_json_strictish(c2.extract_text(completion))
        if parsed:
            return parsed
    except Exception as exc:
        warnings.append(f"IONOS gap analysis failed: {exc}")
    return {}


def run_feature_matrix_gap_analysis(
    *,
    product_profile: Optional[Dict[str, Any]],
    product_profile_path: Optional[str],
    competitor_profiles: Optional[Dict[str, Any]],
    competitor_profiles_path: Optional[str],
    provider: str = "ionos",
    user_root: Path,
    work_root: Path,
) -> FeatureMatrixGapAnalysisResult:
    product = _load_json_obj(
        inline_obj=product_profile,
        path=product_profile_path,
        root_key="product_profile",
        user_root=user_root,
        work_root=work_root,
    )
    comp_payload = _load_json_obj(
        inline_obj=competitor_profiles,
        path=competitor_profiles_path,
        root_key="competitor_profiles",
        user_root=user_root,
        work_root=work_root,
    )

    warnings = _safe_list_str(product.get("extraction_warnings")) + _safe_list_str(comp_payload.get("extraction_warnings"))

    competitors = comp_payload.get("competitor_profiles") if isinstance(comp_payload.get("competitor_profiles"), list) else []
    dims = _build_dimensions(product, [c for c in competitors if isinstance(c, dict)])

    product_fmap = _feature_dict_from_list(product.get("normalized_features") or [])
    baseline_name = str((product.get("metadata") or {}).get("product_name") or "target_product").strip()
    baseline_row = _row_from_feature_map(
        competitor=baseline_name,
        cluster="target",
        fmap=product_fmap,
        dims=dims,
        avg_price=_avg_price(product),
    )

    comp_rows: List[CompetitorRow] = []
    for c in competitors:
        if not isinstance(c, dict):
            continue
        name = str(c.get("name") or "unknown_competitor").strip()
        cluster = str(c.get("cluster") or "unknown").strip()
        fmap = _feature_dict_from_list(c.get("mapped_features") or [])
        row = _row_from_feature_map(
            competitor=name,
            cluster=cluster,
            fmap=fmap,
            dims=dims,
            avg_price=_avg_price(c),
        )
        comp_rows.append(row)

    matrix = ComparisonMatrix(
        provider=str(provider or "ionos").strip().lower() if str(provider or "").strip().lower() in {"ionos", "openai"} else "ionos",
        feature_dimensions=dims,
        baseline_product=baseline_name,
        baseline_row=baseline_row,
        competitor_rows=comp_rows,
    )

    # Heuristic gap + usp core
    gaps: List[GapItem] = []
    usps: List[UspItem] = []
    market_standards: List[str] = []
    differentiators: List[str] = []

    base_present = {_norm_feature_name(c.feature): c.present for c in baseline_row.features}
    for d in dims:
        ratio = _presence_ratio(comp_rows, d)
        key = _norm_feature_name(d)
        has_base = bool(base_present.get(key))

        if ratio >= 0.6:
            market_standards.append(d)
        if has_base and ratio < 0.4:
            differentiators.append(d)
            usps.append(UspItem(feature=d, rationale="Feature beim Zielprodukt vorhanden, bei Wettbewerbern selten."))
        if (not has_base) and ratio >= 0.5:
            gaps.append(
                GapItem(
                    feature=d,
                    status="missing_vs_market",
                    market_presence_ratio=round(ratio, 4),
                    recommendation="Feature als Priorität für Wettbewerbsparität prüfen.",
                )
            )

    llm_payload = {
        "feature_dimensions": dims,
        "baseline_present_features": [c.feature for c in baseline_row.features if c.present],
        "competitor_presence_ratios": {d: round(_presence_ratio(comp_rows, d), 4) for d in dims},
        "price_value": [
            {"competitor": r.competitor, "avg_price": r.avg_price, "value_score": r.value_score}
            for r in comp_rows
        ],
    }
    llm_data = _llm_refine_gaps(provider=provider, payload=llm_payload, warnings=warnings)
    if llm_data:
        try:
            market_standards = _safe_list_str(llm_data.get("market_standards")) or market_standards
            differentiators = _safe_list_str(llm_data.get("differentiators")) or differentiators
            if isinstance(llm_data.get("gaps"), list):
                gaps = [GapItem(**g) for g in llm_data.get("gaps") if isinstance(g, dict)]
            if isinstance(llm_data.get("usps"), list):
                usps = [UspItem(**u) for u in llm_data.get("usps") if isinstance(u, dict)]
        except Exception as exc:
            warnings.append(f"LLM structured output invalid, heuristic retained ({exc}).")

    # Cluster assignment (price/performance buckets)
    cluster_assignment: List[ClusterAssignment] = []
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
            ClusterAssignment(
                competitor=r.competitor,
                cluster=label,
                avg_price=r.avg_price,
                value_score=r.value_score,
            )
        )

    # dedupe lists
    def _dedupe(xs: List[str]) -> List[str]:
        out = []
        seen = set()
        for x in xs:
            k = x.lower().strip()
            if not k or k in seen:
                continue
            seen.add(k)
            out.append(x)
        return out

    gaps_usps = GapsAndUsps(
        gaps=gaps,
        usps=usps,
        market_standards=_dedupe(market_standards),
        differentiators=_dedupe(differentiators),
    )

    matrix.provider = matrix.provider
    return FeatureMatrixGapAnalysisResult(
        comparison_matrix=matrix,
        gaps_and_usps=gaps_usps,
        cluster_assignment=cluster_assignment,
    )
