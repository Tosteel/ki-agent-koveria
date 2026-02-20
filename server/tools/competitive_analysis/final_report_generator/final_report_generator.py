from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from server.services.llm_ionos import IonosLLM
from server.services.llm_openai import LlmOpenai
from server.services.llm_perplexity import LlmPerplexity

from .models import (
    ArtifactChunk,
    FinalReport,
    FinalReportResult,
    PositioningDiagram,
    PositioningPoint,
    RecommendationItem,
    SwotSummary,
    ValidationReport,
)


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

    for c in candidates:
        if c.exists() and c.is_file() and (user_root in c.parents or c == user_root):
            return c

    raise HTTPException(status_code=404, detail=f"File not found: {path}")


def _load_artifacts(
    *,
    artifacts: Optional[Dict[str, Any]],
    artifact_paths: Optional[Dict[str, str]],
    user_root: Path,
    work_root: Path,
    warnings: List[str],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(artifacts, dict):
        out.update(artifacts)

    if isinstance(artifact_paths, dict):
        for key, rel in artifact_paths.items():
            try:
                fp = _resolve_input_path(str(rel or ""), user_root=user_root, work_root=work_root)
                payload = json.loads(fp.read_text(encoding="utf-8"))
                k = str(key)
                if isinstance(payload, dict) and k in payload and isinstance(payload.get(k), dict):
                    payload = payload[k]
                out[k] = payload
            except Exception as exc:
                warnings.append(f"Could not load artifact '{key}' from '{rel}': {exc}")

    # common wrappers
    if isinstance(out.get("gaps_and_usps"), dict):
        g = out["gaps_and_usps"]
        if "comparison_matrix" in g and "comparison_matrix" not in out:
            out["comparison_matrix"] = g.get("comparison_matrix")
        if "cluster_assignment" in g and "cluster_assignment" not in out:
            out["cluster_assignment"] = g.get("cluster_assignment")

    if isinstance(out.get("strategic_analysis"), dict):
        s = out["strategic_analysis"]
        if "swot" in s and "swot" not in out:
            out["swot"] = s.get("swot")
        if "positioning_data" in s and "positioning_data" not in out:
            out["positioning_data"] = s.get("positioning_data")

    return out


def _artifact_chunk(name: str, payload: Any) -> ArtifactChunk:
    txt = json.dumps(payload, ensure_ascii=False)
    return ArtifactChunk(
        artifact=name,
        key_points=[f"artifact={name}", f"size={len(txt)} chars"],
        evidence_refs=[f"{name}:root"],
    )


def _build_product_profile_brief(artifacts: Dict[str, Any]) -> Dict[str, Any]:
    p = artifacts.get("product_profile") if isinstance(artifacts.get("product_profile"), dict) else {}
    md = p.get("metadata") if isinstance(p.get("metadata"), dict) else {}

    claims = [c for c in (p.get("claims") or []) if isinstance(c, dict)]
    top_claims = [str(c.get("text") or "").strip() for c in claims if str(c.get("text") or "").strip()][:6]

    out = {
        "product_name": str(md.get("product_name") or "").strip(),
        "manufacturer": str(md.get("manufacturer") or "").strip(),
        "category": str(p.get("product_category") or "unknown").strip(),
        "target_segments": _safe_list_str(p.get("target_segments"))[:8],
        "use_cases": _safe_list_str(p.get("use_cases"))[:8],
        "key_differentiators": _safe_list_str(p.get("differentiators"))[:8],
        "top_claims": top_claims,
        "feature_count": len(p.get("normalized_features") or []),
    }
    return out


def _build_competitor_overview(artifacts: Dict[str, Any]) -> Dict[str, Any]:
    cl = artifacts.get("competitor_list") if isinstance(artifacts.get("competitor_list"), dict) else {}
    cp = artifacts.get("competitor_profiles") if isinstance(artifacts.get("competitor_profiles"), dict) else {}

    list_items = [x for x in (cl.get("competitors") or []) if isinstance(x, dict)]
    prof_items = [x for x in (cp.get("competitor_profiles") or []) if isinstance(x, dict)]

    by_name = {str(p.get("name") or "").strip().lower(): p for p in prof_items if str(p.get("name") or "").strip()}

    rows: List[Dict[str, Any]] = []
    for c in list_items[:40]:
        name = str(c.get("name") or "").strip()
        p = by_name.get(name.lower())
        quality = (p.get("data_quality") if isinstance(p, dict) else {}) if isinstance(p, dict) else {}
        rows.append(
            {
                "name": name,
                "url": str(c.get("url") or "").strip(),
                "cluster": str(c.get("cluster") or "unknown").strip(),
                "relevance_score": c.get("relevance_score"),
                "similarity_score": c.get("similarity_score"),
                "short_profile": str(c.get("snippet") or "").strip(),
                "data_confidence": quality.get("confidence") if isinstance(quality, dict) else None,
                "mapped_features": len((p.get("mapped_features") or [])) if isinstance(p, dict) else 0,
            }
        )

    return {
        "count": len(rows),
        "table": rows,
        "pagination": {"page_size": 20, "pages": max(1, (len(rows) + 19) // 20)},
    }


def _build_feature_matrix_section(artifacts: Dict[str, Any]) -> Dict[str, Any]:
    m = artifacts.get("comparison_matrix") if isinstance(artifacts.get("comparison_matrix"), dict) else {}
    dims = _safe_list_str(m.get("feature_dimensions"))
    base = m.get("baseline_row") if isinstance(m.get("baseline_row"), dict) else {}
    comp_rows = [r for r in (m.get("competitor_rows") or []) if isinstance(r, dict)]

    def _compact_row(r: Dict[str, Any]) -> Dict[str, Any]:
        feats = [f for f in (r.get("features") or []) if isinstance(f, dict)]
        present = [str(f.get("feature") or "") for f in feats if bool(f.get("present")) and str(f.get("feature") or "")][:40]
        return {
            "competitor": str(r.get("competitor") or ""),
            "cluster": str(r.get("cluster") or "unknown"),
            "avg_price": r.get("avg_price"),
            "value_score": r.get("value_score"),
            "present_features": present,
        }

    rows = [_compact_row(base)] if base else []
    rows.extend(_compact_row(r) for r in comp_rows)

    return {
        "dimensions": dims,
        "rows": rows,
        "pagination": {"page_size": 15, "pages": max(1, (len(rows) + 14) // 15)},
    }


def _build_gap_usp_analysis(artifacts: Dict[str, Any]) -> Dict[str, Any]:
    g = artifacts.get("gaps_and_usps") if isinstance(artifacts.get("gaps_and_usps"), dict) else {}
    gaps = [x for x in (g.get("gaps") or []) if isinstance(x, dict)]
    usps = [x for x in (g.get("usps") or []) if isinstance(x, dict)]

    # prioritize by presence ratio desc
    gaps_sorted = sorted(gaps, key=lambda x: float(x.get("market_presence_ratio") or 0.0), reverse=True)

    return {
        "prioritized_gaps": [
            {
                "feature": str(x.get("feature") or ""),
                "market_presence_ratio": x.get("market_presence_ratio"),
                "recommendation": str(x.get("recommendation") or ""),
            }
            for x in gaps_sorted[:12]
        ],
        "prioritized_usps": [
            {
                "feature": str(x.get("feature") or ""),
                "rationale": str(x.get("rationale") or ""),
            }
            for x in usps[:12]
        ],
        "market_standards": _safe_list_str(g.get("market_standards"))[:20],
        "differentiators": _safe_list_str(g.get("differentiators"))[:20],
    }


def _build_swot(artifacts: Dict[str, Any], gap_usp: Dict[str, Any]) -> SwotSummary:
    s = artifacts.get("swot") if isinstance(artifacts.get("swot"), dict) else {}
    if not s and isinstance(artifacts.get("strategic_analysis"), dict):
        sa = artifacts.get("strategic_analysis")
        if isinstance(sa.get("swot"), dict):
            s = sa.get("swot")

    def _extract(vals: Any) -> List[str]:
        out: List[str] = []
        if isinstance(vals, list):
            for v in vals:
                if isinstance(v, dict):
                    st = str(v.get("statement") or "").strip()
                    if st:
                        out.append(st)
                else:
                    st = str(v or "").strip()
                    if st:
                        out.append(st)
        return out

    strengths = _extract(s.get("strengths"))
    weaknesses = _extract(s.get("weaknesses"))
    opportunities = _extract(s.get("opportunities"))
    threats = _extract(s.get("threats"))

    # fallback enrichments
    if not strengths:
        strengths = [f"USP: {x.get('feature')}" for x in gap_usp.get("prioritized_usps", []) if str(x.get("feature") or "")][:6]
    if not weaknesses:
        weaknesses = [f"Gap: {x.get('feature')}" for x in gap_usp.get("prioritized_gaps", []) if str(x.get("feature") or "")][:6]
    if not opportunities:
        opportunities = [f"Marktchance über {x}" for x in gap_usp.get("market_standards", [])[:6]]
    if not threats:
        threats = ["Technologische Substitution", "Steigender Wettbewerbsdruck"]

    return SwotSummary(
        strengths=strengths[:10],
        weaknesses=weaknesses[:10],
        opportunities=opportunities[:10],
        threats=threats[:10],
    )


def _build_positioning(artifacts: Dict[str, Any]) -> PositioningDiagram:
    pos = artifacts.get("positioning_data") if isinstance(artifacts.get("positioning_data"), dict) else {}
    if not pos and isinstance(artifacts.get("strategic_analysis"), dict):
        sa = artifacts.get("strategic_analysis")
        if isinstance(sa.get("positioning_data"), dict):
            pos = sa.get("positioning_data")

    axes_x = str(pos.get("primary_axis_x") or "Preis")
    axes_y = str(pos.get("primary_axis_y") or "Leistung")

    points: List[PositioningPoint] = []

    def _to_float(v: Any, default: float = 0.0) -> float:
        try:
            return float(v)
        except Exception:
            return default

    def _is_valid_cluster_point(row: Dict[str, Any]) -> bool:
        if not isinstance(row, dict):
            return False
        name = str(row.get("competitor") or row.get("name") or "").strip()
        if not name:
            return False
        return True
    # from explicit clusters
    cc = pos.get("competitor_clusters") if isinstance(pos.get("competitor_clusters"), list) else []
    for c in cc:
        if not _is_valid_cluster_point(c):
            continue
        points.append(
            PositioningPoint(
                name=str(c.get("competitor") or c.get("name") or ""),
                x=_to_float(c.get("avg_price"), 0.0),
                y=_to_float(c.get("value_score"), 0.0),
                point_type="competitor",
            )
        )

    # fallback from comparison matrix
    if not points and isinstance(artifacts.get("comparison_matrix"), dict):
        m = artifacts.get("comparison_matrix")
        br = m.get("baseline_row") if isinstance(m.get("baseline_row"), dict) else None
        if br:
            points.append(
                PositioningPoint(
                    name=str(m.get("baseline_product") or "Own Product"),
                    x=float(br.get("avg_price") or 0.0),
                    y=float(br.get("value_score") or 0.0),
                    point_type="own",
                )
            )
        for r in (m.get("competitor_rows") or []):
            if not isinstance(r, dict):
                continue
            points.append(
                PositioningPoint(
                    name=str(r.get("competitor") or ""),
                    x=float(r.get("avg_price") or 0.0),
                    y=float(r.get("value_score") or 0.0),
                    point_type="competitor",
                )
            )

    interp = _safe_list_str(pos.get("interpretation"))
    if not interp:
        interp = [str(pos.get("position_label") or "Position im Wettbewerbsraum aus Preis-/Leistungsdaten abgeleitet.")]

    return PositioningDiagram(
        axis_x=axes_x,
        axis_y=axes_y,
        points=points[:40],
        interpretation=interp[:8],
    )


def _build_report_context(
    *,
    product_brief: Dict[str, Any],
    competitor_overview: Dict[str, Any],
    feature_matrix: Dict[str, Any],
    gap_usp: Dict[str, Any],
    swot: SwotSummary,
    positioning: PositioningDiagram,
) -> Dict[str, Any]:
    must = []
    must.extend(swot.strengths[:3])
    must.extend(swot.weaknesses[:3])
    must.extend(swot.opportunities[:3])
    must.extend(swot.threats[:3])

    return {
        "product_profile_brief": product_brief,
        "competitor_overview": competitor_overview,
        "feature_matrix_section": feature_matrix,
        "gap_usp_analysis": gap_usp,
        "swot": swot.model_dump(),
        "positioning": {
            "axes": {"x": positioning.axis_x, "y": positioning.axis_y},
            "points": [p.model_dump() for p in positioning.points],
            "interpretation": positioning.interpretation,
        },
        "must_include_claims": [m for m in must if m][:20],
    }


def _llm_call_structured(provider: str, schema_name: str, schema: Dict[str, Any], system: str, user: str, warnings: List[str]) -> Dict[str, Any]:
    p = str(provider or "ionos").strip().lower()
    if p not in {"ionos", "openai", "perplexity"}:
        p = "ionos"

    if p in {"openai", "perplexity"}:
        c = LlmOpenai() if p == "openai" else LlmPerplexity()
        if not c.enabled():
            warnings.append(f"{p} not configured.")
            return {}
        try:
            resp = c._call(
                input_messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                text_format={"type": "json_schema", "name": schema_name, "schema": schema, "strict": False},
            )
            text = ""
            for item in resp.get("output", []):
                for cc in item.get("content", []):
                    if cc.get("type") == "output_text":
                        text += str(cc.get("text") or "")
            return _parse_json_strictish(text)
        except Exception as exc:
            warnings.append(f"{p} call failed ({schema_name}): {exc}")
            return {}

    c2 = IonosLLM()
    if not c2.enabled():
        warnings.append("IONOS not configured.")
        return {}
    try:
        comp = c2.chat_completions(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema, "strict": True},
            },
        )
        return _parse_json_strictish(c2.extract_text(comp))
    except Exception as exc:
        warnings.append(f"IONOS call failed ({schema_name}): {exc}")
        return {}


def _generate_summary_and_recommendations(provider: str, context: Dict[str, Any], warnings: List[str]) -> Dict[str, Any]:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "executive_summary": {"type": "array", "items": {"type": "string"}},
            "strategic_recommendations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "title": {"type": "string"},
                        "action": {"type": "string"},
                        "priority": {"type": "string"},
                        "horizon": {"type": "string"},
                        "evidence_refs": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["title", "action", "priority", "horizon", "evidence_refs"],
                },
            },
        },
        "required": ["executive_summary", "strategic_recommendations"],
    }

    parsed = _llm_call_structured(
        provider=provider,
        schema_name="final_report_summary_recommendations",
        schema=schema,
        system=(
            "Du erstellst die finalen Kapitel 'Executive Summary' und 'Strategische Empfehlungen'. "
            "Nutze ausschließlich den Kontext. Priorisiere 3-5 Empfehlungen mit klaren Aktionen und Evidenz-Referenzen. "
            "Schreibe alle Texte auf Deutsch."
        ),
        user="Input:\n" + json.dumps(context, ensure_ascii=False),
        warnings=warnings,
    )
    return parsed


def _heuristic_summary_and_recommendations(swot: SwotSummary) -> Dict[str, Any]:
    exec_summary = []
    if swot.strengths:
        exec_summary.append(f"Kernstärke: {swot.strengths[0]}")
    if swot.weaknesses:
        exec_summary.append(f"Hauptlücke: {swot.weaknesses[0]}")
    if swot.opportunities:
        exec_summary.append(f"Wichtigste Chance: {swot.opportunities[0]}")
    if swot.threats:
        exec_summary.append(f"Wesentliches Risiko: {swot.threats[0]}")

    recs = []
    for w in swot.weaknesses[:2]:
        recs.append(
            {
                "title": "Gap schließen",
                "action": w,
                "priority": "high",
                "horizon": "kurzfristig",
                "evidence_refs": ["swot.weaknesses"],
            }
        )
    for s in swot.strengths[:2]:
        recs.append(
            {
                "title": "USP skalieren",
                "action": s,
                "priority": "high",
                "horizon": "kurzfristig",
                "evidence_refs": ["swot.strengths"],
            }
        )
    return {"executive_summary": exec_summary[:6], "strategic_recommendations": recs[:5]}


def _localize_report_german(provider: str, report: FinalReport, warnings: List[str]) -> FinalReport:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "product_profile_brief": {"type": "object", "additionalProperties": True},
            "competitor_overview": {"type": "object", "additionalProperties": True},
            "feature_matrix_section": {"type": "object", "additionalProperties": True},
            "gap_usp_analysis": {"type": "object", "additionalProperties": True},
            "swot": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "strengths": {"type": "array", "items": {"type": "string"}},
                    "weaknesses": {"type": "array", "items": {"type": "string"}},
                    "opportunities": {"type": "array", "items": {"type": "string"}},
                    "threats": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["strengths", "weaknesses", "opportunities", "threats"],
            },
            "positioning_diagram": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "axis_x": {"type": "string"},
                    "axis_y": {"type": "string"},
                    "points": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "name": {"type": "string"},
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                                "point_type": {"type": "string"},
                            },
                            "required": ["name", "x", "y", "point_type"],
                        },
                    },
                    "interpretation": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["axis_x", "axis_y", "points", "interpretation"],
            },
            "strategic_recommendations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "title": {"type": "string"},
                        "action": {"type": "string"},
                        "priority": {"type": "string"},
                        "horizon": {"type": "string"},
                        "evidence_refs": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["title", "action", "priority", "horizon", "evidence_refs"],
                },
            },
            "executive_summary": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "product_profile_brief",
            "competitor_overview",
            "feature_matrix_section",
            "gap_usp_analysis",
            "swot",
            "positioning_diagram",
            "strategic_recommendations",
            "executive_summary",
        ],
    }

    payload = report.model_dump()
    localized = _llm_call_structured(
        provider=provider,
        schema_name="final_report_german_localization",
        schema=schema,
        system=(
            "Du lokalisierst einen Wettbewerbsanalyse-Report vollständig auf Deutsch. "
            "Behalte die JSON-Struktur exakt bei. Übersetze ausschließlich Werte/Texte in natürliches Deutsch. "
            "Marken-, Produktnamen, URLs, Zahlen, Einheiten und Feldnamen unverändert lassen."
        ),
        user="Input:\n" + json.dumps(payload, ensure_ascii=False),
        warnings=warnings,
    )

    if not localized:
        return report
    try:
        return FinalReport(**localized)
    except Exception as exc:
        warnings.append(f"German localization failed; kept original report ({exc})")
        return report


def _contains_claim(report: FinalReport, claim: str) -> bool:
    c = claim.lower().strip()
    if not c:
        return True
    hay = []
    hay.extend(report.executive_summary)
    hay.extend(report.swot.strengths)
    hay.extend(report.swot.weaknesses)
    hay.extend(report.swot.opportunities)
    hay.extend(report.swot.threats)
    hay.extend(report.positioning_diagram.interpretation)
    hay.extend([r.action for r in report.strategic_recommendations])
    return c in "\n".join(h.lower() for h in hay)


def _validate_report(report: FinalReport, context: Dict[str, Any]) -> ValidationReport:
    missing: List[str] = []
    if not report.product_profile_brief:
        missing.append("product_profile_brief")
    if not report.competitor_overview:
        missing.append("competitor_overview")
    if not report.feature_matrix_section:
        missing.append("feature_matrix_section")
    if not report.gap_usp_analysis:
        missing.append("gap_usp_analysis")

    if not report.swot.strengths:
        missing.append("swot.strengths")
    if not report.swot.weaknesses:
        missing.append("swot.weaknesses")
    if not report.swot.opportunities:
        missing.append("swot.opportunities")
    if not report.swot.threats:
        missing.append("swot.threats")

    if not report.positioning_diagram.axis_x or not report.positioning_diagram.axis_y:
        missing.append("positioning.axes")
    if not report.positioning_diagram.points:
        missing.append("positioning.points")

    must = _safe_list_str(context.get("must_include_claims"))
    included = sum(1 for m in must if _contains_claim(report, m))
    coverage = included / max(1, len(must)) if must else 1.0
    if must and coverage < 0.9:
        missing.append("must_include_claims_coverage")

    return ValidationReport(coverage_score=round(coverage, 4), missing_items=missing, repaired=False)


def build_final_report(
    *,
    artifacts: Optional[Dict[str, Any]],
    artifact_paths: Optional[Dict[str, str]],
    provider: str = "ionos",
    max_chars_per_artifact: int = 10000,
    user_root: Path,
    work_root: Path,
) -> FinalReportResult:
    warnings: List[str] = []
    all_artifacts = _load_artifacts(
        artifacts=artifacts,
        artifact_paths=artifact_paths,
        user_root=user_root,
        work_root=work_root,
        warnings=warnings,
    )
    if not all_artifacts:
        raise HTTPException(status_code=400, detail="No artifacts available for final report generation.")

    artifact_chunks = [_artifact_chunk(k, v) for k, v in all_artifacts.items()]

    product_brief = _build_product_profile_brief(all_artifacts)
    competitor_overview = _build_competitor_overview(all_artifacts)
    feature_matrix = _build_feature_matrix_section(all_artifacts)
    gap_usp = _build_gap_usp_analysis(all_artifacts)
    swot = _build_swot(all_artifacts, gap_usp)
    positioning = _build_positioning(all_artifacts)

    context = _build_report_context(
        product_brief=product_brief,
        competitor_overview=competitor_overview,
        feature_matrix=feature_matrix,
        gap_usp=gap_usp,
        swot=swot,
        positioning=positioning,
    )

    llm_sr = _generate_summary_and_recommendations(provider=provider, context=context, warnings=warnings)
    if not llm_sr:
        llm_sr = _heuristic_summary_and_recommendations(swot)

    exec_summary = _safe_list_str(llm_sr.get("executive_summary"))[:8]
    recs = []
    for r in (llm_sr.get("strategic_recommendations") or []):
        if isinstance(r, dict):
            try:
                recs.append(RecommendationItem(**r))
            except Exception:
                continue
    if not recs:
        recs = [RecommendationItem(**x) for x in _heuristic_summary_and_recommendations(swot).get("strategic_recommendations", [])]

    report = FinalReport(
        product_profile_brief=product_brief,
        competitor_overview=competitor_overview,
        feature_matrix_section=feature_matrix,
        gap_usp_analysis=gap_usp,
        swot=swot,
        positioning_diagram=positioning,
        strategic_recommendations=recs[:8],
        executive_summary=exec_summary,
    )

    validation = _validate_report(report, context)
    report = _localize_report_german(provider=provider, report=report, warnings=warnings)

    p = str(provider or "ionos").strip().lower()
    if p not in {"ionos", "openai", "perplexity"}:
        p = "ionos"

    return FinalReportResult(
        provider=p,
        report_context=context,
        artifact_chunks=artifact_chunks,
        final_report=report,
        validation=validation,
        extraction_warnings=warnings,
    )
