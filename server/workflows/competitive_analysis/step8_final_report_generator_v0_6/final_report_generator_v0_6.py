from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from server.services.llm_ionos import IonosLLM
from server.services.llm_openai import LlmOpenai
from server.services.llm_perplexity import LlmPerplexity
from server.workflows.competitive_analysis.step8_final_report_generator_v0_5.final_report_generator_v0_5 import (
    build_final_report_v0_5,
)

from .models import ArtifactChunk, FinalReport, FinalReportResult, ValidationReport


def _norm_name(v: Any) -> str:
    return str(v or "").strip().lower()


def _to_float(v: Any) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None


def _safe_list_str(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
    for v in values:
        s = str(v or "").strip()
        if s:
            out.append(s)
    return out


def _row_rank(row: Dict[str, Any]) -> Tuple[float, float, float]:
    mapped = _to_float(row.get("mapped_features")) or 0.0
    rel = _to_float(row.get("relevance_score")) or 0.0
    sim = _to_float(row.get("similarity_score")) or 0.0
    return mapped, rel, sim


def _derive_cluster(avg_price: Optional[float], value_score: Optional[float]) -> str:
    if avg_price is None and value_score is None:
        return "data_gap"
    v = value_score if value_score is not None else 0.0
    p = avg_price if avg_price is not None else 0.0
    if v >= 0.85 and p >= 1100:
        return "premium_leader"
    if v >= 0.85:
        return "value_leader"
    if v >= 0.65:
        return "mainstream"
    return "niche_challenger"


def _compact_feature_list(vals: List[str], max_items: int = 3) -> str:
    clean = []
    seen = set()
    for v in vals:
        s = str(v or "").strip()
        k = s.lower()
        if not s or k in seen:
            continue
        seen.add(k)
        clean.append(s)
        if len(clean) >= max_items:
            break
    return ", ".join(clean)


def _enhance_sections(report: Dict[str, Any]) -> Dict[str, Any]:
    comp = report.get("competitor_overview") if isinstance(report.get("competitor_overview"), dict) else {}
    matrix = report.get("feature_matrix_section") if isinstance(report.get("feature_matrix_section"), dict) else {}

    table = [r for r in (comp.get("table") or []) if isinstance(r, dict)]
    rows = [r for r in (matrix.get("rows") or []) if isinstance(r, dict)]

    # Deduplicate feature matrix rows by competitor name (keep best value_score, then avg_price presence)
    best_matrix: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        name = _norm_name(r.get("competitor"))
        if not name:
            continue
        prev = best_matrix.get(name)
        if prev is None:
            best_matrix[name] = r
            continue
        prev_vs = _to_float(prev.get("value_score")) or -1.0
        cur_vs = _to_float(r.get("value_score")) or -1.0
        if cur_vs > prev_vs:
            best_matrix[name] = r
            continue
        if cur_vs == prev_vs:
            prev_has_price = _to_float(prev.get("avg_price")) is not None
            cur_has_price = _to_float(r.get("avg_price")) is not None
            if cur_has_price and not prev_has_price:
                best_matrix[name] = r

    dedup_matrix_rows = list(best_matrix.values())

    # Resolve unknown clusters on matrix rows
    cluster_map: Dict[str, str] = {}
    for r in dedup_matrix_rows:
        name = _norm_name(r.get("competitor"))
        if not name:
            continue
        raw_cluster = str(r.get("cluster") or "").strip()
        avg_price = _to_float(r.get("avg_price"))
        value_score = _to_float(r.get("value_score"))
        if not raw_cluster or raw_cluster.lower() == "unknown":
            raw_cluster = _derive_cluster(avg_price, value_score)
            r["cluster"] = raw_cluster
        cluster_map[name] = raw_cluster

    # Deduplicate competitor overview + cluster resolve + short profile from significant features
    best_comp: Dict[str, Dict[str, Any]] = {}
    for r in table:
        name = _norm_name(r.get("name"))
        if not name:
            continue
        prev = best_comp.get(name)
        if prev is None or _row_rank(r) > _row_rank(prev):
            best_comp[name] = r

    dedup_comp_rows: List[Dict[str, Any]] = []
    for _, r in best_comp.items():
        name = _norm_name(r.get("name"))
        m = best_matrix.get(name, {})
        avg_price = _to_float(m.get("avg_price"))
        value_score = _to_float(m.get("value_score"))

        cluster = str(r.get("cluster") or "").strip()
        if not cluster or cluster.lower() == "unknown":
            cluster = cluster_map.get(name) or _derive_cluster(avg_price, value_score)
            r["cluster"] = cluster

        short_profile = str(r.get("short_profile") or "").strip()
        if not short_profile:
            pf = [str(x) for x in (m.get("present_features") or []) if str(x or "").strip()]
            if pf:
                short_profile = "Signifikante Merkmale: " + _compact_feature_list(pf, max_items=3)
            else:
                short_profile = "Datenlage begrenzt."
            r["short_profile"] = short_profile

        if r.get("mapped_features") in (None, 0):
            if isinstance(m.get("present_features"), list):
                r["mapped_features"] = len([x for x in m.get("present_features") if str(x or "").strip()])

        dedup_comp_rows.append(r)

    # Sort competitors by relevance desc
    dedup_comp_rows.sort(key=lambda x: (_to_float(x.get("relevance_score")) or 0.0), reverse=True)

    comp["table"] = dedup_comp_rows
    comp["count"] = len(dedup_comp_rows)
    comp["pagination"] = {
        "page_size": 20,
        "pages": max(1, (len(dedup_comp_rows) + 19) // 20),
    }

    matrix["rows"] = dedup_matrix_rows
    matrix["pagination"] = {
        "page_size": 15,
        "pages": max(1, (len(dedup_matrix_rows) + 14) // 15),
    }

    report["competitor_overview"] = comp
    report["feature_matrix_section"] = matrix
    return report


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
            try:
                return json.loads(text)
            except Exception:
                return {}
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
        try:
            return json.loads(c2.extract_text(comp))
        except Exception:
            return {}
    except Exception as exc:
        warnings.append(f"IONOS call failed ({schema_name}): {exc}")
        return {}


def _render_value(v: Any) -> str:
    if isinstance(v, list):
        vals = [str(x).strip() for x in v if str(x or "").strip()]
        return ", ".join(vals)
    if isinstance(v, dict):
        return ", ".join(f"{k}: {v2}" for k, v2 in v.items())
    return str(v or "").strip()


def _build_professional_sections(report: Dict[str, Any], provider: str, warnings: List[str]) -> Dict[str, Any]:
    pb = report.get("product_profile_brief") if isinstance(report.get("product_profile_brief"), dict) else {}
    comp = report.get("competitor_overview") if isinstance(report.get("competitor_overview"), dict) else {}
    matrix = report.get("feature_matrix_section") if isinstance(report.get("feature_matrix_section"), dict) else {}
    gap = report.get("gap_usp_analysis") if isinstance(report.get("gap_usp_analysis"), dict) else {}
    swot = report.get("swot") if isinstance(report.get("swot"), dict) else {}
    pos = report.get("positioning_diagram") if isinstance(report.get("positioning_diagram"), dict) else {}
    recs = report.get("strategic_recommendations") if isinstance(report.get("strategic_recommendations"), list) else []

    produktprofil_de = {
        "felder": [
            {"label": "Produktname", "value": _render_value(pb.get("product_name"))},
            {"label": "Hersteller", "value": _render_value(pb.get("manufacturer"))},
            {"label": "Kategorie", "value": _render_value(pb.get("category"))},
            {"label": "Zielsegmente", "value": _render_value(pb.get("target_segments"))},
            {"label": "Use Cases", "value": _render_value(pb.get("use_cases"))},
            {"label": "Schluessel-Differenzierer", "value": _render_value(pb.get("key_differentiators"))},
            {"label": "Claims", "value": _render_value(pb.get("top_claims"))},
            {"label": "Feature-Anzahl", "value": _render_value(pb.get("feature_count"))},
        ],
        "claims_aufzaehlung": [str(x).strip() for x in (pb.get("top_claims") or []) if str(x or "").strip()],
    }

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "executive_summary_text": {"type": "string"},
            "competitor_overview_intro_text": {"type": "string"},
            "feature_matrix_performance_text": {"type": "string"},
            "feature_matrix_soft_text": {"type": "string"},
            "feature_matrix_price_text": {"type": "string"},
            "feature_matrix_weighting_text": {"type": "string"},
            "gaps_text": {"type": "string"},
            "usps_text": {"type": "string"},
            "swot_strengths_text": {"type": "string"},
            "swot_weaknesses_text": {"type": "string"},
            "swot_opportunities_text": {"type": "string"},
            "swot_threats_text": {"type": "string"},
            "positioning_intro_text": {"type": "string"},
            "positioning_insights_text": {"type": "string"},
            "recommendations_intro_text": {"type": "string"}
        },
        "required": [
            "executive_summary_text",
            "competitor_overview_intro_text",
            "feature_matrix_performance_text",
            "feature_matrix_soft_text",
            "feature_matrix_price_text",
            "feature_matrix_weighting_text",
            "gaps_text",
            "usps_text",
            "swot_strengths_text",
            "swot_weaknesses_text",
            "swot_opportunities_text",
            "swot_threats_text",
            "positioning_intro_text",
            "positioning_insights_text",
            "recommendations_intro_text"
        ]
    }

    payload = {
        "product_profile_brief": pb,
        "produktprofil_de": produktprofil_de,
        "competitor_overview": {
            "count": comp.get("count"),
            "top_rows": (comp.get("table") or [])[:12],
        },
        "feature_matrix_section": matrix,
        "gap_usp_analysis": gap,
        "swot": swot,
        "positioning_diagram": pos,
        "strategic_recommendations": recs,
    }

    llm_out = _llm_call_structured(
        provider=provider,
        schema_name="final_report_sections_v06",
        schema=schema,
        system=(
            "Du schreibst einen professionellen Wettbewerbsanalyse-Bericht auf Deutsch. "
            "Nur Fliesstext, keine Markdown-Symbole. "
            "Praezise, analytisch, managementtauglich. "
            "Keine Halluzinationen; nur bereitgestellte Daten verwenden."
        ),
        user=(
            "Erzeuge die Abschnitte gemaess Schema. "
            "Executive Summary als Fliesstext. "
            "Feature Matrix in drei Abschnitte (Performance, Soft Features, Preis) plus eigener Absatz zu Gewichtung/Value-Berechnung. "
            "Gap/USP je ein Fliesstext-Absatz. SWOT vier Abschnitte als Fliesstext. "
            "Positionierung mit kurzer 2x2-Erklaerung + wichtigste Erkenntnisse. "
            "Vor den strategischen Empfehlungen einen professionellen Einleitungsabsatz.\n\n"
            + json.dumps(payload, ensure_ascii=False)
        ),
        warnings=warnings,
    )

    if not llm_out:
        # Minimal deterministic fallback (no fabricated facts)
        llm_out = {
            "executive_summary_text": f"{_render_value(pb.get('product_name'))} ist im Segment {_render_value(pb.get('category'))} positioniert und adressiert die Segmente {_render_value(pb.get('target_segments'))}.",
            "competitor_overview_intro_text": f"Die Wettbewerbsuebersicht umfasst {len(comp.get('table') or [])} relevante Modelle. Cluster wurden fuer bislang unbekannte Eintraege datenbasiert aufgeloest.",
            "feature_matrix_performance_text": "Die Performance-Features zeigen eine hohe Abdeckung zentraler technischen Kennzahlen im Vergleichsfeld.",
            "feature_matrix_soft_text": "Bei Soft-Features liegt der Fokus auf Bedienkomfort, Automationsgrad und Pflegefunktionen.",
            "feature_matrix_price_text": "Die Preisdimension ordnet die Modelle entlang Preisniveau und wahrgenommenem Wertbeitrag ein.",
            "feature_matrix_weighting_text": "Der Value-Score wird als normierter Vergleichswert aus Feature-Abdeckung und Preisniveau verwendet; hoehere Werte bedeuten relativ staerkere Positionierung innerhalb des betrachteten Sets.",
            "gaps_text": "Die priorisierten Gaps markieren Merkmale mit hoher Marktpraesenz, die im Zielprodukt ausgebaut werden sollten.",
            "usps_text": "Die priorisierten USPs beschreiben Merkmale, mit denen sich das Zielprodukt im Wettbewerbsumfeld differenzieren kann.",
            "swot_strengths_text": ", ".join(_safe_list_str(swot.get("strengths"))) or "Keine belastbaren Staerken identifiziert.",
            "swot_weaknesses_text": ", ".join(_safe_list_str(swot.get("weaknesses"))) or "Keine belastbaren Schwaechen identifiziert.",
            "swot_opportunities_text": ", ".join(_safe_list_str(swot.get("opportunities"))) or "Keine belastbaren Chancen identifiziert.",
            "swot_threats_text": ", ".join(_safe_list_str(swot.get("threats"))) or "Keine belastbaren Gefahren identifiziert.",
            "positioning_intro_text": "Die 2x2-Positionierung stellt Preisniveau (X) und Leistungs-/Wertbeitrag (Y) gegenueber und zeigt die relative Marktpositionierung je Modell.",
            "positioning_insights_text": ", ".join(_safe_list_str(pos.get("interpretation"))) or "Keine zusaetzlichen Positionierungserkenntnisse.",
            "recommendations_intro_text": "Die folgenden Empfehlungen priorisieren die naechsten Schritte entlang Wirkung, Zeithorizont und Evidenzlage.",
        }

    llm_out["produktprofil_de"] = produktprofil_de
    return llm_out


def build_final_report_v0_6(
    *,
    artifacts: Optional[Dict[str, Any]],
    artifact_paths: Optional[Dict[str, str]],
    provider: str = "openai",
    max_chars_per_artifact: int = 12000,
    user_root: Path,
    work_root: Path,
) -> FinalReportResult:
    base = build_final_report_v0_5(
        artifacts=artifacts,
        artifact_paths=artifact_paths,
        provider=provider,
        max_chars_per_artifact=max_chars_per_artifact,
        user_root=user_root,
        work_root=work_root,
    )

    warnings = list(base.extraction_warnings or [])
    report_dict = base.final_report.model_dump()
    report_dict = _enhance_sections(report_dict)
    sections_v06 = _build_professional_sections(report_dict, provider=provider, warnings=warnings)
    report_dict["report_sections_v0_6"] = sections_v06

    # Keep legacy field shape but make executive summary truly flowing text.
    exec_text = str(sections_v06.get("executive_summary_text") or "").strip()
    report_dict["executive_summary"] = [exec_text] if exec_text else []

    final_report = FinalReport(**report_dict)
    artifact_chunks_v06 = [
        ArtifactChunk(**(c.model_dump() if hasattr(c, "model_dump") else dict(c)))
        for c in (base.artifact_chunks or [])
    ]
    validation_v06 = ValidationReport(
        **(base.validation.model_dump() if hasattr(base.validation, "model_dump") else dict(base.validation))
    )

    return FinalReportResult(
        provider=base.provider,
        report_context=base.report_context,
        artifact_chunks=artifact_chunks_v06,
        final_report=final_report,
        validation=validation_v06,
        extraction_warnings=warnings,
    )
