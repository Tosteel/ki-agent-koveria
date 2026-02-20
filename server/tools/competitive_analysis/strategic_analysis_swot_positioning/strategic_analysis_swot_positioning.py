from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from server.services.llm_ionos import IonosLLM
from server.services.llm_openai import LlmOpenai
from server.services.llm_perplexity import LlmPerplexity

from .models import (
    PositioningData,
    PrioritizedStatement,
    StrategicAnalysisResult,
    StrategicImplication,
    SwotData,
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


def _load_gaps_payload(
    *,
    gaps_and_usps: Optional[Dict[str, Any]],
    gaps_and_usps_path: Optional[str],
    user_root: Path,
    work_root: Path,
) -> Dict[str, Any]:
    payload: Dict[str, Any]
    if isinstance(gaps_and_usps, dict) and gaps_and_usps:
        payload = gaps_and_usps
    else:
        p = _resolve_input_path(str(gaps_and_usps_path or ""), user_root=user_root.resolve(), work_root=work_root.resolve())
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON in path: {gaps_and_usps_path}") from exc

    # tolerate combined wrapper documents
    if "gaps_and_usps" in payload and isinstance(payload.get("gaps_and_usps"), dict):
        payload = payload["gaps_and_usps"] | {
            "comparison_matrix": payload.get("comparison_matrix"),
            "cluster_assignment": payload.get("cluster_assignment"),
            "extraction_warnings": payload.get("extraction_warnings"),
        }

    return payload


def _prio(statement: str, confidence: float, impact: float, relevance: float, evidence: str = "") -> PrioritizedStatement:
    return PrioritizedStatement(
        statement=statement,
        confidence=max(0.0, min(1.0, round(confidence, 4))),
        impact=max(0.0, min(1.0, round(impact, 4))),
        relevance=max(0.0, min(1.0, round(relevance, 4))),
        evidence=evidence,
    )


def _heuristic_swot_and_positioning(payload: Dict[str, Any], evidences: Optional[Dict[str, Any]], warnings: List[str]) -> StrategicAnalysisResult:
    gaps = payload.get("gaps") if isinstance(payload.get("gaps"), list) else []
    usps = payload.get("usps") if isinstance(payload.get("usps"), list) else []
    market_standards = _safe_list_str(payload.get("market_standards"))
    differentiators = _safe_list_str(payload.get("differentiators"))

    comparison_matrix = payload.get("comparison_matrix") if isinstance(payload.get("comparison_matrix"), dict) else {}
    cluster_assignment = payload.get("cluster_assignment") if isinstance(payload.get("cluster_assignment"), list) else []

    strengths: List[PrioritizedStatement] = []
    weaknesses: List[PrioritizedStatement] = []
    opportunities: List[PrioritizedStatement] = []
    threats: List[PrioritizedStatement] = []

    for u in usps[:8]:
        if isinstance(u, dict):
            feat = str(u.get("feature") or "").strip()
            rat = str(u.get("rationale") or "").strip()
            if feat:
                strengths.append(_prio(f"Differenzierungsmerkmal: {feat}", 0.78, 0.75, 0.80, rat))

    for d in differentiators[:8]:
        strengths.append(_prio(f"Wettbewerbliche Differenzierung über {d}", 0.72, 0.70, 0.78, "Feature-Matrix"))

    for g in gaps[:10]:
        if not isinstance(g, dict):
            continue
        feat = str(g.get("feature") or "").strip()
        ratio = float(g.get("market_presence_ratio") or 0.0)
        rec = str(g.get("recommendation") or "").strip()
        if not feat:
            continue
        weaknesses.append(_prio(f"Feature-Lücke bei {feat}", min(0.95, 0.5 + ratio / 2.0), 0.75, 0.82, rec))
        opportunities.append(_prio(f"Roadmap-Chance: {feat} zur Marktparität ausbauen", 0.68, 0.72, 0.80, rec))

    if market_standards:
        opportunities.append(
            _prio(
                "Marktstandard gezielt übertreffen statt nur matchen",
                0.62,
                0.74,
                0.71,
                ", ".join(market_standards[:6]),
            )
        )

    # external risks: substitution + competitive pressure from cluster density
    cluster_labels = [str(c.get("cluster") or "unknown").strip() for c in cluster_assignment if isinstance(c, dict)]
    if cluster_labels:
        dominant = max(set(cluster_labels), key=cluster_labels.count)
        threats.append(
            _prio(
                f"Hoher Wettbewerbsdruck im Cluster '{dominant}'",
                0.66,
                0.69,
                0.70,
                "Cluster-Zuordnung aus Preis-/Leistungsraum",
            )
        )

    threats.append(
        _prio(
            "Risiko technologischer Substitution durch alternative Lüftungs-/Automationskonzepte",
            0.55,
            0.73,
            0.67,
            "Branchenübliche Innovationsdynamik",
        )
    )

    # ensure lists not empty
    if not strengths:
        strengths.append(_prio("Keine klaren USPs nachweisbar - Fokus auf Schärfung der Positionierung", 0.45, 0.55, 0.60, "Heuristik"))
    if not weaknesses:
        weaknesses.append(_prio("Keine signifikanten Feature-Lücken identifiziert", 0.5, 0.45, 0.5, "Heuristik"))

    baseline_name = str(comparison_matrix.get("baseline_product") or "target_product").strip() if comparison_matrix else "target_product"
    pos_label = "balanced_player"
    if len(differentiators) >= 3:
        pos_label = "differentiated_specialist"
    elif len(gaps) >= 4:
        pos_label = "catch_up_challenger"

    competitor_clusters: List[Dict[str, Any]] = []
    for c in cluster_assignment[:20]:
        if isinstance(c, dict):
            competitor_clusters.append(
                {
                    "competitor": str(c.get("competitor") or ""),
                    "cluster": str(c.get("cluster") or "unknown"),
                    "avg_price": c.get("avg_price"),
                    "value_score": c.get("value_score"),
                }
            )

    swot = SwotData(
        strengths=strengths[:10],
        weaknesses=weaknesses[:10],
        opportunities=opportunities[:10],
        threats=threats[:10],
    )

    positioning = PositioningData(
        market_space="Wettbewerbsraum basierend auf Feature-Abdeckung und Preis-/Leistungsclustern",
        primary_axis_x="Preisniveau",
        primary_axis_y="Leistungs-/Wertbeitrag",
        position_label=f"{baseline_name}: {pos_label}",
        competitor_clusters=competitor_clusters,
    )

    implications: List[StrategicImplication] = []
    for w in weaknesses[:3]:
        implications.append(
            StrategicImplication(
                title="Lücken schließen",
                action=w.statement,
                horizon="short-term",
                priority="high",
            )
        )
    for s in strengths[:2]:
        implications.append(
            StrategicImplication(
                title="USP vermarkten",
                action=s.statement,
                horizon="short-term",
                priority="high",
            )
        )
    implications.append(
        StrategicImplication(
            title="Substitutionsrisiko absichern",
            action="Technologieradar und Partnerschaften für alternative Lösungen etablieren.",
            horizon="mid-term",
            priority="medium",
        )
    )

    provider_val = "ionos"
    return StrategicAnalysisResult(
        provider=provider_val,
        swot=swot,
        positioning_data=positioning,
        strategic_implications=implications[:8],
        extraction_warnings=warnings,
    )


def _llm_refine(provider: str, base: StrategicAnalysisResult, context_payload: Dict[str, Any], warnings: List[str]) -> StrategicAnalysisResult:
    p = str(provider or "ionos").strip().lower()
    if p not in {"ionos", "openai", "perplexity"}:
        p = "ionos"

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "swot": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "strengths": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {"statement": {"type": "string"}, "confidence": {"type": "number"}, "impact": {"type": "number"}, "relevance": {"type": "number"}, "evidence": {"type": "string"}}, "required": ["statement", "confidence", "impact", "relevance", "evidence"]}},
                    "weaknesses": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {"statement": {"type": "string"}, "confidence": {"type": "number"}, "impact": {"type": "number"}, "relevance": {"type": "number"}, "evidence": {"type": "string"}}, "required": ["statement", "confidence", "impact", "relevance", "evidence"]}},
                    "opportunities": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {"statement": {"type": "string"}, "confidence": {"type": "number"}, "impact": {"type": "number"}, "relevance": {"type": "number"}, "evidence": {"type": "string"}}, "required": ["statement", "confidence", "impact", "relevance", "evidence"]}},
                    "threats": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {"statement": {"type": "string"}, "confidence": {"type": "number"}, "impact": {"type": "number"}, "relevance": {"type": "number"}, "evidence": {"type": "string"}}, "required": ["statement", "confidence", "impact", "relevance", "evidence"]}},
                },
                "required": ["strengths", "weaknesses", "opportunities", "threats"],
            },
            "positioning_data": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "market_space": {"type": "string"},
                    "primary_axis_x": {"type": "string"},
                    "primary_axis_y": {"type": "string"},
                    "position_label": {"type": "string"},
                    "competitor_clusters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "competitor": {"type": "string"},
                                "cluster": {"type": "string"},
                                "avg_price": {"type": ["number", "null"]},
                                "value_score": {"type": ["number", "null"]},
                            },
                            "required": ["competitor", "cluster", "avg_price", "value_score"],
                        },
                    },
                },
                "required": ["market_space", "primary_axis_x", "primary_axis_y", "position_label", "competitor_clusters"],
            },
            "strategic_implications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "title": {"type": "string"},
                        "action": {"type": "string"},
                        "horizon": {"type": "string"},
                        "priority": {"type": "string"},
                    },
                    "required": ["title", "action", "horizon", "priority"],
                },
            },
        },
        "required": ["swot", "positioning_data", "strategic_implications"],
    }

    system = (
        "Erzeuge eine strategische SWOT- und Positionierungsanalyse auf Basis der gegebenen Evidenzen. "
        "Trenne interne/externe Faktoren sauber und priorisiere Aussagen über confidence/impact/relevance. "
        "Antworte strikt als JSON gemäß Schema."
    )
    user = "Kontext:\n" + json.dumps(context_payload, ensure_ascii=False)

    if p in {"openai", "perplexity"}:
        client = LlmOpenai() if p == "openai" else LlmPerplexity()
        if not client.enabled():
            warnings.append(f"{p} not configured; heuristic strategic analysis used.")
            return base
        try:
            resp = client._call(
                input_messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                text_format={"type": "json_schema", "name": "strategic_analysis", "schema": schema, "strict": False},
            )
            text = ""
            for item in resp.get("output", []):
                for c in item.get("content", []):
                    if c.get("type") == "output_text":
                        text += str(c.get("text") or "")
            parsed = _parse_json_strictish(text)
        except Exception as exc:
            warnings.append(f"{p} strategic analysis failed: {exc}")
            return base
    else:
        client = IonosLLM()
        if not client.enabled():
            warnings.append("IONOS not configured; heuristic strategic analysis used.")
            return base
        try:
            comp = client.chat_completions(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "strategic_analysis", "schema": schema, "strict": True},
                },
            )
            parsed = _parse_json_strictish(client.extract_text(comp))
        except Exception as exc:
            warnings.append(f"IONOS strategic analysis failed: {exc}")
            return base

    if not parsed:
        return base

    try:
        out = StrategicAnalysisResult(
            provider=p,
            swot=SwotData(**parsed.get("swot", {})),
            positioning_data=PositioningData(**parsed.get("positioning_data", {})),
            strategic_implications=[StrategicImplication(**x) for x in (parsed.get("strategic_implications") or []) if isinstance(x, dict)],
            extraction_warnings=warnings,
        )
        return out
    except Exception as exc:
        warnings.append(f"Strategic analysis output invalid; heuristic kept ({exc})")
        return base


def run_strategic_analysis(
    *,
    gaps_and_usps: Optional[Dict[str, Any]],
    gaps_and_usps_path: Optional[str],
    evidences: Optional[Dict[str, Any]],
    provider: str = "ionos",
    user_root: Path,
    work_root: Path,
) -> StrategicAnalysisResult:
    payload = _load_gaps_payload(
        gaps_and_usps=gaps_and_usps,
        gaps_and_usps_path=gaps_and_usps_path,
        user_root=user_root,
        work_root=work_root,
    )

    warnings = _safe_list_str(payload.get("extraction_warnings"))
    base = _heuristic_swot_and_positioning(payload=payload, evidences=evidences, warnings=warnings)

    context_payload = {
        "gaps_and_usps": {
            "gaps": payload.get("gaps"),
            "usps": payload.get("usps"),
            "market_standards": payload.get("market_standards"),
            "differentiators": payload.get("differentiators"),
        },
        "comparison_matrix": payload.get("comparison_matrix"),
        "cluster_assignment": payload.get("cluster_assignment"),
        "evidences": evidences or {},
        "heuristic_baseline": {
            "swot": base.swot.model_dump(),
            "positioning_data": base.positioning_data.model_dump(),
            "strategic_implications": [x.model_dump() for x in base.strategic_implications],
        },
    }

    refined = _llm_refine(provider=provider, base=base, context_payload=context_payload, warnings=warnings)
    p_norm = str(provider or "ionos").strip().lower()
    refined.provider = p_norm if p_norm in {"ionos", "openai", "perplexity"} else "ionos"
    return refined
