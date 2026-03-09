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


def _empty_swot_and_positioning(warnings: List[str]) -> StrategicAnalysisResult:
    return StrategicAnalysisResult(
        provider="ionos",
        swot=SwotData(strengths=[], weaknesses=[], opportunities=[], threats=[]),
        positioning_data=PositioningData(
            market_space="",
            primary_axis_x="",
            primary_axis_y="",
            position_label="",
            competitor_clusters=[],
        ),
        strategic_implications=[],
        extraction_warnings=warnings,
    )


def _to_float(v: Any) -> Optional[float]:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().replace(",", ".")
        try:
            return float(s)
        except Exception:
            return None
    return None


def _compute_positioning_data(payload: Dict[str, Any]) -> PositioningData:
    cm = payload.get("comparison_matrix") if isinstance(payload.get("comparison_matrix"), dict) else {}
    ca = payload.get("cluster_assignment") if isinstance(payload.get("cluster_assignment"), list) else []

    rows: List[Dict[str, Any]] = []
    base_row = cm.get("baseline_row")
    if isinstance(base_row, dict):
        rows.append(base_row)
    comp_rows = cm.get("competitor_rows")
    if isinstance(comp_rows, list):
        rows.extend([r for r in comp_rows if isinstance(r, dict)])

    ca_by_name: Dict[str, Dict[str, Any]] = {}
    for item in ca:
        if not isinstance(item, dict):
            continue
        n = str(item.get("competitor") or "").strip()
        if n:
            ca_by_name[n] = item

    priced: List[float] = []
    for r in rows:
        ap = _to_float(r.get("avg_price"))
        if ap is not None:
            priced.append(ap)
    median_price = sorted(priced)[len(priced) // 2] if priced else None

    clusters: List[Dict[str, Any]] = []
    for r in rows:
        name = str(r.get("competitor") or "").strip()
        if not name:
            continue
        avg_price = _to_float(r.get("avg_price"))
        value_score = _to_float(r.get("value_score"))
        label = str(r.get("cluster") or "unknown").strip() or "unknown"

        if name in ca_by_name:
            cx = ca_by_name[name]
            label = str(cx.get("cluster") or label).strip() or "unknown"
            if cx.get("avg_price") is not None:
                avg_price = _to_float(cx.get("avg_price"))
            if cx.get("value_score") is not None:
                value_score = _to_float(cx.get("value_score"))

        if label == "unknown" and median_price is not None and avg_price is not None and value_score is not None:
            if avg_price <= median_price and value_score >= 0.4:
                label = "value_leader"
            elif avg_price > median_price and value_score >= 0.4:
                label = "premium_performer"
            elif avg_price <= median_price and value_score < 0.4:
                label = "budget_basic"
            else:
                label = "premium_basic"

        clusters.append(
            {
                "competitor": name,
                "cluster": label,
                "avg_price": avg_price,
                "value_score": value_score,
            }
        )

    target = next((x for x in clusters if str(x.get("competitor") or "").strip().lower() == "target"), None)
    if target is None:
        target = next((x for x in clusters if str(x.get("cluster") or "").strip().lower() == "target"), None)
    if target is None and clusters:
        target = clusters[0]
    position_label = str((target or {}).get("cluster") or "").strip()

    market_space = str(cm.get("baseline_product") or "").strip()
    return PositioningData(
        market_space=market_space,
        primary_axis_x="Preisniveau",
        primary_axis_y="Leistungs-/Wertbeitrag",
        position_label=position_label,
        competitor_clusters=clusters,
    )


def _as_statements(values: Any, default_evidence_ref: str) -> List[PrioritizedStatement]:
    out: List[PrioritizedStatement] = []
    if not isinstance(values, list):
        return out
    for v in values:
        if isinstance(v, dict):
            s = str(v.get("statement") or "").strip()
            if not s:
                continue
            refs = [str(x).strip() for x in (v.get("evidence_refs") or []) if str(x).strip()]
            out.append(
                PrioritizedStatement(
                    statement=s,
                    confidence=float(v.get("confidence") or 0.6),
                    impact=float(v.get("impact") or 0.6),
                    relevance=float(v.get("relevance") or 0.6),
                    evidence=str(v.get("evidence") or "").strip(),
                    evidence_refs=refs or [default_evidence_ref],
                )
            )
        else:
            s = str(v or "").strip()
            if not s:
                continue
            out.append(
                PrioritizedStatement(
                    statement=s,
                    confidence=0.6,
                    impact=0.6,
                    relevance=0.6,
                    evidence="",
                    evidence_refs=[default_evidence_ref],
                )
            )
    return out


def _heuristic_swot_and_implications(payload: Dict[str, Any], positioning: PositioningData) -> tuple[SwotData, List[StrategicImplication]]:
    gaps = [x for x in (payload.get("gaps") or []) if isinstance(x, dict)]
    usps = [x for x in (payload.get("usps") or []) if isinstance(x, dict)]
    market = [str(x).strip() for x in (payload.get("market_standards") or []) if str(x).strip()]
    differentiators = [str(x).strip() for x in (payload.get("differentiators") or []) if str(x).strip()]
    clusters = [x for x in (payload.get("cluster_assignment") or []) if isinstance(x, dict)]

    strengths: List[PrioritizedStatement] = []
    for u in usps[:4]:
        feat = str(u.get("feature") or "").strip()
        rat = str(u.get("rationale") or "").strip()
        if feat:
            strengths.append(
                PrioritizedStatement(
                    statement=f"USP: {feat}",
                    confidence=0.7,
                    impact=0.7,
                    relevance=0.7,
                    evidence=rat,
                    evidence_refs=["gaps_and_usps.usps"],
                )
            )
    for d in differentiators[:2]:
        strengths.append(
            PrioritizedStatement(
                statement=f"Differenzierung über {d}",
                confidence=0.65,
                impact=0.7,
                relevance=0.65,
                evidence="",
                evidence_refs=["gaps_and_usps.differentiators"],
            )
        )

    weaknesses: List[PrioritizedStatement] = []
    for g in gaps[:5]:
        feat = str(g.get("feature") or "").strip()
        rec = str(g.get("recommendation") or "").strip()
        if feat:
            weaknesses.append(
                PrioritizedStatement(
                    statement=f"Feature-Lücke bei {feat}",
                    confidence=0.75,
                    impact=0.72,
                    relevance=0.72,
                    evidence=rec,
                    evidence_refs=["gaps_and_usps.gaps"],
                )
            )

    opportunities: List[PrioritizedStatement] = []
    for g in gaps[:3]:
        feat = str(g.get("feature") or "").strip()
        if feat:
            opportunities.append(
                PrioritizedStatement(
                    statement=f"Roadmap-Chance: {feat} zur Marktparität ausbauen",
                    confidence=0.68,
                    impact=0.74,
                    relevance=0.7,
                    evidence=str(g.get("recommendation") or "").strip(),
                    evidence_refs=["gaps_and_usps.gaps"],
                )
            )
    if market:
        opportunities.append(
            PrioritizedStatement(
                statement="Marktstandards gezielt übertreffen statt nur matchen",
                confidence=0.66,
                impact=0.71,
                relevance=0.68,
                evidence=", ".join(market[:4]),
                evidence_refs=["gaps_and_usps.market_standards"],
            )
        )

    threats: List[PrioritizedStatement] = []
    top_cluster = None
    for c in clusters:
        cl = str(c.get("cluster") or "").strip().lower()
        if cl:
            top_cluster = cl
            break
    if top_cluster:
        threats.append(
            PrioritizedStatement(
                statement=f"Hoher Wettbewerbsdruck im Cluster '{top_cluster}'",
                confidence=0.72,
                impact=0.75,
                relevance=0.72,
                evidence="",
                evidence_refs=["cluster_assignment"],
            )
        )
    threats.append(
        PrioritizedStatement(
            statement="Risiko der Funktionsparität durch ähnliche Feature-Sets im Wettbewerb",
            confidence=0.7,
            impact=0.7,
            relevance=0.69,
            evidence="",
            evidence_refs=["comparison_matrix.competitor_rows"],
        )
    )

    swot = SwotData(
        strengths=strengths[:10],
        weaknesses=weaknesses[:10],
        opportunities=opportunities[:10],
        threats=threats[:10],
    )

    recs: List[StrategicImplication] = []
    for w in swot.weaknesses[:2]:
        recs.append(
            StrategicImplication(
                title="Gap schließen",
                action=w.statement,
                horizon="short-term",
                priority="high",
            )
        )
    for s in swot.strengths[:2]:
        recs.append(
            StrategicImplication(
                title="USP skalieren",
                action=s.statement,
                horizon="short-term",
                priority="high",
            )
        )
    if not recs:
        recs.append(
            StrategicImplication(
                title="Positionierung schärfen",
                action=f"Position '{positioning.position_label or 'unknown'}' mit klaren Nutzenbotschaften im Markt verankern.",
                horizon="mid-term",
                priority="medium",
            )
        )

    return swot, recs[:6]


def _llm_refine(provider: str, base: StrategicAnalysisResult, context_payload: Dict[str, Any], warnings: List[str]) -> StrategicAnalysisResult:
    p = str(provider or "ionos").strip().lower()
    if p not in {"ionos", "openai", "perplexity"}:
        p = "ionos"

    swot_item_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "statement": {"type": "string"},
            "confidence": {"type": "number"},
            "impact": {"type": "number"},
            "relevance": {"type": "number"},
            "evidence": {"type": "string"},
            "evidence_refs": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        },
        "required": ["statement", "confidence", "impact", "relevance", "evidence", "evidence_refs"],
    }

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "swot": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "strengths": {"type": "array", "items": swot_item_schema, "minItems": 1},
                    "weaknesses": {"type": "array", "items": swot_item_schema, "minItems": 1},
                    "opportunities": {"type": "array", "items": swot_item_schema, "minItems": 1},
                    "threats": {"type": "array", "items": swot_item_schema, "minItems": 1},
                },
                "required": ["strengths", "weaknesses", "opportunities", "threats"],
            },
            "strategic_implications": {
                "type": "array",
                "minItems": 1,
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
        "required": ["swot", "strategic_implications"],
    }

    system = (
        "Erzeuge eine strategische SWOT-Analyse und strategische Implikationen auf Basis der gegebenen Evidenzen. "
        "Die Positionierungsdaten sind bereits deterministisch berechnet und dürfen nicht neu berechnet werden. "
        "Nutze sie nur zur verbalen Einordnung in SWOT/Implikationen. "
        "Nutze ausschließlich folgende Quellen aus dem Kontext: "
        "comparison_matrix.baseline_row, comparison_matrix.competitor_rows, gaps_and_usps.gaps, gaps_and_usps.usps, cluster_assignment. "
        "Jede SWOT-Aussage muss evidence_refs mit mindestens einem dieser Quellenpfade enthalten. "
        "Trenne interne/externe Faktoren sauber und priorisiere Aussagen über confidence/impact/relevance. "
        "Antworte strikt als JSON gemäß Schema."
    )
    user = "Kontext:\n" + json.dumps(context_payload, ensure_ascii=False)

    if p in {"openai", "perplexity"}:
        client = LlmOpenai() if p == "openai" else LlmPerplexity()
        if not client.enabled():
            warnings.append(f"{p} not configured; no heuristic strategic text fallback used.")
            return base
        try:
            resp = client._call(
                input_messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                text_format={"type": "json_schema", "name": "strategic_analysis", "schema": schema, "strict": True},
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
            warnings.append("IONOS not configured; no heuristic strategic text fallback used.")
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
        swot_data = parsed.get("swot") if isinstance(parsed.get("swot"), dict) else {}
        strengths = _as_statements(swot_data.get("strengths"), "llm.swot.strengths")
        weaknesses = _as_statements(swot_data.get("weaknesses"), "llm.swot.weaknesses")
        opportunities = _as_statements(swot_data.get("opportunities"), "llm.swot.opportunities")
        threats = _as_statements(swot_data.get("threats"), "llm.swot.threats")

        merged_swot = SwotData(
            strengths=strengths or base.swot.strengths,
            weaknesses=weaknesses or base.swot.weaknesses,
            opportunities=opportunities or base.swot.opportunities,
            threats=threats or base.swot.threats,
        )

        impls: List[StrategicImplication] = []
        for x in (parsed.get("strategic_implications") or []):
            if not isinstance(x, dict):
                continue
            t = str(x.get("title") or "").strip()
            a = str(x.get("action") or "").strip()
            if not t or not a:
                continue
            impls.append(
                StrategicImplication(
                    title=t,
                    action=a,
                    horizon=str(x.get("horizon") or "mid-term").strip(),
                    priority=str(x.get("priority") or "medium").strip(),
                )
            )

        out = StrategicAnalysisResult(
            provider=p,
            swot=merged_swot,
            positioning_data=base.positioning_data,
            strategic_implications=impls or base.strategic_implications,
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
    base = _empty_swot_and_positioning(warnings=warnings)
    base.positioning_data = _compute_positioning_data(payload)
    heur_swot, heur_impl = _heuristic_swot_and_implications(payload, base.positioning_data)
    base.swot = heur_swot
    base.strategic_implications = heur_impl

    cm = payload.get("comparison_matrix") if isinstance(payload.get("comparison_matrix"), dict) else {}
    context_payload = {
        "comparison_matrix": {
            "baseline_row": cm.get("baseline_row"),
            "competitor_rows": cm.get("competitor_rows"),
        },
        "gaps_and_usps": {
            "gaps": payload.get("gaps"),
            "usps": payload.get("usps"),
        },
        "cluster_assignment": payload.get("cluster_assignment"),
    }

    refined = _llm_refine(provider=provider, base=base, context_payload=context_payload, warnings=warnings)
    p_norm = str(provider or "ionos").strip().lower()
    refined.provider = p_norm if p_norm in {"ionos", "openai", "perplexity"} else "ionos"
    return refined
