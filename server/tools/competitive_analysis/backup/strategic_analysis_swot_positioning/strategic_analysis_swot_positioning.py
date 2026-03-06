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
            warnings.append(f"{p} not configured; no heuristic strategic text fallback used.")
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
    base = _empty_swot_and_positioning(warnings=warnings)

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
    }

    refined = _llm_refine(provider=provider, base=base, context_payload=context_payload, warnings=warnings)
    p_norm = str(provider or "ionos").strip().lower()
    refined.provider = p_norm if p_norm in {"ionos", "openai", "perplexity"} else "ionos"
    return refined
