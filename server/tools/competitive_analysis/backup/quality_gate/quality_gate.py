from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException

from server.services.llm_ionos import IonosLLM
from server.services.llm_openai import LlmOpenai
from server.services.llm_perplexity import LlmPerplexity

from .models import CompetitiveQualityGateReport, QualityIssue


_STEP_ROOTS = {
    1: "parsed_doc",
    2: "product_profile",
    3: "analysis_plan",
    4: "competitor_list",
    5: "competitor_profiles",
    6: "comparison_matrix",
    7: "strategic_analysis",
    8: "final_report",
    9: "review_status",
    10: "pdf_publish_result",
}

_STEP_EXPECTATIONS = {
    1: "parsed_doc: strukturierte Dokumentextraktion mit sinnvollen sections/measurements/metadata.",
    2: "product_profile: normalisierte Features/Claims ohne Artefakte, konsistente Einheiten, keine offensichtlichen Unsinns-Features.",
    3: "analysis_plan: klare Zielsetzung, relevante Wettbewerberkandidaten, konsistentes Feature-Schema.",
    4: "competitor_list: relevante Wettbewerber, valide URLs, keine Duplikate, keine Off-Topic-Kandidaten.",
    5: "competitor_profiles: pro Wettbewerber nachvollziehbare Features, Quellen und Datenqualität.",
    6: "comparison_matrix/feature_matrix_gap: konsistente Dimensionen, plausible Gaps/USPs, keine abgeschnittenen Feature-Namen.",
    7: "strategic_analysis: SWOT/Positionierung logisch aus Matrix ableitbar, keine fachfremden Risiken.",
    8: "final_report: konsistente Kernaussagen, keine Widersprüche zwischen USP/Gap, plausible Positionierung.",
    9: "review_status: klarer Freigabestatus und Begründung.",
    10: "pdf_publish_result: plausibler Publish-Status und Artefaktpfade.",
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


def _load_artifact(
    *,
    artifact: Optional[Dict[str, Any]],
    artifact_path: Optional[str],
    user_root: Path,
    work_root: Path,
) -> Dict[str, Any]:
    if isinstance(artifact, dict) and artifact:
        return artifact

    p = _resolve_input_path(str(artifact_path or ""), user_root=user_root.resolve(), work_root=work_root.resolve())
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in artifact_path: {artifact_path}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Artifact must be a JSON object.")
    return payload


def _detect_root_and_step(payload: Dict[str, Any], step_hint: Optional[int]) -> Tuple[str, Optional[int]]:
    if isinstance(payload.get("_step"), int):
        step_from_payload = int(payload["_step"])
    else:
        step_from_payload = None

    if step_hint in _STEP_ROOTS:
        rk = _STEP_ROOTS[step_hint]
        if rk in payload and isinstance(payload.get(rk), dict):
            return rk, step_hint

    for s, root in _STEP_ROOTS.items():
        if root in payload and isinstance(payload.get(root), dict):
            return root, (step_from_payload or s)

    # allow direct root object payload
    if step_hint in _STEP_ROOTS:
        return _STEP_ROOTS[step_hint], step_hint
    return "", step_from_payload


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


def _llm_assess_and_optionally_repair(
    *,
    provider: str,
    step: Optional[int],
    mode: str,
    root_key: str,
    root_obj: Dict[str, Any],
    max_context_chars: int,
) -> Tuple[float, List[QualityIssue], List[str], Optional[Dict[str, Any]]]:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "score": {"type": "number"},
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "severity": {"type": "string", "enum": ["error", "warning", "info"]},
                        "code": {"type": "string"},
                        "message": {"type": "string"},
                        "path": {"type": "string"},
                    },
                    "required": ["severity", "code", "message", "path"],
                },
            },
            "actions": {"type": "array", "items": {"type": "string"}},
            "repaired_root_object": {"anyOf": [{"type": "object"}, {"type": "null"}]},
        },
        "required": ["score", "issues", "actions", "repaired_root_object"],
    }
    expectation = _STEP_EXPECTATIONS.get(step or 0, "Schrittartefakt sollte inhaltlich konsistent, plausibel und verwendbar sein.")
    wants_repair = str(mode) == "validate_and_repair"
    sys = (
        "Du bist ein strenges Quality-Gate für Competitive-Analysis-Artefakte. "
        "Prüfe inhaltliche Plausibilität, Vollständigkeit, Konsistenz und Nutzbarkeit. "
        "Markiere echte Probleme mit präzisen Pfaden. "
        "Wenn reparieren gefordert ist: korrigiere minimal-invasiv, keine Halluzinationen, keine neuen externen Fakten erfinden, "
        "nur aus vorhandenem Kontext bereinigen/normalisieren. "
        "Wenn nicht reparieren: repaired_root_object muss null sein."
    )
    user = (
        f"Step: {step}\n"
        f"Mode: {mode}\n"
        f"Erwartung: {expectation}\n"
        f"Root key: {root_key}\n"
        f"Root object:\n{json.dumps(root_obj, ensure_ascii=False)[:max_context_chars]}"
    )

    p = str(provider or "openai").strip().lower()
    if p not in {"openai", "ionos", "perplexity"}:
        p = "openai"

    if p in {"openai", "perplexity"}:
        client = LlmOpenai() if p == "openai" else LlmPerplexity()
        if not client.enabled():
            raise HTTPException(status_code=400, detail=f"{p} not configured for competitive_quality_gate.")
        fmt = {"type": "json_schema", "name": "competitive_quality_gate", "schema": schema, "strict": False}
        try:
            resp = client._call(
                input_messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
                text_format=fmt,
            )
            parsed = _parse_json_strictish(_openai_extract_output_text(resp))
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"{p} quality gate failed: {exc}") from exc
    else:
        client = IonosLLM()
        if not client.enabled():
            raise HTTPException(status_code=400, detail="IONOS not configured for competitive_quality_gate.")
        response_format = {"type": "json_schema", "json_schema": {"name": "competitive_quality_gate", "schema": schema, "strict": True}}
        try:
            completion = client.chat_completions(
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
                response_format=response_format,
            )
            parsed = _parse_json_strictish(client.extract_text(completion))
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"IONOS quality gate failed: {exc}") from exc

    try:
        score = float(parsed.get("score", 0.0))
    except Exception:
        score = 0.0
    score = max(0.0, min(1.0, round(score, 4)))

    issues: List[QualityIssue] = []
    for x in (parsed.get("issues") or []):
        if not isinstance(x, dict):
            continue
        sev = str(x.get("severity") or "warning").strip().lower()
        if sev not in {"error", "warning", "info"}:
            sev = "warning"
        issues.append(
            QualityIssue(
                severity=sev,
                code=str(x.get("code") or "unspecified"),
                message=str(x.get("message") or ""),
                path=str(x.get("path") or ""),
            )
        )

    actions = [str(a).strip() for a in (parsed.get("actions") or []) if str(a or "").strip()]
    repaired_root = parsed.get("repaired_root_object")
    if wants_repair:
        if repaired_root is not None and not isinstance(repaired_root, dict):
            raise HTTPException(status_code=502, detail="LLM returned invalid repaired_root_object.")
        if repaired_root is None:
            actions.append("No repair object returned by LLM.")
    else:
        repaired_root = None
    return score, issues, actions, repaired_root


def run_competitive_quality_gate(
    *,
    artifact: Optional[Dict[str, Any]],
    artifact_path: Optional[str],
    step: Optional[int],
    mode: str,
    provider: str,
    max_context_chars: int,
    user_root: Path,
    work_root: Path,
) -> Tuple[Dict[str, Any], CompetitiveQualityGateReport]:
    payload = _load_artifact(
        artifact=artifact,
        artifact_path=artifact_path,
        user_root=user_root,
        work_root=work_root,
    )
    root_key, step_detected = _detect_root_and_step(payload, step_hint=step)
    root_obj = payload.get(root_key) if root_key and isinstance(payload.get(root_key), dict) else payload

    score, issues, actions, repaired_root = _llm_assess_and_optionally_repair(
        provider=provider,
        step=(step_detected or step),
        mode=mode,
        root_key=(root_key or "<direct>"),
        root_obj=root_obj,
        max_context_chars=max_context_chars,
    )

    repaired = isinstance(repaired_root, dict)
    out_payload = dict(payload)
    if repaired:
        if root_key:
            out_payload[root_key] = repaired_root
        else:
            out_payload = repaired_root

    report = CompetitiveQualityGateReport(
        step_detected=step_detected or step,
        root_key=root_key,
        score=score,
        issues=issues,
        actions=actions,
        repaired=repaired,
    )
    return out_payload, report
