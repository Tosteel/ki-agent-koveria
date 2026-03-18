from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import HTTPException

from server.workflows.competitive_analysis.backup.strategic_analysis_swot_positioning import (
    run_strategic_analysis,
)


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


def _load_feature_matrix_gap(
    *,
    feature_matrix_gap: Optional[Dict[str, Any]],
    feature_matrix_gap_path: Optional[str],
    user_root: Path,
    work_root: Path,
) -> Dict[str, Any]:
    if isinstance(feature_matrix_gap, dict) and feature_matrix_gap:
        payload = feature_matrix_gap
    else:
        p = _resolve_input_path(str(feature_matrix_gap_path or ""), user_root=user_root.resolve(), work_root=work_root.resolve())
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON in path: {feature_matrix_gap_path}") from exc

    # tolerate wrappers from file_write payloads
    if "feature_matrix_gap" in payload and isinstance(payload.get("feature_matrix_gap"), dict):
        payload = payload["feature_matrix_gap"]

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid feature_matrix_gap payload")
    return payload


def run_stratetic_analysis_swot_positioning_v0_5(
    *,
    feature_matrix_gap: Optional[Dict[str, Any]],
    feature_matrix_gap_path: Optional[str],
    comparison_matrix: Optional[Dict[str, Any]],
    gaps_and_usps: Optional[Dict[str, Any]],
    evidences: Optional[Dict[str, Any]],
    provider: str,
    user_root: Path,
    work_root: Path,
):
    payload: Dict[str, Any] = {}
    if isinstance(feature_matrix_gap, dict) and feature_matrix_gap:
        payload = _load_feature_matrix_gap(
            feature_matrix_gap=feature_matrix_gap,
            feature_matrix_gap_path=None,
            user_root=user_root,
            work_root=work_root,
        )
    elif str(feature_matrix_gap_path or "").strip():
        payload = _load_feature_matrix_gap(
            feature_matrix_gap=None,
            feature_matrix_gap_path=feature_matrix_gap_path,
            user_root=user_root,
            work_root=work_root,
        )

    cm = comparison_matrix if isinstance(comparison_matrix, dict) else payload.get("comparison_matrix")
    gu = gaps_and_usps if isinstance(gaps_and_usps, dict) else payload.get("gaps_and_usps")
    ca = payload.get("cluster_assignment") if isinstance(payload.get("cluster_assignment"), list) else []
    ew = payload.get("extraction_warnings") if isinstance(payload.get("extraction_warnings"), list) else []

    if not isinstance(cm, dict) or not isinstance(gu, dict):
        raise HTTPException(status_code=400, detail="comparison_matrix and gaps_and_usps are required.")

    # Feed into existing strategic tool in its supported combined shape.
    base_payload = {
        "comparison_matrix": cm,
        "gaps_and_usps": gu,
        "cluster_assignment": ca,
        "extraction_warnings": ew,
    }

    return run_strategic_analysis(
        gaps_and_usps=base_payload,
        gaps_and_usps_path=None,
        evidences=evidences,
        provider=provider,
        user_root=user_root,
        work_root=work_root,
    )
