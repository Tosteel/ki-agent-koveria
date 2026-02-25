from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

from server.core.settings import Settings
from server.tools.rag_knowledgebase.service import RagService


def normalize_offer_id(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return "offerflow_default"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return safe or "offerflow_default"


def offer_dir(s: Settings, user_id: str, offer_id: str) -> Path:
    root = s.user_work_dir(user_id).resolve() / "offerflow" / normalize_offer_id(offer_id)
    root.mkdir(parents=True, exist_ok=True)
    return root


def write_json(path: Path, data: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def metadata_payload(
    *,
    step: int,
    trade: str = "",
    region: str = "",
    project_type: str = "",
    scope_tags: Optional[Sequence[str]] = None,
    size: str = "",
    outcome: str = "",
) -> Dict[str, Any]:
    return {
        "step": int(step),
        "trade": str(trade or ""),
        "region": str(region or ""),
        "project_type": str(project_type or ""),
        "scope_tags": [str(x).strip() for x in (scope_tags or []) if str(x).strip()],
        "size": str(size or ""),
        "outcome": str(outcome or ""),
    }


def _to_upload_pairs(payload: Dict[str, Any]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for key, value in payload.items():
        if value is None:
            pairs.append((str(key), ""))
            continue
        if isinstance(value, (dict, list)):
            pairs.append((str(key), json.dumps(value, ensure_ascii=False)))
            continue
        pairs.append((str(key), str(value)))
    return pairs


def safe_rag_query(
    service: RagService,
    *,
    query: str,
    classification: str,
    top_k: int,
) -> Dict[str, Any]:
    try:
        data = service.query(query=query, top_k=top_k, classification=classification)
        return {"ok": True, "data": data, "error": ""}
    except Exception as exc:
        return {"ok": False, "data": {"hits": []}, "error": str(exc)}


def safe_rag_upload(
    service: RagService,
    *,
    classification: str,
    local_path: str,
    custom_metadata: Dict[str, Any],
    extra_fields: Optional[Dict[str, Any]] = None,
    files: Optional[Iterable[tuple[str, tuple[str, bytes, str]]]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "classification": classification,
        "local_path": local_path,
        "custom_metadata": custom_metadata,
    }
    if extra_fields:
        payload.update(extra_fields)

    try:
        data = service.upload(
            data=_to_upload_pairs(payload),
            files=list(files or []),
        )
        return {"ok": True, "data": data, "error": ""}
    except Exception as exc:
        return {"ok": False, "data": {}, "error": str(exc)}
