from __future__ import annotations
from pathlib import Path

from fastapi import HTTPException


def _safe_join(base: Path, rel_path: str) -> Path:
    rel = Path(rel_path.strip().lstrip("/"))
    if rel.is_absolute() or ".." in rel.parts:
        raise HTTPException(status_code=400, detail="Invalid path")
    p = (base / rel).resolve()
    base_resolved = base.resolve()
    if not str(p).startswith(str(base_resolved)):
        raise HTTPException(status_code=400, detail="Invalid path")
    return p


def read_text(work_dir: Path, rel_path: str, encoding: str = "utf-8", uploads_dir: Path | None = None) -> str:
    rel = Path(rel_path.strip().lstrip("/"))
    if rel.is_absolute() or ".." in rel.parts:
        raise HTTPException(status_code=400, detail="Invalid path")

    candidates: list[Path] = []
    parts = list(rel.parts)
    if parts and parts[0] == "uploads":
        if uploads_dir is None:
            raise HTTPException(status_code=404, detail="File not found")
        sub = Path(*parts[1:]) if len(parts) > 1 else Path("")
        candidates.append(_safe_join(uploads_dir, str(sub)))
    elif parts and parts[0] == "work":
        sub = Path(*parts[1:]) if len(parts) > 1 else Path("")
        candidates.append(_safe_join(work_dir, str(sub)))
    else:
        candidates.append(_safe_join(work_dir, str(rel)))
        if uploads_dir is not None:
            candidates.append(_safe_join(uploads_dir, str(rel)))

    p = next((c for c in candidates if c.exists() and c.is_file()), None)
    if p is None:
        raise HTTPException(status_code=404, detail="File not found")
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return p.read_text(encoding=encoding, errors="ignore")


def write_text(work_dir: Path, rel_path: str, content: str, encoding: str = "utf-8", overwrite: bool = True) -> int:
    p = _safe_join(work_dir, rel_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and not overwrite:
        raise HTTPException(status_code=409, detail="File exists and overwrite=false")
    data = content.encode(encoding, errors="ignore")
    p.write_bytes(data)
    return len(data)
