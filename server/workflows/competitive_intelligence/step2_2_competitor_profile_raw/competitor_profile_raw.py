from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from fastapi import HTTPException

from server.services.llm_brave import LlmBrave

from .models import (
    Step22CompanyInput,
    Step22CompanyRawProfile,
    Step22CompetitorProfileRawRequest,
    Step22CompetitorProfileRawResult,
    Step22RawSearchItem,
)


def _clean_text(value: str) -> str:
    return " ".join(str(value or "").split()).strip()


def _resolve_input_path(path: str, *, user_root: Path, work_root: Path) -> Path:
    raw = str(path or "").strip().lstrip("/")
    p = Path(raw)
    if not raw or p.is_absolute() or ".." in p.parts:
        raise HTTPException(status_code=400, detail=f"Invalid path: {path}")

    user_root = user_root.resolve()
    work_root = work_root.resolve()
    uploads_root = (user_root / "uploads").resolve()
    uploads_root.mkdir(parents=True, exist_ok=True)

    candidates: List[Path] = []
    parts = list(p.parts)
    if parts and parts[0] == "uploads":
        candidates.append((uploads_root / Path(*parts[1:])).resolve())
    elif parts and parts[0] == "work":
        candidates.append((work_root / Path(*parts[1:])).resolve())
    else:
        candidates.append((work_root / p).resolve())
        candidates.append((uploads_root / p).resolve())

    for candidate in candidates:
        if candidate.exists() and candidate.is_file() and (user_root in candidate.parents or candidate == user_root):
            return candidate
    raise HTTPException(status_code=404, detail=f"File not found: {path}")


def _load_json(
    *,
    inline_obj: Any,
    path: str | None,
    user_root: Path,
    work_root: Path,
) -> Any:
    if inline_obj is not None:
        return inline_obj
    resolved = _resolve_input_path(str(path or ""), user_root=user_root, work_root=work_root)
    try:
        return json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in path: {path}") from exc


def _parse_companies(raw: Any) -> List[Step22CompanyInput]:
    payload = raw
    if isinstance(raw, dict):
        if isinstance(raw.get("companies"), list):
            payload = raw.get("companies")
        elif isinstance(raw.get("competitors"), list):
            payload = raw.get("competitors")
    if not isinstance(payload, list):
        raise HTTPException(status_code=400, detail="companies payload must be a list or object with companies[].")

    out: List[Step22CompanyInput] = []
    for item in payload:
        if isinstance(item, Step22CompanyInput):
            out.append(item)
            continue

        row: Dict[str, Any] | None = None
        if isinstance(item, dict):
            row = item
        elif hasattr(item, "model_dump") and callable(getattr(item, "model_dump", None)):
            try:
                dumped = item.model_dump()
                if isinstance(dumped, dict):
                    row = dumped
            except Exception:
                row = None

        if not isinstance(row, dict):
            continue

        company = _clean_text(str(row.get("company") or row.get("name") or ""))
        website = _clean_text(str(row.get("website") or row.get("url") or ""))
        region = _clean_text(str(row.get("region") or row.get("location") or ""))
        if not company:
            continue
        out.append(Step22CompanyInput(company=company, website=website, region=region))
    return out


def _parse_trend_context(raw: Any) -> List[str]:
    if not isinstance(raw, dict):
        return []
    payload = raw.get("market_trends_summary") if isinstance(raw.get("market_trends_summary"), dict) else raw
    if not isinstance(payload, dict):
        return []
    rows = payload.get("summaries") if isinstance(payload.get("summaries"), list) else []
    out: List[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        summary = _clean_text(str(row.get("summary") or ""))
        if summary and summary not in out:
            out.append(summary)
        if len(out) >= 6:
            break
    return out


def _build_queries(*, company: str, website: str, region: str, trend_context: List[str]) -> List[Tuple[str, str, str]]:
    base_q1 = ", ".join([x for x in [company, website, region] if _clean_text(x)])
    base_no_url = ", ".join([x for x in [company, region] if _clean_text(x)])
    trend_hint = _clean_text("; ".join(trend_context[:3]))

    q1 = f"Unternehmensprofil und Zielgruppe von {base_q1}. Preissegment: Low-Budget, Mittelklasse oder Premium."
    q2 = f"Angebote und Aktionen von {base_no_url} auf Website und Social Media. {trend_hint}".strip()
    q3 = f"Bewertung auf Google und Reichweite in Social Media (Facebook, Instagram und co.) von {base_no_url}."
    q4 = f"Aktuelle Pressemitteilungen und Berichterstattung ueber {base_no_url}."
    return [
        ("query_1", "unternehmensprofil_zielgruppe", _clean_text(q1)),
        ("query_2", "angebote_aktionen", _clean_text(q2)),
        ("query_3", "bewertungen_reichweite_social", _clean_text(q3)),
        ("query_4", "presse_berichterstattung", _clean_text(q4)),
    ]


def _run_brave_query(
    *,
    llm: LlmBrave,
    query: str,
    req: Step22CompetitorProfileRawRequest,
) -> Tuple[str, Dict[str, Any], str]:
    raw_response: Dict[str, Any] = {}
    raw_text = ""
    warning = ""
    try:
        # Important: plain user query only, no system prompt.
        raw_response = llm.chat_completions(
            messages=[{"role": "user", "content": query}],
            model="brave",
            stream=req.brave_stream,
            language=req.brave_language,
            country=req.brave_country,
            enable_entities=req.brave_enable_entities,
            enable_citations=req.brave_enable_citations,
            enable_research=req.brave_enable_research,
            timeout_s=req.timeout_s,
        )
        raw_text = llm.extract_text(raw_response) or ""
        if not raw_text:
            warning = "brave_empty_response"
    except Exception as exc:
        warning = f"brave_error: {exc}"
    return raw_text, raw_response if isinstance(raw_response, dict) else {}, warning


def run_step_2_2_competitor_profile_raw(
    *,
    req: Step22CompetitorProfileRawRequest,
    user_root: Path,
    work_root: Path,
) -> Step22CompetitorProfileRawResult:
    warnings: List[str] = []

    companies_raw = _load_json(
        inline_obj=req.companies,
        path=req.companies_path,
        user_root=user_root,
        work_root=work_root,
    )
    companies = _parse_companies(companies_raw)
    if not companies:
        warnings.append("companies_empty")
        return Step22CompetitorProfileRawResult(
            provider=str(req.provider or "brave").strip().lower() or "brave",
            companies=[],
            trend_context=[],
            extraction_warnings=warnings,
        )

    trend_context: List[str] = []
    if req.market_trends_summary is not None or (req.market_trends_summary_path or "").strip():
        trend_raw = _load_json(
            inline_obj=req.market_trends_summary,
            path=req.market_trends_summary_path,
            user_root=user_root,
            work_root=work_root,
        )
        trend_context = _parse_trend_context(trend_raw)

    llm = LlmBrave()
    if not llm.enabled():
        warnings.append("brave_not_configured")
        return Step22CompetitorProfileRawResult(
            provider=str(req.provider or "brave").strip().lower() or "brave",
            companies=[],
            trend_context=trend_context,
            extraction_warnings=warnings,
        )

    out: List[Step22CompanyRawProfile] = []
    for comp in companies[: req.max_companies]:
        queries = _build_queries(
            company=comp.company,
            website=comp.website,
            region=comp.region,
            trend_context=trend_context,
        )
        raw_searches: List[Step22RawSearchItem] = []
        for query_id, topic, query in queries:
            raw_text, raw_response, warning = _run_brave_query(llm=llm, query=query, req=req)
            if warning:
                warnings.append(f"{comp.company}:{query_id}:{warning}")
            raw_searches.append(
                Step22RawSearchItem(
                    query_id=query_id,
                    topic=topic,
                    query=query,
                    raw_text=raw_text,
                    raw_response=raw_response,
                    warning=warning,
                )
            )

        out.append(
            Step22CompanyRawProfile(
                company=comp.company,
                website=comp.website,
                region=comp.region,
                raw_searches=raw_searches,
            )
        )

    return Step22CompetitorProfileRawResult(
        provider=str(req.provider or "brave").strip().lower() or "brave",
        companies=out,
        trend_context=trend_context,
        extraction_warnings=list(dict.fromkeys([w for w in warnings if _clean_text(w)])),
    )
