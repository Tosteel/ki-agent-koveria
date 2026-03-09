from __future__ import annotations

from pathlib import Path
from typing import List

from server.tools.startup_matchup.common import (
    brave_answers_text,
    clean_text,
    dedupe_startup_hits,
    load_json_obj,
    parse_startup_hits,
    safe_list_str,
)

from .models import QueryLog, StartupCandidatesRaw, StartupMatchupStep4Request, StartupSearchResult


def _fallback_queries(gap_analysis: dict, max_queries: int) -> List[str]:
    fields = safe_list_str(gap_analysis.get("startup_search_fields"))
    out: List[str] = []
    for field in fields:
        out.append(f"{field} startup companies")
        if len(out) >= max_queries:
            break
    if not out:
        out = ["innovative B2B startup companies"]
    return out[:max_queries]


def _build_brave_query(query: str, per_query_results: int) -> str:
    q = clean_text(query)
    return (
        f"Find up to {per_query_results} relevant startup companies for: {q}. "
        "Return structured JSON with key search_results and items having snippet, url, source. "
        "Do not invent startup names; leave startup_name empty if unknown. "
        "Include only publicly reachable company websites."
    )


def run_step_4(*, req: StartupMatchupStep4Request, user_root: Path, work_root: Path) -> StartupCandidatesRaw:
    warnings: List[str] = []
    gap = load_json_obj(
        inline_obj=req.gap_analysis,
        path=req.gap_analysis_path,
        root_key="gap_analysis",
        user_root=user_root,
        work_root=work_root,
    )

    queries = safe_list_str(gap.get("startup_search_queries"))
    if not queries:
        queries = _fallback_queries(gap, req.max_queries)
        warnings.append("No startup_search_queries found; generated fallback queries from startup_search_fields.")

    queries = queries[: req.max_queries]

    rows: List[dict] = []
    logs: List[QueryLog] = []
    for query in queries:
        brave_query = _build_brave_query(query, req.per_query_results)
        raw_text = brave_answers_text(
            query=brave_query,
            enable_research=req.brave_enable_research,
            stream=req.brave_stream,
            language=req.brave_language,
            country=req.brave_country,
            warnings=warnings,
        )
        logs.append(QueryLog(query=query, result_excerpt=clean_text(raw_text)))

        parsed_rows = parse_startup_hits(raw_text, source_query=query, max_hits=req.per_query_results)
        if not parsed_rows:
            warnings.append(f"No startup hits extracted for query: {query}")
            continue
        rows.extend(parsed_rows)

    deduped = dedupe_startup_hits(rows, max_items=500)
    search_results = [StartupSearchResult(**r) for r in deduped]

    if not search_results:
        warnings.append("Startup search returned no candidates.")

    return StartupCandidatesRaw(
        queries=queries,
        search_results=search_results,
        query_logs=logs,
        extraction_warnings=warnings,
    )
