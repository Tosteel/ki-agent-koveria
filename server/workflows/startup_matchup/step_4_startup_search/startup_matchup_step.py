from __future__ import annotations

from pathlib import Path
from typing import List

from server.workflows.startup_matchup.common import (
    brave_answers_text,
    clean_text,
    load_json_obj,
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


def _build_brave_query(query: str) -> str:
    return clean_text(query)


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
        brave_query = _build_brave_query(query)
        raw_text = brave_answers_text(
            query=brave_query,
            enable_research=req.brave_enable_research,
            stream=req.brave_stream,
            language=req.brave_language,
            country=req.brave_country,
            warnings=warnings,
        )
        logs.append(QueryLog(query=query, result_excerpt=raw_text))

        # Raw passthrough: keep Brave response 1:1 in step4 output without parsing.
        rows.append(
            {
                "snippet": raw_text,
                "url": "",
                "source": "brave_answers",
            }
        )

    search_results = [StartupSearchResult(**r) for r in rows]

    if not search_results:
        warnings.append("Startup search returned no candidates.")

    return StartupCandidatesRaw(
        queries=queries,
        search_results=search_results,
        query_logs=logs,
        extraction_warnings=warnings,
    )
