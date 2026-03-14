from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from fastapi import HTTPException

from server.services.llm_ionos import IonosLLM
from server.services.llm_openai import LlmOpenai

from .models import (
    Step13MarketTrendsSummaryRequest,
    Step13MarketTrendsSummaryResult,
    Step13TrendSummaryItem,
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


def _load_payload(
    *,
    inline_obj: dict | None,
    path: str | None,
    user_root: Path,
    work_root: Path,
) -> dict:
    if isinstance(inline_obj, dict) and inline_obj:
        payload = inline_obj
    else:
        resolved = _resolve_input_path(str(path or ""), user_root=user_root, work_root=work_root)
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON in path: {path}") from exc
    if isinstance(payload.get("market_trends_structured"), dict):
        payload = payload["market_trends_structured"]
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid payload: expected object.")
    return payload


def _tokens(value: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[A-Za-z0-9ÄÖÜäöüß]{3,}", str(value or ""))}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def _extract_openai_output_text(resp: dict) -> str:
    out = ""
    for item in resp.get("output", []):
        for c in item.get("content", []):
            if c.get("type") == "output_text":
                out += str(c.get("text") or "")
    return _clean_text(out)


def _parse_json_strictish(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    t = str(text).strip()
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


def _fallback_summary(statements: List[str]) -> str:
    if not statements:
        return ""
    ranked = sorted(statements, key=lambda x: len(_tokens(x)), reverse=True)
    return _clean_text(ranked[0])


def _llm_text_summarize(*, provider: str, statements: List[str], warnings: List[str]) -> str:
    if not statements:
        return ""
    p = str(provider or "ionos").strip().lower()
    if p not in {"ionos", "openai"}:
        p = "ionos"

    user_prompt = (
        "Fasse die folgenden Trendaussagen zu einer einzigen, praezisen Kernaussage auf Deutsch zusammen.\n"
        "Regeln: Genau ein Satz, keine Einleitung, keine erfundenen Fakten.\n"
        f"Aussagen:\n- " + "\n- ".join(statements[:8])
    )
    system_prompt = "Du komprimierst mehrere Aussagen zu einer belastbaren Trend-Kernaussage."

    try:
        if p == "openai":
            client = LlmOpenai()
            if not client.enabled():
                warnings.append("openai_not_configured")
                return _fallback_summary(statements)
            resp = client._call(
                input_messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            out = _extract_openai_output_text(resp)
            return out or _fallback_summary(statements)

        client_i = IonosLLM()
        if not client_i.enabled():
            warnings.append("ionos_not_configured")
            return _fallback_summary(statements)
        comp = client_i.chat_completions(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        out = _clean_text(client_i.extract_text(comp))
        return out or _fallback_summary(statements)
    except Exception as exc:
        warnings.append(f"summary_llm_failed({p}): {exc}")
        return _fallback_summary(statements)


def _normalize_provider(provider: str, warnings: List[str]) -> str:
    p = str(provider or "ionos").strip().lower()
    if p not in {"ionos", "openai"}:
        warnings.append(f"unsupported_provider:{p};fallback_to_ionos")
        return "ionos"
    return p


def _extract_entries(sources: List[Any]) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for source in sources:
        if not isinstance(source, dict):
            continue
        url = _clean_text(str(source.get("url") or ""))
        if not url:
            continue
        statements: List[str] = []
        raw_list = source.get("kernaussage")
        if isinstance(raw_list, list):
            statements.extend([_clean_text(str(x or "")) for x in raw_list])
        kmb = source.get("kernaussage_mit_bildern")
        if isinstance(kmb, list):
            for item in kmb:
                if isinstance(item, dict):
                    statements.append(_clean_text(str(item.get("aussage") or "")))
        for statement in statements:
            if not statement:
                continue
            pair = (statement.lower(), url)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            entries.append({"statement": statement, "url": url, "tokens": _tokens(statement)})
    return entries


def _build_heuristic_summaries(
    *,
    entries: List[Dict[str, object]],
    provider: str,
    req: Step13MarketTrendsSummaryRequest,
    warnings: List[str],
) -> List[Step13TrendSummaryItem]:
    clusters: List[List[Dict[str, object]]] = []
    threshold = float(req.similarity_threshold)
    for entry in entries:
        best_idx = -1
        best_score = 0.0
        entry_tokens = entry["tokens"] if isinstance(entry.get("tokens"), set) else set()
        for idx, cluster in enumerate(clusters):
            cluster_tokens: set[str] = set()
            for c in cluster:
                c_tokens = c["tokens"] if isinstance(c.get("tokens"), set) else set()
                cluster_tokens |= c_tokens
            score = _jaccard(entry_tokens, cluster_tokens)
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx >= 0 and best_score >= threshold:
            clusters[best_idx].append(entry)
        else:
            clusters.append([entry])

    out: List[Step13TrendSummaryItem] = []
    for cluster in clusters:
        statements: List[str] = []
        urls: List[str] = []
        for c in cluster:
            s = _clean_text(str(c.get("statement") or ""))
            u = _clean_text(str(c.get("url") or ""))
            if s and s not in statements:
                statements.append(s)
            if u and u not in urls:
                urls.append(u)
        out.append(
            Step13TrendSummaryItem(
                summary=_llm_text_summarize(provider=provider, statements=statements, warnings=warnings),
                source_urls=urls,
                source_count=len(urls),
                evidence_points=statements[: req.max_evidence_per_item],
            )
        )

    out.sort(key=lambda x: (x.source_count, len(x.evidence_points)), reverse=True)
    return out[: req.max_summary_items]


def _llm_cluster_summaries(
    *,
    provider: str,
    entries: List[Dict[str, object]],
    req: Step13MarketTrendsSummaryRequest,
    warnings: List[str],
) -> List[Step13TrendSummaryItem]:
    indexed = []
    for i, e in enumerate(entries, start=1):
        indexed.append(
            {
                "idx": i,
                "statement": _clean_text(str(e.get("statement") or "")),
                "url": _clean_text(str(e.get("url") or "")),
            }
        )
    if not indexed:
        return []

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "summaries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "summary": {"type": "string"},
                        "statement_indices": {"type": "array", "items": {"type": "integer"}},
                    },
                    "required": ["summary", "statement_indices"],
                },
            }
        },
        "required": ["summaries"],
    }
    system_prompt = (
        "Du gruppierst inhaltlich aehnliche Trendaussagen ueber mehrere Quellen hinweg. "
        "Aehnliche Aussagen aus unterschiedlichen URLs muessen in denselben Cluster. "
        "Liefere nur JSON gemaess Schema."
    )
    user_prompt = (
        "Cluster diese Aussagen in wenige, konsolidierte Themen. "
        "Nutze statement_indices fuer die Zuordnung und liefere pro Cluster eine summary.\n"
        f"Daten:\n{json.dumps(indexed, ensure_ascii=False)}"
    )

    parsed: Dict[str, Any] = {}
    try:
        if provider == "openai":
            client = LlmOpenai()
            if not client.enabled():
                warnings.append("openai_not_configured_for_llm_clustering")
                return []
            resp = client._call(
                input_messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                text_format={
                    "type": "json_schema",
                    "name": "market_trends_llm_clustering",
                    "schema": schema,
                    "strict": False,
                },
            )
            parsed = _parse_json_strictish(_extract_openai_output_text(resp))
        else:
            client_i = IonosLLM()
            if not client_i.enabled():
                warnings.append("ionos_not_configured_for_llm_clustering")
                return []
            comp = client_i.chat_completions(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "market_trends_llm_clustering",
                        "schema": schema,
                        "strict": False,
                    },
                },
            )
            parsed = _parse_json_strictish(client_i.extract_text(comp))
    except Exception as exc:
        warnings.append(f"llm_clustering_failed({provider}): {exc}")
        return []

    rows = parsed.get("summaries") if isinstance(parsed.get("summaries"), list) else []
    if not rows:
        return []

    by_idx = {x["idx"]: x for x in indexed}
    out: List[Step13TrendSummaryItem] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        summary = _clean_text(str(row.get("summary") or ""))
        idxs = row.get("statement_indices") if isinstance(row.get("statement_indices"), list) else []
        statements: List[str] = []
        urls: List[str] = []
        for idx in idxs:
            try:
                i = int(idx)
            except Exception:
                continue
            item = by_idx.get(i)
            if not item:
                continue
            s = _clean_text(str(item.get("statement") or ""))
            u = _clean_text(str(item.get("url") or ""))
            if s and s not in statements:
                statements.append(s)
            if u and u not in urls:
                urls.append(u)
        if not statements:
            continue
        if not summary:
            summary = _fallback_summary(statements)
        out.append(
            Step13TrendSummaryItem(
                summary=summary,
                source_urls=urls,
                source_count=len(urls),
                evidence_points=statements[: req.max_evidence_per_item],
            )
        )

    out.sort(key=lambda x: (x.source_count, len(x.evidence_points)), reverse=True)
    return out[: req.max_summary_items]


def run_step_1_3_market_trends_summary(
    *,
    req: Step13MarketTrendsSummaryRequest,
    user_root: Path,
    work_root: Path,
) -> Step13MarketTrendsSummaryResult:
    warnings: List[str] = []
    payload = _load_payload(
        inline_obj=req.market_trends_structured,
        path=req.market_trends_structured_path,
        user_root=user_root,
        work_root=work_root,
    )
    sources = payload.get("sources") if isinstance(payload.get("sources"), list) else []
    entries = _extract_entries(sources)

    if not entries:
        warnings.append("no_kernaussagen_found")
        return Step13MarketTrendsSummaryResult(
            provider=str(req.provider or "ionos").strip().lower() or "ionos",
            summaries=[],
            extraction_warnings=warnings,
        )

    provider = _normalize_provider(req.provider, warnings)

    # LLM-first clustering.
    llm_summaries = _llm_cluster_summaries(
        provider=provider,
        entries=entries,
        req=req,
        warnings=warnings,
    )
    if llm_summaries:
        warnings.append("llm_primary_clustering_used")
        summaries = llm_summaries
    else:
        warnings.append("llm_clustering_unavailable_used_heuristic_fallback")
        summaries = _build_heuristic_summaries(
            entries=entries,
            provider=provider,
            req=req,
            warnings=warnings,
        )

    return Step13MarketTrendsSummaryResult(
        provider=provider,
        summaries=summaries,
        extraction_warnings=list(dict.fromkeys([w for w in warnings if _clean_text(w)])),
    )
