from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from fastapi import HTTPException

from server.services.llm_ionos import IonosLLM
from server.services.llm_openai import LlmOpenai
from server.services.llm_perplexity import LlmPerplexity
from server.tools.competitive_analysis.backup.competitor_identification import CompetitorCandidate, CompetitorList

from .models import CompetitorIdentificationQualityReport


_GENERIC_LISTING_HINTS = (
    "comparison chart",
    "price comparison",
    "preisvergleich",
    "bestenliste",
    "top 10",
    "marktuebersicht",
    "marktübersicht",
    "kategorie",
    "category",
    "/category/",
    "/categories/",
    "/collections/",
    "/collection/",
    "/tag/",
    "/blog/",
    "/news/",
    "/forum/",
    "/thread/",
    "wiki",
)

_GENERIC_NAME_HINTS = (
    "saug- und wischroboter",
    "saugroboter",
    "wischroboter",
    "kaffeevollautomat",
    "wechselrichter",
    "rechargeable battery",
    "lithium-ion rechargeable battery",
)


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


def _extract_any_output_text(resp: Dict[str, Any]) -> str:
    txt = _openai_extract_output_text(resp)
    if txt:
        return txt
    try:
        return LlmPerplexity._extract_text(resp)  # type: ignore[attr-defined]
    except Exception:
        return ""


def _resolve_input_path(path: str, user_root: Path, work_root: Path) -> Path:
    raw = str(path or "").strip().lstrip("/")
    p = Path(raw)
    if not raw or p.is_absolute() or ".." in p.parts:
        raise HTTPException(status_code=400, detail=f"Invalid path: {path}")

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

    for c in candidates:
        if c.exists() and c.is_file() and (user_root in c.parents or c == user_root):
            return c
    raise HTTPException(status_code=404, detail=f"File not found: {path}")


def _load_json_obj(
    *,
    inline_obj: Optional[Dict[str, Any]],
    path: Optional[str],
    user_root: Path,
    work_root: Path,
) -> Dict[str, Any]:
    payload: Dict[str, Any]
    if isinstance(inline_obj, dict) and inline_obj:
        payload = inline_obj
    else:
        p = _resolve_input_path(str(path or ""), user_root=user_root.resolve(), work_root=work_root.resolve())
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON in path: {path}") from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    return payload


def _load_competitor_list(
    *,
    competitor_list: Optional[Dict[str, Any]],
    competitor_list_path: Optional[str],
    user_root: Path,
    work_root: Path,
) -> CompetitorList:
    payload = _load_json_obj(
        inline_obj=competitor_list,
        path=competitor_list_path,
        user_root=user_root,
        work_root=work_root,
    )

    if "competitor_list" in payload and isinstance(payload.get("competitor_list"), dict):
        payload = payload["competitor_list"]

    try:
        return CompetitorList(**payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid competitor_list payload: {exc}") from exc


def _extract_product_identity(
    *,
    product_profile: Optional[Dict[str, Any]],
    product_profile_path: Optional[str],
    user_root: Path,
    work_root: Path,
) -> Tuple[str, str, str]:
    if not product_profile and not (product_profile_path or "").strip():
        return "", "", ""

    payload = _load_json_obj(
        inline_obj=product_profile,
        path=product_profile_path,
        user_root=user_root,
        work_root=work_root,
    )
    if "product_profile" in payload and isinstance(payload.get("product_profile"), dict):
        payload = payload["product_profile"]

    meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    product_name = str(meta.get("product_name") or payload.get("product_name") or "").strip().lower()
    manufacturer = str(meta.get("manufacturer") or payload.get("manufacturer") or "").strip().lower()
    product_category = str(payload.get("product_category") or "").strip().lower()
    return product_name, manufacturer, product_category


def _model_signal(text: str) -> bool:
    # token with letters+digits usually signals concrete model identifiers
    for tok in re.findall(r"[A-Za-z0-9.-]{3,}", str(text or "")):
        has_alpha = any(ch.isalpha() for ch in tok)
        has_digit = any(ch.isdigit() for ch in tok)
        if has_alpha and has_digit:
            return True
    return False


def _url_domain(url: str) -> str:
    try:
        d = urlparse(str(url or "")).netloc.lower().replace("www.", "")
    except Exception:
        d = ""
    return d


def _norm_name(name: str) -> str:
    t = re.sub(r"[^a-z0-9]+", " ", str(name or "").lower()).strip()
    return re.sub(r"\s+", " ", t)


def _is_generic_listing_candidate(c: CompetitorCandidate) -> bool:
    hay = " ".join([
        str(c.name or ""),
        str(c.snippet or ""),
        str(c.url or ""),
    ]).lower()
    has_listing_hint = any(h in hay for h in _GENERIC_LISTING_HINTS)
    has_model = _model_signal(str(c.name or "") + " " + str(c.snippet or ""))
    return has_listing_hint and not has_model


def _is_weak_unknown_candidate(c: CompetitorCandidate, min_relevance_score: float) -> bool:
    if str(c.source_type or "unknown").lower() != "unknown":
        return False
    if c.matched_dimensions:
        return False
    return float(c.relevance_score or 0.0) < float(min_relevance_score)


def _is_manufacturer_node_without_model_signal(c: CompetitorCandidate) -> bool:
    if str(c.source_type or "").lower() != "manufacturer_page":
        return False
    return not _model_signal(str(c.name or "") + " " + str(c.snippet or ""))


def _is_self_or_variant(c: CompetitorCandidate, product_name: str, manufacturer: str) -> bool:
    if not product_name and not manufacturer:
        return False
    n = _norm_name(c.name)
    if product_name and product_name in n:
        return True
    if manufacturer and manufacturer in n and product_name:
        # conservative: avoid dropping pure brand competitors if product isn't contained
        return product_name.split(" ")[0] in n
    return False


def _tokenize_category(category: str) -> List[str]:
    toks = re.findall(r"[a-z0-9]+", str(category or "").lower())
    return [t for t in toks if len(t) >= 5]


def _is_generic_name_without_model_signal(c: CompetitorCandidate, product_category: str) -> bool:
    name = str(c.name or "").strip().lower()
    if not name:
        return True
    if _model_signal(name):
        return False

    has_generic_hint = any(h in name for h in _GENERIC_NAME_HINTS)
    cat_tokens = _tokenize_category(product_category)
    has_category_overlap = bool(cat_tokens) and any(t in name for t in cat_tokens)

    # Strenger nur dann droppen, wenn klar generischer Name vorliegt
    # (Kategoriebegriff ohne Modellsignal).
    return has_generic_hint or has_category_overlap


def _dedupe_by_name_domain(candidates: List[CompetitorCandidate]) -> Tuple[List[CompetitorCandidate], int]:
    best: Dict[Tuple[str, str], CompetitorCandidate] = {}
    removed = 0
    for c in candidates:
        key = (_norm_name(c.name), _url_domain(c.url))
        prev = best.get(key)
        if prev is None:
            best[key] = c
            continue
        prev_score = float(prev.relevance_score or 0.0) + float(prev.similarity_score or 0.0)
        cur_score = float(c.relevance_score or 0.0) + float(c.similarity_score or 0.0)
        if cur_score > prev_score:
            best[key] = c
        removed += 1
    return list(best.values()), removed


def _llm_candidate_judge_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "is_competitor_product": {"type": "boolean"},
            "confidence": {"type": "number"},
            "reason_code": {
                "type": "string",
                "enum": [
                    "valid_product_competitor",
                    "generic_listing",
                    "category_or_brand_node",
                    "self_or_variant",
                    "non_product_content",
                    "insufficient_signal",
                    "other",
                ],
            },
            "short_reason": {"type": "string"},
        },
        "required": ["is_competitor_product", "confidence", "reason_code", "short_reason"],
    }


def _llm_rejection_reason_code(
    *,
    is_competitor: bool,
    reason_code: str,
) -> str:
    if is_competitor:
        return ""
    rc = str(reason_code or "").strip().lower()
    allowed_negative = {
        "generic_listing",
        "category_or_brand_node",
        "self_or_variant",
        "non_product_content",
        "insufficient_signal",
        "other",
    }
    if rc in allowed_negative:
        return rc
    # Schützt gegen inkonsistente LLM-Antworten wie:
    # is_competitor_product=false + reason_code=valid_product_competitor
    return "inconsistent_label"


def _llm_validate_candidate(
    *,
    provider: str,
    candidate: CompetitorCandidate,
    product_name: str,
    manufacturer: str,
    product_category: str,
    warnings: List[str],
) -> Dict[str, Any]:
    p = str(provider or "perplexity").strip().lower()
    if p not in {"openai", "perplexity", "ionos"}:
        p = "perplexity"

    schema = _llm_candidate_judge_schema()
    system = (
        "Du prüfst Kandidaten einer Wettbewerberliste. "
        "Entscheide, ob der Kandidat ein konkretes Wettbewerbsprodukt (oder klare Produktserie) ist. "
        "Nicht als Wettbewerbsprodukt zählen: reine Kategorie-/Brand-Seiten, Preisvergleichslisten, Foren/News, "
        "oder irrelevante Produkte. Nutze primär name, url und snippet."
    )
    user = json.dumps(
        {
            "target_product_name": product_name,
            "target_manufacturer": manufacturer,
            "target_category": product_category,
            "candidate": {
                "name": candidate.name,
                "url": candidate.url,
                "source_type": candidate.source_type,
                "snippet": candidate.snippet,
                "source_query": candidate.source_query,
                "matched_dimensions": list(candidate.matched_dimensions or []),
            },
        },
        ensure_ascii=False,
    )

    if p in {"openai", "perplexity"}:
        client = LlmOpenai() if p == "openai" else LlmPerplexity()
        if not client.enabled():
            warnings.append(f"{p} not configured; skipped LLM candidate validation.")
            return {}
        rf = {
            "type": "json_schema",
            "name": "competitor_candidate_validation",
            "schema": schema,
            "strict": False,
        }
        try:
            resp = client._call(  # type: ignore[attr-defined]
                input_messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                text_format=rf,
            )
            return _parse_json_strictish(_extract_any_output_text(resp))
        except Exception as exc:
            warnings.append(f"{p} LLM candidate validation failed for '{candidate.name}': {exc}")
            return {}

    c2 = IonosLLM()
    if not c2.enabled():
        warnings.append("IONOS not configured; skipped LLM candidate validation.")
        return {}
    try:
        completion = c2.chat_completions(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "competitor_candidate_validation",
                    "schema": schema,
                    "strict": True,
                },
            },
        )
        return _parse_json_strictish(c2.extract_text(completion))
    except Exception as exc:
        warnings.append(f"IONOS LLM candidate validation failed for '{candidate.name}': {exc}")
        return {}


def _llm_name_normalization_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "normalized_name": {"type": "string"},
            "changed": {"type": "boolean"},
            "confidence": {"type": "number"},
            "reason": {"type": "string"},
        },
        "required": ["normalized_name", "changed", "confidence", "reason"],
    }


def _llm_normalize_candidate_name(
    *,
    provider: str,
    candidate: CompetitorCandidate,
    product_category: str,
    warnings: List[str],
) -> Dict[str, Any]:
    p = str(provider or "perplexity").strip().lower()
    if p not in {"openai", "perplexity", "ionos"}:
        p = "perplexity"

    schema = _llm_name_normalization_schema()
    system = (
        "Du normalisierst Wettbewerbernamen auf die präzise Produkt-/Modellbezeichnung. "
        "Entferne generische Zusätze (z. B. Produkttyp, Marketing-Text, Satzteile nach 'mit/für/inkl.'). "
        "Behalte Hersteller + Modell. Keine Halluzinationen. Wenn unsicher, unverändert zurückgeben."
    )
    user = json.dumps(
        {
            "product_category": product_category,
            "candidate": {
                "name": candidate.name,
                "url": candidate.url,
                "snippet": candidate.snippet,
                "source_type": candidate.source_type,
            },
            "output_rule": "Return best concise product/model name.",
        },
        ensure_ascii=False,
    )

    if p in {"openai", "perplexity"}:
        client = LlmOpenai() if p == "openai" else LlmPerplexity()
        if not client.enabled():
            warnings.append(f"{p} not configured; skipped LLM name normalization.")
            return {}
        fmt = {
            "type": "json_schema",
            "name": "competitor_name_normalization",
            "schema": schema,
            "strict": False,
        }
        try:
            resp = client._call(  # type: ignore[attr-defined]
                input_messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                text_format=fmt,
            )
            return _parse_json_strictish(_extract_any_output_text(resp))
        except Exception as exc:
            warnings.append(f"{p} LLM name normalization failed for '{candidate.name}': {exc}")
            return {}

    c2 = IonosLLM()
    if not c2.enabled():
        warnings.append("IONOS not configured; skipped LLM name normalization.")
        return {}
    try:
        completion = c2.chat_completions(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "competitor_name_normalization",
                    "schema": schema,
                    "strict": True,
                },
            },
        )
        return _parse_json_strictish(c2.extract_text(completion))
    except Exception as exc:
        warnings.append(f"IONOS LLM name normalization failed for '{candidate.name}': {exc}")
        return {}


def run_competitor_identification_quality_gate(
    *,
    competitor_list: Optional[Dict[str, Any]],
    competitor_list_path: Optional[str],
    product_profile: Optional[Dict[str, Any]],
    product_profile_path: Optional[str],
    provider: str,
    min_relevance_score: float,
    drop_generic_listing_pages: bool,
    drop_weak_unknown_candidates: bool,
    drop_manufacturer_nodes_without_model_signal: bool,
    dedupe_by_name_and_domain: bool,
    enable_llm_snippet_validation: bool,
    llm_min_keep_confidence: float,
    max_llm_checks: int,
    user_root: Path,
    work_root: Path,
) -> Tuple[CompetitorList, CompetitorIdentificationQualityReport]:
    cl = _load_competitor_list(
        competitor_list=competitor_list,
        competitor_list_path=competitor_list_path,
        user_root=user_root,
        work_root=work_root,
    )
    product_name, manufacturer, product_category = _extract_product_identity(
        product_profile=product_profile,
        product_profile_path=product_profile_path,
        user_root=user_root,
        work_root=work_root,
    )

    drop_reasons: Dict[str, int] = {}
    kept: List[CompetitorCandidate] = []
    dropped_pre_dedupe = 0
    dropped_details: List[str] = []

    warnings = list(cl.extraction_warnings or [])
    llm_checked = 0
    llm_dropped = 0
    llm_name_checked = 0
    llm_name_changed = 0
    llm_name_max_checks = max(0, min(40, int(max_llm_checks)))

    for c in cl.competitors:
        reasons: List[str] = []

        if _is_self_or_variant(c, product_name=product_name, manufacturer=manufacturer):
            reasons.append("self_or_variant")

        if drop_generic_listing_pages and _is_generic_listing_candidate(c):
            reasons.append("generic_listing_page")

        if drop_weak_unknown_candidates and _is_weak_unknown_candidate(c, min_relevance_score=min_relevance_score):
            reasons.append("weak_unknown_candidate")

        if drop_manufacturer_nodes_without_model_signal and _is_manufacturer_node_without_model_signal(c):
            reasons.append("manufacturer_node_without_model_signal")

        if not reasons and llm_name_checked < llm_name_max_checks:
            llm_name_checked += 1
            normalized = _llm_normalize_candidate_name(
                provider=provider,
                candidate=c,
                product_category=product_category,
                warnings=warnings,
            )
            if normalized:
                raw_new_name = str(normalized.get("normalized_name") or "").strip()
                changed = bool(normalized.get("changed"))
                conf = float(normalized.get("confidence") or 0.0)
                if changed and raw_new_name and raw_new_name != c.name and conf >= 0.5:
                    original_name = c.name
                    c = c.model_copy(update={"name": raw_new_name})
                    llm_name_changed += 1
                    warnings.append(
                        f"Normalized competitor name via LLM: '{original_name}' -> '{raw_new_name}' (confidence={conf:.2f})"
                    )

        if not reasons and _is_generic_name_without_model_signal(c, product_category):
            reasons.append("generic_name_without_model_signal")

        llm_conf: float | None = None
        if not reasons and enable_llm_snippet_validation and llm_checked < max(0, int(max_llm_checks)):
            llm_checked += 1
            judged = _llm_validate_candidate(
                provider=provider,
                candidate=c,
                product_name=product_name,
                manufacturer=manufacturer,
                product_category=product_category,
                warnings=warnings,
            )
            if judged:
                is_competitor = bool(judged.get("is_competitor_product"))
                conf = float(judged.get("confidence") or 0.0)
                llm_conf = conf
                rc = str(judged.get("reason_code") or "other").strip().lower()
                if (not is_competitor) and conf >= float(llm_min_keep_confidence):
                    reject_code = _llm_rejection_reason_code(is_competitor=is_competitor, reason_code=rc)
                    reasons.append(f"llm_{reject_code or 'rejected'}")
                    llm_dropped += 1

        if reasons:
            dropped_pre_dedupe += 1
            for r in reasons:
                drop_reasons[r] = int(drop_reasons.get(r, 0)) + 1
            domain = _url_domain(c.url)
            conf_part = f", llm_conf={llm_conf:.2f}" if llm_conf is not None else ""
            dropped_details.append(
                f"Dropped candidate '{c.name}' ({domain or 'no-domain'}): reasons={','.join(reasons)}{conf_part}"
            )
            continue

        kept.append(c)

    deduped_removed = 0
    if dedupe_by_name_and_domain:
        kept, deduped_removed = _dedupe_by_name_domain(kept)

    warnings.append(f"Identification QG applied with provider={provider}.")
    if drop_reasons:
        summary = ", ".join([f"{k}={v}" for k, v in sorted(drop_reasons.items())])
        warnings.append(f"Identification QG dropped candidates: {summary}.")
    if deduped_removed > 0:
        warnings.append(f"Identification QG deduped {deduped_removed} duplicate candidates by (name, domain).")
    if enable_llm_snippet_validation:
        warnings.append(
            f"Identification QG LLM validation checked={llm_checked}, dropped={llm_dropped}, min_keep_confidence={llm_min_keep_confidence}."
        )
    warnings.append(
        f"Identification QG LLM name normalization checked={llm_name_checked}, changed={llm_name_changed}, max_checks={llm_name_max_checks}."
    )
    if dropped_details:
        max_detail_lines = 80
        warnings.extend(dropped_details[:max_detail_lines])
        if len(dropped_details) > max_detail_lines:
            warnings.append(
                f"Identification QG dropped detail lines truncated: {len(dropped_details) - max_detail_lines} more."
            )

    cleaned = CompetitorList(
        schema_version=cl.schema_version,
        provider=cl.provider,
        generated_queries=list(cl.generated_queries or []),
        min_competitors_target=cl.min_competitors_target,
        competitors=kept,
        extraction_warnings=warnings,
    )

    report = CompetitorIdentificationQualityReport(
        total_input_competitors=len(cl.competitors or []),
        total_output_competitors=len(kept),
        dropped_competitors=dropped_pre_dedupe,
        deduped_competitors=deduped_removed,
        drop_reasons=drop_reasons,
        notes=[
            "post_identification_quality_gate=enabled",
            f"provider={provider}",
            f"min_relevance_score={min_relevance_score}",
            f"enable_llm_snippet_validation={enable_llm_snippet_validation}",
            f"max_llm_checks={max_llm_checks}",
        ],
    )
    return cleaned, report
