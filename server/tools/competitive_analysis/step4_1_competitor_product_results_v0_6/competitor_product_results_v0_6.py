from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from server.services.llm_ionos import IonosLLM
from server.services.llm_openai import LlmOpenai
from server.services.llm_perplexity import LlmPerplexity
from server.tools.competitive_analysis.backup.competitor_identification import (
    _clean_text,
    _load_json_obj,
)

from .models import CompetitorProductResultsV06, ProductCompetitorSlim


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
    s = t.find("{")
    e = t.rfind("}")
    if s >= 0 and e > s:
        try:
            obj = json.loads(t[s : e + 1])
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _heuristic_generic(name: str, url: str) -> bool:
    n = _clean_text(name).lower()
    u = _clean_text(url).lower()
    generic_terms = (
        "home",
        "übersicht",
        "overview",
        "startseite",
        "kategorie",
        "category",
        "produkte",
        "products",
        "familie",
        "family",
    )
    if any(t in n for t in generic_terms):
        return True
    if re.search(r"/(home|overview|kategorie|category|products?)/?$", u):
        return True
    return False


def _llm_filter_candidate(
    *,
    provider: str,
    product_name: str,
    manufacturer: str,
    url: str,
    url_type: str,
    relevance_score: float,
    similarity_score: float,
    warnings: List[str],
) -> Dict[str, Any]:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "is_german_or_english": {"type": "boolean"},
            "is_generic": {"type": "boolean"},
            "keep": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["is_german_or_english", "is_generic", "keep", "reason"],
    }
    system = (
        "Du klassifizierst, ob ein Suchtreffer ein konkretes Produkt ist. "
        "Regeln: "
        "1) Behalte nur Produktnamen in deutscher oder englischer Sprache. "
        "2) Verwerfe generische Seiten-/Familien-/Kategoriebezeichnungen. "
        "3) keep=true nur bei konkretem Einzelprodukt. "
        "Antworte strikt als JSON nach Schema."
    )
    user = (
        f"product_name={product_name}\n"
        f"manufacturer={manufacturer}\n"
        f"url={url}\n"
        f"url_type={url_type}\n"
        f"relevance_score={relevance_score}\n"
        f"similarity_score={similarity_score}"
    )

    p = str(provider or "openai").strip().lower()
    if p not in {"openai", "perplexity", "ionos"}:
        p = "openai"

    if p in {"openai", "perplexity"}:
        c = LlmOpenai() if p == "openai" else LlmPerplexity()
        if not c.enabled():
            return {}
        try:
            resp = c._call(
                input_messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                text_format={"type": "json_schema", "name": "competitor_product_filter_v06", "schema": schema, "strict": False},
            )
            text = ""
            for item in resp.get("output", []):
                for cc in item.get("content", []):
                    if cc.get("type") == "output_text":
                        text += str(cc.get("text") or "")
            return _parse_json_strictish(text)
        except Exception as exc:
            warnings.append(f"{p} product filter failed: {exc}")
            return {}

    c2 = IonosLLM()
    if not c2.enabled():
        return {}
    try:
        comp = c2.chat_completions(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "competitor_product_filter_v06", "schema": schema, "strict": True},
            },
        )
        return _parse_json_strictish(c2.extract_text(comp))
    except Exception as exc:
        warnings.append(f"ionos product filter failed: {exc}")
        return {}


def build_competitor_product_results_v0_6(
    *,
    competitor_search_results: Optional[Dict[str, Any]],
    competitor_search_results_path: Optional[str],
    provider: str = "openai",
    top_n: int = 20,
    verbose_terminal: bool = False,
    user_root=None,
    work_root=None,
) -> CompetitorProductResultsV06:
    def _log(msg: str) -> None:
        if verbose_terminal:
            print(f"[competitor_product_results_v0_6] {msg}")

    payload = _load_json_obj(
        inline_obj=competitor_search_results,
        path=competitor_search_results_path,
        root_key="competitor_search_results",
        user_root=user_root,
        work_root=work_root,
    )

    warnings: List[str] = [str(w).strip() for w in (payload.get("extraction_warnings") or []) if str(w).strip()]
    generated_queries = [str(q).strip() for q in (payload.get("generated_queries") or []) if str(q).strip()]
    candidates = payload.get("competitors") if isinstance(payload.get("competitors"), list) else []
    candidates = [c for c in candidates if isinstance(c, dict)]

    sorted_candidates = sorted(
        candidates,
        key=lambda x: float(x.get("relevance_score") or 0.0),
        reverse=True,
    )
    target = max(1, int(top_n))

    out: List[ProductCompetitorSlim] = []
    _log(f"start candidates={len(sorted_candidates)} top_n={target} provider={provider}")
    for idx, c in enumerate(sorted_candidates, start=1):
        if len(out) >= target:
            break
        product_name = _clean_text(str(c.get("product_name") or ""))
        manufacturer = _clean_text(str(c.get("manufacturer") or ""))
        url = _clean_text(str(c.get("url") or ""))
        url_type = _clean_text(str(c.get("url_type") or "unknown")) or "unknown"
        relevance = float(c.get("relevance_score") or 0.0)
        similarity = float(c.get("similarity_score") or 0.0)
        if not product_name or not url:
            continue

        llm_eval = _llm_filter_candidate(
            provider=provider,
            product_name=product_name,
            manufacturer=manufacturer,
            url=url,
            url_type=url_type,
            relevance_score=relevance,
            similarity_score=similarity,
            warnings=warnings,
        )
        if llm_eval:
            is_lang = bool(llm_eval.get("is_german_or_english"))
            is_generic = bool(llm_eval.get("is_generic"))
            keep = bool(llm_eval.get("keep"))
            reason = _clean_text(str(llm_eval.get("reason") or ""))
        else:
            # fallback only when LLM unavailable/failed
            is_lang = True
            is_generic = _heuristic_generic(product_name, url)
            keep = not is_generic
            reason = "fallback_heuristic"

        _log(
            f"candidate {idx}: keep={keep} lang={is_lang} generic={is_generic} "
            f"relevance={relevance:.4f} name={product_name}"
        )
        if reason:
            _log(f"reason={reason}")

        if (not is_lang) or is_generic or (not keep):
            continue

        out.append(
            ProductCompetitorSlim(
                product_name=product_name,
                manufacturer=manufacturer,
                url=url,
                url_type=url_type,
                relevance_score=round(relevance, 4),
                similarity_score=round(similarity, 4),
            )
        )

    _log(f"done selected={len(out)} warnings={len(warnings)}")
    return CompetitorProductResultsV06(
        schema_version="1.0",
        provider=str(provider or "openai").strip().lower(),
        generated_queries=generated_queries,
        competitors=out,
        extraction_warnings=list(dict.fromkeys([w for w in warnings if _clean_text(w)])),
    )

