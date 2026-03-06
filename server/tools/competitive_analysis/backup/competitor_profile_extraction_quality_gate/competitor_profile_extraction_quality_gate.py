from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

from fastapi import HTTPException

from server.tools.competitive_analysis.backup.competitor_profile_extraction.competitor_profile_extraction import (
    _extract_prices,
    _fetch_page,
    _openai_web_search_urls,
    _perplexity_web_search_urls,
)
from server.tools.competitive_analysis.backup.competitor_profile_extraction.models import (
    CompetitorProfile,
    CompetitorProfiles,
    MappedFeature,
    PriceInfo,
    SourceEvidence,
)

_VENDOR_DOMAIN_HINTS = (
    "shop",
    "store",
    "kaufen",
    "tienda",
    "preis",
    "solar-pur",
    "fm-solar",
    "coenergia",
    "vpsolar",
    "amazon",
    "ebay",
)

_OFFICIAL_DOMAIN_NEGATIVE_HINTS = (
    "shop",
    "store",
    "market",
    "dealer",
    "reseller",
    "wiki",
    "linkedin",
    "facebook",
    "instagram",
    "youtube",
    "device.report",
    "scribd",
)

_GENERIC_NAME_TOKENS = {
    "series",
    "hybrid",
    "inverter",
    "wechselrichter",
    "residential",
    "single",
    "phase",
    "three",
    "smart",
    "energy",
    "plus",
}

_NOISY_RAWNAME_PATTERNS = (
    r"^the\s+.+supports",
    r"^schritte?\s+zum\s+install",
    r"^communication interface",
    r"^interfaces?$",
    r"^leading$",
    r"^italien$",
)

_NOISY_VALUE_HINTS = (
    "hybrid-inverter",
    "zugelassene batterien",
    "supports the following",
    "tn-s",
    "tn-c",
    "wifi+",
    "rs485",
    "can-bms",
)

_NORM_HINTS = ("iec ", "en ", "vde", "cei", "nrs", "g98", "g99", "une ", "ove")


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


def _load_competitor_profiles(
    *,
    competitor_profiles: Optional[Dict[str, Any]],
    competitor_profiles_path: Optional[str],
    user_root: Path,
    work_root: Path,
) -> CompetitorProfiles:
    payload: Dict[str, Any]
    if isinstance(competitor_profiles, dict) and competitor_profiles:
        payload = competitor_profiles
    else:
        p = _resolve_input_path(str(competitor_profiles_path or ""), user_root=user_root.resolve(), work_root=work_root.resolve())
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON in competitor_profiles_path: {competitor_profiles_path}") from exc

    if "competitor_profiles" in payload and isinstance(payload.get("competitor_profiles"), dict):
        payload = payload["competitor_profiles"]

    try:
        return CompetitorProfiles(**payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid competitor_profiles payload: {exc}") from exc


def _dedupe_str(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for v in values:
        s = str(v or "").strip()
        if not s:
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def _url_domain(url: str) -> str:
    try:
        d = urlparse(str(url or "")).netloc.lower()
    except Exception:
        d = ""
    return d.replace("www.", "")


def _domain_is_vendor(domain: str) -> bool:
    d = str(domain or "").lower()
    return any(h in d for h in _VENDOR_DOMAIN_HINTS)


def _brand_tokens(name: str) -> List[str]:
    toks = re.split(r"[^a-z0-9]+", str(name or "").lower())
    out: List[str] = []
    for t in toks:
        if len(t) < 3:
            continue
        if t in _GENERIC_NAME_TOKENS:
            continue
        out.append(t)
    return out[:4]


def _model_tokens(name: str) -> List[str]:
    raw = re.split(r"\s+", str(name or "").strip())
    out: List[str] = []
    for r in raw:
        rr = re.sub(r"[^A-Za-z0-9.-]", "", r)
        if not rr:
            continue
        if any(ch.isdigit() for ch in rr) or "-" in rr or "." in rr:
            out.append(re.sub(r"[^a-z0-9]", "", rr.lower()))
    return _dedupe_str(out)


def _is_official_domain_for_competitor(domain: str, competitor_name: str) -> bool:
    d = str(domain or "").lower().replace("www.", "")
    if not d:
        return False
    if any(h in d for h in _OFFICIAL_DOMAIN_NEGATIVE_HINTS):
        return False
    brands = _brand_tokens(competitor_name)
    if not brands:
        return False
    return any(b in d for b in brands)


def _model_match_score(text: str, competitor_name: str) -> int:
    t = str(text or "").lower()
    score = 0
    for tok in _model_tokens(competitor_name):
        if tok and tok in re.sub(r"[^a-z0-9]", "", t):
            score += 1
    return score


def _is_model_match(text: str, competitor_name: str, min_hits: int) -> bool:
    tokens = _model_tokens(competitor_name)
    if not tokens:
        return competitor_name.lower().split(" ")[0] in str(text or "").lower()
    need = min(min_hits, len(tokens))
    return _model_match_score(text, competitor_name) >= max(1, need)


def _build_price_search_query(name: str) -> str:
    return f'"{name}" price EUR buy shop store'


def _search_urls(provider: str, query: str, max_results: int, warnings: List[str], name: str) -> List[str]:
    p = str(provider or "perplexity").strip().lower()
    if p not in {"openai", "perplexity"}:
        p = "perplexity"

    if p == "openai":
        key = os.getenv("OPENAI_API_KEY", "").strip()
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
        if not key:
            warnings.append("OPENAI_API_KEY missing; quality gate web search disabled.")
            return []
        try:
            return _openai_web_search_urls(query, api_key=key, model=model, max_results=max_results)
        except Exception as exc:
            warnings.append(f"OpenAI web search failed for '{name}': {exc}")
            return []

    key = os.getenv("PERPLEXITY_API_KEY", "").strip()
    model = os.getenv("PERPLEXITY_MODEL", "sonar-pro").strip() or "sonar-pro"
    if not key:
        warnings.append("PERPLEXITY_API_KEY missing; quality gate web search disabled.")
        return []
    try:
        return _perplexity_web_search_urls(query, api_key=key, model=model, max_results=max_results)
    except Exception as exc:
        warnings.append(f"Perplexity web search failed for '{name}': {exc}")
        return []


def _find_official_url(name: str, urls: List[str], warnings: List[str], provider: str) -> str:
    for u in urls:
        if _is_official_domain_for_competitor(_url_domain(u), name):
            return u

    q = f'"{name}" official datasheet technical specifications'
    extra = _search_urls(provider, q, 8, warnings, name)
    for u in extra:
        if _is_official_domain_for_competitor(_url_domain(u), name):
            return u
    return ""


def _has_cross_source_support(feat: MappedFeature, sources: List[SourceEvidence], this_domain: str) -> bool:
    value_txt = str(feat.value or "").strip().lower()
    raw_txt = str(feat.raw_name or "").strip().lower()
    if not value_txt and not raw_txt:
        return False
    for s in sources:
        d = _url_domain(s.url)
        if not d or d == this_domain:
            continue
        ex = str(s.excerpt or "").lower()
        if value_txt and value_txt in ex:
            return True
        if raw_txt and raw_txt in ex:
            return True
    return False


def _dedupe_prices(prices: List[PriceInfo]) -> List[PriceInfo]:
    out: List[PriceInfo] = []
    seen: set[str] = set()
    for p in prices:
        key = f"{p.currency.lower()}|{p.value}|{_url_domain(p.source_url)}"
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _to_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _extract_text_from_sources(cp: CompetitorProfile, source_url: str) -> str:
    su = str(source_url or "").strip().lower()
    if not su:
        return ""
    for s in cp.sources:
        if str(s.url or "").strip().lower() == su:
            return str(s.excerpt or "")
    return ""


def _validate_and_correct_existing_prices(
    *,
    cp: CompetitorProfile,
    min_model_hits: int,
    warnings: List[str],
) -> Tuple[List[PriceInfo], List[SourceEvidence], int, int, int]:
    checked_pages = 0
    corrected = 0
    dropped = 0
    added_sources: List[SourceEvidence] = []
    cache: Dict[str, Tuple[str, str]] = {}  # url -> (title, text)
    out: List[PriceInfo] = []

    for pr in list(cp.prices or []):
        src = str(pr.source_url or "").strip()
        d = _url_domain(src)
        if not src or not _domain_is_vendor(d):
            # Keep non-vendor prices as-is; enrichment step covers vendor checks.
            out.append(pr)
            continue

        title = ""
        text = _extract_text_from_sources(cp, src)
        if src in cache:
            title, text_cached = cache[src]
            if not text:
                text = text_cached
        elif not text:
            try:
                title, text, _html, _ct = _fetch_page(src)
                checked_pages += 1
                cache[src] = (title, text)
                added_sources.append(
                    SourceEvidence(
                        url=src,
                        title=title[:180],
                        retrieved_at=datetime.now(timezone.utc).isoformat(),
                        excerpt=text[:1500],
                    )
                )
            except Exception as exc:
                warnings.append(f"Existing price validation crawl failed for '{cp.name}' url={src}: {exc}")
                dropped += 1
                continue

        gate_text = f"{title}\n{text[:20000]}"
        if not _is_model_match(gate_text, cp.name, min_hits=min_model_hits):
            dropped += 1
            warnings.append(f"Dropped price (model mismatch) for '{cp.name}' from {src}")
            continue

        extracted = _extract_prices(text, source_url=src)
        if not extracted:
            # If page matches model but no parseable price on page, keep original.
            out.append(pr)
            continue

        cur_val = _to_float(pr.value)
        if cur_val is None:
            best = extracted[0]
            out.append(best)
            corrected += 1
            continue

        # choose nearest extracted price and correct if deviation is large
        nearest = min(
            extracted,
            key=lambda x: abs((_to_float(x.value) or cur_val) - cur_val),
        )
        near_val = _to_float(nearest.value)
        if near_val is None:
            out.append(pr)
            continue

        rel_diff = abs(near_val - cur_val) / max(1.0, abs(cur_val))
        if rel_diff > 0.35:
            fixed = PriceInfo(
                raw=nearest.raw,
                value=near_val,
                currency=nearest.currency or pr.currency,
                package=pr.package,
                source_url=src,
            )
            out.append(fixed)
            corrected += 1
            warnings.append(
                f"Corrected existing price for '{cp.name}' from {cur_val} to {near_val} ({src})"
            )
        else:
            out.append(pr)

    return _dedupe_prices(out), added_sources, checked_pages, corrected, dropped


def _is_noisy_mapped_feature(
    mf: MappedFeature,
    *,
    competitor_name: str,
    target_schema: set[str],
) -> Optional[str]:
    sf = str(mf.schema_feature or "").strip()
    raw = str(mf.raw_name or "").strip()
    raw_l = raw.lower()
    val = str(mf.value or "").strip()
    val_l = val.lower()
    unit = str(mf.unit or "").strip().lower()
    nval = _to_float(mf.normalized_value)

    if not raw and not val:
        return "empty_feature"
    if sf and sf not in target_schema:
        return "schema_not_target"
    if sf.lower() == "other":
        return "schema_other"
    if any(re.search(p, raw_l) for p in _NOISY_RAWNAME_PATTERNS):
        return "rawname_noise_pattern"
    if len(raw) > 120:
        return "rawname_too_long"
    if "|" in val and not unit and nval is None:
        return "title_like_value"
    if any(h in val_l for h in _NOISY_VALUE_HINTS) and not unit and nval is None:
        return "value_noise_phrase"
    if sf.lower() == "compliance":
        if any(h in val_l for h in _NORM_HINTS) and nval is not None and nval >= 1000 and not unit:
            return "compliance_norm_misparsed_numeric"
    if sf.lower() == "durchmesser":
        if not unit and any(ch.isalpha() for ch in val) and "mm" not in val_l and "cm" not in val_l and "m" not in val_l:
            return "diameter_without_unit_context"
    # Model-independent weak sentence fragments
    if raw and raw_l.startswith("the ") and len(val.split()) > 3 and nval is None:
        return "sentence_fragment"
    # obvious mismatch to competitor name when value is another brand line
    brands = _brand_tokens(competitor_name)
    if any(b in val_l for b in ("solaredge", "huawei", "goodwe", "growatt", "sungrow", "solis", "deye", "sma")):
        if brands and not any(b in val_l for b in brands):
            return "other_brand_in_value"
    return None


def _enrich_prices_for_profile(
    *,
    cp: CompetitorProfile,
    provider: str,
    max_pages: int,
    min_model_hits: int,
    warnings: List[str],
) -> Tuple[List[PriceInfo], List[SourceEvidence], int]:
    candidate_urls: List[str] = []

    for s in cp.sources:
        d = _url_domain(s.url)
        if _domain_is_vendor(d):
            candidate_urls.append(s.url)

    q = _build_price_search_query(cp.name)
    candidate_urls.extend(_search_urls(provider, q, max(max_pages * 2, 8), warnings, cp.name))

    deduped_urls: List[str] = []
    seen: set[str] = set()
    for u in candidate_urls:
        uu = str(u or "").strip()
        if not uu.startswith("http"):
            continue
        k = uu.lower()
        if k in seen:
            continue
        seen.add(k)
        deduped_urls.append(uu)

    added_sources: List[SourceEvidence] = []
    new_prices: List[PriceInfo] = []
    checked = 0

    for u in deduped_urls:
        if checked >= max_pages:
            break
        d = _url_domain(u)
        if not _domain_is_vendor(d):
            continue
        checked += 1
        try:
            title, text, _html, _ct = _fetch_page(u)
        except Exception as exc:
            warnings.append(f"Price enrichment crawling failed for '{cp.name}' url={u}: {exc}")
            continue

        gate_text = f"{title}\n{text[:20000]}"
        if not _is_model_match(gate_text, cp.name, min_hits=min_model_hits):
            continue

        prices = _extract_prices(text, source_url=u)
        if not prices:
            continue

        new_prices.extend(prices)
        added_sources.append(
            SourceEvidence(
                url=u,
                title=title[:180],
                retrieved_at=datetime.now(timezone.utc).isoformat(),
                excerpt=text[:1500],
            )
        )

    return _dedupe_prices(new_prices), added_sources, checked


def run_competitor_profile_extraction_quality_gate(
    *,
    competitor_profiles: Optional[Dict[str, Any]],
    competitor_profiles_path: Optional[str],
    provider: str,
    enrich_prices: bool,
    max_price_pages_per_competitor: int,
    require_model_token_hits: int,
    verbose_progress: bool,
    drop_unverified_features: bool,
    user_root: Path,
    work_root: Path,
) -> CompetitorProfiles:
    profiles = _load_competitor_profiles(
        competitor_profiles=competitor_profiles,
        competitor_profiles_path=competitor_profiles_path,
        user_root=user_root,
        work_root=work_root,
    )

    warnings: List[str] = list(profiles.extraction_warnings or [])
    out_profiles: List[CompetitorProfile] = []
    target_schema_set = {str(x).strip() for x in (profiles.target_feature_schema or []) if str(x).strip()}

    total = len(profiles.competitor_profiles or [])
    for i, cp in enumerate(profiles.competitor_profiles or [], start=1):
        if verbose_progress:
            print(
                f"[competitor_profile_extraction_quality_gate] {i}/{total} processing: {cp.name}",
                flush=True,
            )

        cur = cp.model_copy(deep=True)

        base_domain = _url_domain(cur.url)
        if base_domain and not _is_official_domain_for_competitor(base_domain, cur.name):
            warnings.append(
                f"Non-official base URL detected for '{cur.name}': {cur.url}"
            )
            official = _find_official_url(cur.name, [s.url for s in cur.sources], warnings, provider)
            if official:
                warnings.append(f"Official URL candidate for '{cur.name}': {official}")
                cur.data_quality.notes.append(
                    "Official source candidate found; original profile.url retained for source traceability."
                )

        verified_features: List[MappedFeature] = []
        dropped_noisy = 0
        dropped_unverified = 0
        for mf in cur.mapped_features:
            noisy_reason = _is_noisy_mapped_feature(
                mf,
                competitor_name=cur.name,
                target_schema=target_schema_set,
            )
            if noisy_reason:
                dropped_noisy += 1
                continue
            src_domain = _url_domain(mf.source_url)
            source_is_official = _is_official_domain_for_competitor(src_domain, cur.name)
            supported = _has_cross_source_support(mf, cur.sources, src_domain)
            if not source_is_official and not supported:
                if drop_unverified_features:
                    dropped_unverified += 1
                    continue
                note = (
                    f"Feature for '{cur.name}' not cross-validated from independent source: "
                    f"{mf.raw_name or mf.schema_feature}"
                )
                cur.data_quality.notes.append(note)
            verified_features.append(mf)
        if dropped_noisy > 0:
            warnings.append(f"Dropped {dropped_noisy} noisy mapped_features for '{cur.name}'.")
            cur.data_quality.notes.append(f"Noisy mapped_features removed: {dropped_noisy}")
        if dropped_unverified > 0:
            warnings.append(f"Dropped {dropped_unverified} unverified mapped_features for '{cur.name}'.")
        cur.mapped_features = verified_features

        existing_checked = 0
        existing_corrected = 0
        existing_dropped = 0
        if cur.prices:
            validated_prices, validation_sources, existing_checked, existing_corrected, existing_dropped = _validate_and_correct_existing_prices(
                cp=cur,
                min_model_hits=require_model_token_hits,
                warnings=warnings,
            )
            cur.prices = validated_prices
            if existing_checked > 0:
                cur.data_quality.notes.append(f"Existing price pages checked: {existing_checked}")
            if existing_corrected > 0:
                cur.data_quality.notes.append(f"Existing prices corrected: {existing_corrected}")
            if existing_dropped > 0:
                cur.data_quality.notes.append(f"Existing prices dropped: {existing_dropped}")
            existing_src = {str(s.url or "").strip().lower() for s in cur.sources}
            for src in validation_sources:
                if str(src.url or "").strip().lower() in existing_src:
                    continue
                cur.sources.append(src)

        if enrich_prices:
            existing_prices = list(cur.prices or [])
            extra_prices, added_sources, checked_pages = _enrich_prices_for_profile(
                cp=cur,
                provider=provider,
                max_pages=max_price_pages_per_competitor,
                min_model_hits=require_model_token_hits,
                warnings=warnings,
            )
            if checked_pages > 0:
                cur.data_quality.notes.append(f"Price pages checked: {checked_pages}")
            if extra_prices:
                cur.prices = _dedupe_prices(existing_prices + extra_prices)
                warnings.append(f"Added {len(extra_prices)} model-matched prices for '{cur.name}'.")
                # add only non-duplicate sources
                existing_src = {str(s.url or "").strip().lower() for s in cur.sources}
                for src in added_sources:
                    if str(src.url or "").strip().lower() in existing_src:
                        continue
                    cur.sources.append(src)

        if not cur.prices:
            cur.data_quality.notes.append("No model-matched seller price found.")

        # tiny confidence adjustment based on verified price availability
        if cur.prices and cur.data_quality.confidence < 0.95:
            cur.data_quality.confidence = min(0.95, cur.data_quality.confidence + 0.04)

        cur.data_quality.notes = _dedupe_str(cur.data_quality.notes)
        out_profiles.append(cur)

    warnings = _dedupe_str(warnings)

    return CompetitorProfiles(
        schema_version=profiles.schema_version,
        provider=profiles.provider,
        target_feature_schema=list(profiles.target_feature_schema or []),
        competitor_profiles=out_profiles,
        extraction_warnings=warnings,
        batch_offset=profiles.batch_offset,
        batch_limit=profiles.batch_limit,
        batch_total_candidates=profiles.batch_total_candidates,
        processed_count=profiles.processed_count,
    )
