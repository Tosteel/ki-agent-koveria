from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from fastapi import HTTPException

from server.tools.langsearch.langsearch import search_langsearch

from .models import CompetitorCandidate, CompetitorList

OPENAI_URL = os.getenv("OPENAI_URL", "https://api.openai.com/v1/responses").strip() or "https://api.openai.com/v1/responses"
PERPLEXITY_URL = (
    os.getenv("PERPLEXITY_URL", "https://api.perplexity.ai/chat/completions").strip()
    or "https://api.perplexity.ai/chat/completions"
)
STOPWORDS = {
    "und", "oder", "der", "die", "das", "ein", "eine", "mit", "von", "für", "fuer", "the", "and", "for", "mit", "ohne",
    "zu", "in", "on", "at", "is", "are", "a", "an", "de", "www", "com", "gmbh",
}
PLACEHOLDER_PATTERNS = [
    re.compile(r"\bhersteller\s+[a-z]\b", re.IGNORECASE),
    re.compile(r"\bmanufacturer\s+[a-z]\b", re.IGNORECASE),
    re.compile(r"\bexample\b", re.IGNORECASE),
    re.compile(r"\bdummy\b", re.IGNORECASE),
    re.compile(r"\btest\b", re.IGNORECASE),
]
FAMILY_TRIM_TOKENS = {
    "gt", "gts", "gtr", "rs", "r", "s", "se", "x", "z",
    "coupe", "coupé", "sedan", "saloon", "wagon", "estate",
    "hatch", "hatchback", "convertible", "cabrio", "roadster",
    "edition", "performance", "sport", "plus", "ultra", "pro",
    "premium", "standard", "base", "type", "model", "series",
}
LEGAL_NAME_TOKENS = {
    "inc", "corp", "corporation", "company", "co", "ltd", "llc", "gmbh", "ag", "sa", "plc", "kg", "srl",
}
TRUSTED_EXTERNAL_DOMAIN_HINTS = (
    "auto-motor-und-sport",
    "autobild",
    "motor1",
    "caranddriver",
    "topgear",
    "edmunds",
    "carwow",
    "whichcar",
    "autocar",
    "wikipedia.org",
)
LOW_TRUST_URL_HINTS = (
    "autokatalog",
    "used-cars",
    "gebrauchtwagen",
    "classified",
    "listing",
    "listings",
    "vehicle-list",
    "inventory",
    "dealer",
    "marketplace",
    "vergleich",
)
LOW_TRUST_DOMAIN_TOKENS = (
    "autoscout",
    "autotrader",
    "mobile.",
    "ebay",
    "craigslist",
    "olx",
)
NOT_FOUND_MARKERS = (
    "page not found",
    "not found",
    "404",
    "seite konnte leider nicht gefunden werden",
    "die seite konnte leider nicht gefunden werden",
    "seite nicht gefunden",
    "sorry, this page",
)


def _safe_list_str(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
    for v in values:
        s = str(v or "").strip()
        if s:
            out.append(s)
    return out


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

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            if user_root in candidate.parents or candidate == user_root:
                return candidate

    raise HTTPException(status_code=404, detail=f"File not found: {path}")


def _load_json_obj(
    *,
    inline_obj: Optional[Dict[str, Any]],
    path: Optional[str],
    root_key: str,
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

    if root_key in payload and isinstance(payload.get(root_key), dict):
        payload = payload[root_key]

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail=f"Invalid {root_key} payload")
    return payload


def _tokenize(text: str) -> List[str]:
    raw = re.findall(r"[a-zA-Z0-9äöüÄÖÜß-]{2,}", (text or "").lower())
    return [t for t in raw if t not in STOPWORDS]


def _cosine_similarity(a: str, b: str) -> float:
    ta = _tokenize(a)
    tb = _tokenize(b)
    if not ta or not tb:
        return 0.0
    ca = Counter(ta)
    cb = Counter(tb)
    keys = set(ca) | set(cb)
    dot = sum(ca.get(k, 0) * cb.get(k, 0) for k in keys)
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    if na == 0 or nb == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


def _cluster_for_url(url: str, title: str, snippet: str) -> str:
    txt = f"{url} {title} {snippet}".lower()
    if any(k in txt for k in ("youtube.com", "youtu.be", "vimeo.com", "tiktok.com", "dailymotion.com")):
        return "video"
    if any(k in txt for k in ("amazon", "ebay", "alibaba", "mercateo", "shop", "store")):
        return "marketplace"
    if any(k in txt for k in ("wikipedia", "blog", "news", "magazin", "heise", "medium", "autozeitung", "motor1", "caranddriver", "topgear")):
        return "media"
    if any(k in txt for k in ("datasheet", "datenblatt", "pdf", "produkte", "product", "solutions")):
        return "manufacturer"
    return "unknown"


def _domain(url: str) -> str:
    s = str(url or "").lower()
    s = re.sub(r"^https?://", "", s)
    s = s.split("/", 1)[0]
    return s.replace("www.", "")


def _url_path(url: str) -> str:
    try:
        return (urlparse(str(url or "")).path or "").lower()
    except Exception:
        return ""


def _clean_url(url: str) -> str:
    u = str(url or "").strip()
    if not u:
        return ""
    # normalize locale segment style like /en_US/ -> /en-us/
    u = re.sub(
        r"/([a-z]{2})_([a-z]{2})(?=/|$)",
        lambda m: f"/{m.group(1).lower()}-{m.group(2).lower()}",
        u,
        flags=re.IGNORECASE,
    )
    u = re.sub(r"#.*$", "", u)
    # drop noisy tracking query params, keep path
    u = re.sub(r"\?(utm_[^=]+=[^&]+&?)+", "?", u, flags=re.IGNORECASE)
    u = u.rstrip("?&")
    return u


def _pick_name(title: str, url: str) -> str:
    t = str(title or "").strip()
    if t:
        return t[:140]
    d = _domain(url)
    if d:
        return d
    return "Unknown Competitor"


def _clean_text(s: str) -> str:
    txt = str(s or "")
    txt = txt.replace("\x00", " ")
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def _norm_compact(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def _token_set(s: str) -> set[str]:
    return {t for t in _tokenize(_norm_compact(s)) if t}


def _name_key(s: str) -> str:
    toks = [t for t in _tokenize(_norm_compact(s)) if len(t) >= 2]
    return " ".join(toks[:8]).strip()


def _model_stem(s: str) -> str:
    drop = {
        "series", "model", "edition", "version", "mk", "type", "typ", "class", "klasse",
        "pro", "plus", "ultra", "max", "mini", "standard", "premium", "basic",
    }
    toks = [t for t in _tokenize(_norm_compact(s)) if len(t) >= 2 and t not in drop]
    if not toks:
        return ""
    # Keep a short, stable stem for dedupe across minor trims/variants.
    return " ".join(toks[:3])


def _family_key(s: str) -> str:
    toks = [t for t in _tokenize(_norm_compact(s)) if len(t) >= 2 and t not in FAMILY_TRIM_TOKENS]
    if not toks:
        return ""
    if len(toks) == 1:
        return toks[0]
    return f"{toks[0]} {toks[1]}"


def _name_tokens(name: str) -> List[str]:
    return [t for t in _tokenize(_norm_compact(name)) if len(t) >= 2 and t not in LEGAL_NAME_TOKENS]


def _brand_token(name: str) -> str:
    toks = _name_tokens(name)
    return toks[0] if toks else ""


def _model_tokens(name: str) -> List[str]:
    toks = _name_tokens(name)
    if len(toks) <= 1:
        return []
    return toks[1:]


def _build_main_anchor(product_name: str, manufacturer: str) -> str:
    pn = _clean_text(str(product_name or "").replace("_", " "))
    mf = _clean_text(manufacturer)
    if pn and mf and mf.lower() not in pn.lower():
        return f"{mf} {pn}".strip()
    return pn or mf or "product"


def _build_market_fit_text(profile: Dict[str, Any]) -> str:
    category = str(profile.get("product_category") or "").strip()
    segments = _safe_list_str(profile.get("target_segments"))
    use_cases = _safe_list_str(profile.get("use_cases"))
    differentiators = _safe_list_str(profile.get("differentiators"))
    claims = [str(c.get("text") or "").strip() for c in (profile.get("claims") or []) if isinstance(c, dict)]
    parts: List[str] = []
    if category:
        parts.append(f"category: {category}")
    if segments:
        parts.append("segments: " + ", ".join(segments[:10]))
    if use_cases:
        parts.append("use_cases: " + ", ".join(use_cases[:10]))
    if differentiators:
        parts.append("differentiators: " + ", ".join(differentiators[:10]))
    if claims:
        parts.append("claims: " + " | ".join(claims[:10]))
    return "\n".join(parts).strip()


def _source_quality_multiplier(cluster: str, url: str, title: str, snippet: str) -> float:
    txt = f"{url} {title} {snippet}".lower()
    if cluster == "video":
        return 0.62
    if cluster == "marketplace":
        return 0.72
    if cluster == "media":
        return 0.86

    # Strong signals for canonical product/spec pages.
    if any(k in txt for k in ("/product", "/products", "/model", "/models", "/solutions", "/showroom")):
        return 1.10
    if any(k in txt for k in ("datasheet", "datenblatt", "technical data", "specs", "spezifikation", ".pdf")):
        return 1.06
    return 1.0


def _garbled_ratio(s: str) -> float:
    if not s:
        return 1.0
    ok = 0
    for ch in s:
        o = ord(ch)
        if ch in "\n\r\t" or 32 <= o <= 126 or ch in "äöüÄÖÜß€°–—-/%()[],:.;+_&'\"":
            ok += 1
    return 1.0 - (ok / max(1, len(s)))


def _is_placeholder_candidate(*, name: str, url: str, snippet: str) -> bool:
    full = f"{name} {url} {snippet}"
    low = full.lower()
    if "\x00" in full:
        return True
    if "hersteller-" in low and re.search(r"hersteller-[a-z]\b", low):
        return True
    if _garbled_ratio(full) > 0.20:
        return True
    for pat in PLACEHOLDER_PATTERNS:
        if pat.search(full):
            return True
    return False


def _is_url_reachable(url: str, timeout_s: int = 6) -> bool:
    u = _clean_url(url)
    if not u.startswith("http"):
        return False
    try:
        h = requests.head(u, timeout=timeout_s, allow_redirects=True)
        if 200 <= h.status_code < 400:
            return True
        # Many sites block HEAD; fallback to lightweight GET.
        g = requests.get(u, timeout=timeout_s, allow_redirects=True, stream=True)
        return 200 <= g.status_code < 400
    except Exception:
        return False


def _is_url_usable_for_candidate(url: str, candidate_name: str, timeout_s: int = 8) -> bool:
    u = _clean_url(url)
    if not u.startswith("http"):
        return False
    try:
        r = requests.get(u, timeout=timeout_s, allow_redirects=True)
        if not (200 <= r.status_code < 400):
            return False
        final_url = _clean_url(str(r.url or u))
        if not final_url.startswith("http"):
            final_url = u
        ctype = str(r.headers.get("Content-Type") or "").lower()
        if "html" not in ctype and "xhtml" not in ctype:
            # PDFs/other docs are acceptable if reachable
            return True
        txt = str(r.text or "")
        low = txt.lower()
        if any(
            k in low
            for k in (
                "site maintenance",
                "oops! something went wrong",
                "error code",
                "access denied",
                "404",
                "page not found",
                "not found",
                "coming soon",
                "enable javascript",
                "robot check",
                "captcha",
            )
        ):
            return False
        if any(m in low for m in NOT_FOUND_MARKERS):
            return False

        # Very short HTML pages are often placeholders/cookie gates.
        body_plain = re.sub(r"<[^>]+>", " ", low)
        body_plain = re.sub(r"\s+", " ", body_plain).strip()
        if len(body_plain) < 180:
            return False

        # Extract title for stricter lexical checks.
        title_match = re.search(r"<title[^>]*>(.*?)</title>", low, flags=re.IGNORECASE | re.DOTALL)
        title_txt = ""
        if title_match:
            title_txt = re.sub(r"\s+", " ", title_match.group(1)).strip()

        # Require lexical relation to candidate in URL/title first (strong),
        # otherwise fall back to weak body match.
        n_tokens = [t for t in _tokenize(_norm_compact(candidate_name)) if len(t) >= 3]
        if not n_tokens:
            return True
        strong_text = _norm_compact(f"{final_url} {title_txt}")
        strong_tokens = set(_tokenize(strong_text))
        if any(t in strong_tokens for t in n_tokens[:4]):
            return True
        weak_hits = sum(1 for t in n_tokens[:4] if t in body_plain)
        return weak_hits >= 2
    except Exception:
        return False


def _is_same_brand_candidate(*, candidate_name: str, candidate_url: str, manufacturer: str) -> bool:
    mf_tokens = [t for t in _tokenize(_norm_compact(manufacturer)) if len(t) >= 2 and t not in LEGAL_NAME_TOKENS]
    if not mf_tokens:
        return False
    cand_tokens = set(_tokenize(_norm_compact(f"{candidate_name} {_domain(candidate_url)}")))
    return bool(set(mf_tokens) & cand_tokens)


def _url_model_match_score(candidate_name: str, url: str) -> float:
    """
    0..1 score for how well URL path/domain reflects model tokens
    (tokens after brand token in candidate name).
    """
    model = _model_tokens(candidate_name)
    if not model:
        return 1.0
    txt = set(_tokenize(_norm_compact(f"{_domain(url)} {_url_path(url)}")))
    if not txt:
        return 0.0
    overlap = len(set(model) & txt)
    return overlap / max(1, len(set(model)))


def _is_domain_root_reachable(domain: str, timeout_s: int = 6) -> bool:
    d = str(domain or "").strip().lower().replace("www.", "")
    if not d:
        return False
    for root in (f"https://{d}", f"http://{d}"):
        if _is_url_reachable(root, timeout_s=timeout_s):
            return True
    return False


def _candidate_key(c: CompetitorCandidate) -> str:
    nk = _name_key(c.name)
    sk = _model_stem(c.name)
    fk = _family_key(c.name)
    dk = _domain(c.url)
    if fk and len(fk.split()) >= 2:
        return fk
    if sk:
        return sk
    return f"{nk}|{dk}" if nk and dk else (nk or dk or _clean_url(c.url).lower())


def _name_url_consistent(name: str, url: str, title: str = "", snippet: str = "") -> bool:
    """
    Generic consistency check: replacement URL must still look like the same entity
    as the candidate name (brand/model token presence).
    """
    n_tokens = [t for t in _tokenize(_norm_compact(name)) if len(t) >= 3]
    if not n_tokens:
        return True
    text = _norm_compact(f"{url} {title} {snippet}")
    tset = set(_tokenize(text))
    if not tset:
        return False

    # Require brand token (first token) OR at least two name tokens.
    brand_ok = n_tokens[0] in tset
    overlap = len(set(n_tokens) & tset)
    return bool(brand_ok or overlap >= 2)


def _name_url_consistent_strict(name: str, url: str, title: str = "", snippet: str = "") -> bool:
    """
    Stricter consistency for external fallback sources:
    require >=2 overlapping name tokens to avoid model swaps.
    """
    n_tokens = [t for t in _tokenize(_norm_compact(name)) if len(t) >= 3]
    if len(n_tokens) < 2:
        return _name_url_consistent(name, url, title, snippet)
    text = _norm_compact(f"{url} {title} {snippet}")
    tset = set(_tokenize(text))
    if not tset:
        return False
    overlap = len(set(n_tokens) & tset)
    return overlap >= 2


def _is_trusted_external_url(url: str, candidate_domain: str) -> bool:
    d = _domain(url)
    if not d:
        return False
    if candidate_domain and d == candidate_domain:
        return False
    return any(h in d for h in TRUSTED_EXTERNAL_DOMAIN_HINTS)


def _is_low_trust_url(url: str) -> bool:
    d = _domain(url)
    p = _url_path(url)
    if not d and not p:
        return False
    # Trusted editorial sources should not be demoted by generic path hints.
    if any(h in d for h in TRUSTED_EXTERNAL_DOMAIN_HINTS):
        return False
    if any(tok in d for tok in LOW_TRUST_DOMAIN_TOKENS):
        return True
    if any(h in p for h in LOW_TRUST_URL_HINTS):
        return True
    return False


def _find_replacement_urls(
    *,
    cand: CompetitorCandidate,
    all_candidates: List[CompetitorCandidate],
    provider: str,
    openai_key: str,
    openai_model: str,
    perplexity_key: str,
    perplexity_model: str,
    per_query_results: int,
    product_name: str,
    manufacturer: str,
) -> List[Tuple[str, str]]:
    key = _candidate_key(cand)
    out: List[Tuple[str, str]] = []
    seen: set[str] = set()

    def _push(u: str, source: str) -> None:
        cu = _clean_url(u)
        if not cu or cu in seen:
            return
        seen.add(cu)
        out.append((cu, source))

    # 1) Prefer already-seen alternatives from same candidate group.
    alts = [c for c in all_candidates if _candidate_key(c) == key and _clean_url(c.url) != _clean_url(cand.url)]
    for alt in alts:
        u = _clean_url(alt.url)
        if _is_low_trust_url(u):
            continue
        if not _name_url_consistent(cand.name, u, alt.name, alt.snippet):
            continue
        if _is_self_or_variant_candidate(
            candidate_name=cand.name,
            candidate_url=u,
            candidate_snippet=alt.snippet,
            product_name=product_name,
            manufacturer=manufacturer,
        ):
            continue
        _push(u, "candidate_group_alt")
        if len(out) >= 3:
            return out[:3]

    # 2) Query-specific recovery.
    q = f"{cand.name} official technical data specifications"
    fetched: List[Dict[str, str]] = []
    if provider == "openai" and openai_key:
        try:
            fetched = _openai_search(q, max(4, per_query_results), api_key=openai_key, model=openai_model)
        except Exception:
            fetched = []
    elif provider == "perplexity" and perplexity_key:
        try:
            fetched = _perplexity_search(q, max(4, per_query_results), api_key=perplexity_key, model=perplexity_model)
        except Exception:
            fetched = []
    if not fetched:
        try:
            fetched = _langsearch_fallback(q, max(4, per_query_results))
        except Exception:
            fetched = []
    for r in fetched:
        u = str(r.get("url") or "").strip()
        if not u:
            continue
        ru = _clean_url(u)
        if _is_low_trust_url(ru):
            continue
        title = _clean_text(str(r.get("title") or ""))
        snip = _clean_text(str(r.get("snippet") or ""))
        if not _name_url_consistent(cand.name, ru, title, snip):
            continue
        if _is_self_or_variant_candidate(
            candidate_name=cand.name,
            candidate_url=ru,
            candidate_snippet=snip,
            product_name=product_name,
            manufacturer=manufacturer,
        ):
            continue
        _push(ru, "replacement_search")
        if len(out) >= 3:
            return out[:3]
    return out[:3]


def _score_url_option(
    *,
    url: str,
    candidate: CompetitorCandidate,
    reachable_cache: Dict[str, bool],
    usable_cache: Dict[str, bool],
) -> float:
    u = _clean_url(url)
    if not u:
        return -1.0
    r_key = u.lower()
    if r_key not in reachable_cache:
        reachable_cache[r_key] = _is_url_reachable(u)
    n_key = _name_key(candidate.name)
    u_key = f"{u.lower()}|{n_key}"
    if u_key not in usable_cache:
        usable_cache[u_key] = _is_url_usable_for_candidate(u, candidate.name)
    reachable = reachable_cache[r_key]
    usable = usable_cache[u_key]
    consistent = _name_url_consistent(candidate.name, u, candidate.name, candidate.snippet)
    model_match = _url_model_match_score(candidate.name, u)
    trusted_external = _is_trusted_external_url(u, _domain(candidate.url))

    score = 0.0
    if usable:
        score += 1.0
    elif reachable:
        score += 0.35
    if consistent:
        score += 0.30
    score += 0.25 * model_match
    if trusted_external and usable:
        score += 0.08
    if reachable and consistent:
        score += 0.10
    if model_match < 0.34:
        score -= 0.20
    return score


def _is_self_or_variant_candidate(
    *,
    candidate_name: str,
    candidate_url: str,
    candidate_snippet: str,
    product_name: str,
    manufacturer: str,
) -> bool:
    """
    Generic exclusion rule: prevent exact product / close same-family variants
    from being included as competitors.
    """
    pn = _norm_compact(str(product_name or "").replace("_", " "))
    if not pn:
        return False

    cand_name = _norm_compact(candidate_name)
    cand_domain = _norm_compact(_domain(candidate_url))
    cand_title_domain = f"{cand_name} {cand_domain}".strip()
    cand_txt = _norm_compact(f"{candidate_name} {candidate_url} {candidate_snippet}")
    if not cand_txt:
        return False

    pn_tokens = _token_set(product_name)
    cand_tokens = _token_set(cand_txt)
    if not pn_tokens or not cand_tokens:
        return False

    mf = _norm_compact(manufacturer)
    same_brand = False
    if mf:
        same_brand = mf in cand_title_domain
        if not same_brand:
            mf_tokens = _token_set(mf)
            cand_td_tokens = _token_set(cand_title_domain)
            same_brand = bool(mf_tokens and (mf_tokens & cand_td_tokens))
    else:
        # Without manufacturer, apply only strict exact-family check.
        return pn in cand_title_domain

    if not same_brand:
        return False

    # High confidence self/variant signals only.
    if pn in cand_title_domain:
        return True
    overlap = len(pn_tokens & _token_set(cand_title_domain)) / max(1, len(pn_tokens))
    return overlap >= 0.7


def _expand_queries_for_undercoverage(
    *,
    base_queries: List[str],
    plan: Dict[str, Any],
    profile: Dict[str, Any],
    max_extra: int = 5,
) -> List[str]:
    metadata = profile.get("metadata") if isinstance(profile.get("metadata"), dict) else {}
    product_name = str(metadata.get("product_name") or "").strip()
    manufacturer = str(metadata.get("manufacturer") or "").strip()
    category = str(plan.get("product_category") or "").strip()

    anchor = _build_main_anchor(product_name, manufacturer) or category or "product"
    category_anchor = category or "product"

    extras = [
        f"{anchor} alternatives",
        f"{anchor} competitors",
        f"{anchor} competing products",
        f"{anchor} vs alternatives",
        f"{category_anchor} market leaders alternatives to {anchor}",
        f"{category_anchor} similar products to {anchor}",
    ]
    if manufacturer and product_name:
        extras.append(f"{manufacturer} alternatives to {product_name}")

    seen = {q.lower().strip() for q in base_queries}
    out: List[str] = []
    for q in extras:
        k = q.lower().strip()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(q)
        if len(out) >= max_extra:
            break
    return out


def _openai_search(query: str, per_query_results: int, api_key: str, model: str) -> List[Dict[str, str]]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    system = (
        "Nutze Websuche und liefere nur JSON mit dem Schema: "
        "{\"results\":[{\"name\":\"\",\"url\":\"\",\"snippet\":\"\"}]}. "
        f"Maximal {int(per_query_results)} Ergebnisse. "
        "Wichtig: Nur reale Unternehmen/Produktanbieter zurückgeben. "
        "Keine Platzhalter, keine generischen Namen wie 'Hersteller A/B/C', keine erfundenen Domains."
    )
    user = f"Finde relevante Wettbewerber für die Suchanfrage: {query}"

    payload: Dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system}]},
            {"role": "user", "content": [{"type": "input_text", "text": user}]},
        ],
        "tools": [{"type": "web_search_preview"}],
        "tool_choice": {"type": "web_search_preview"},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "web_search_results",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "results": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "name": {"type": "string"},
                                    "url": {"type": "string"},
                                    "snippet": {"type": "string"},
                                },
                                "required": ["name", "url", "snippet"],
                            },
                        }
                    },
                    "required": ["results"],
                },
                "strict": False,
            }
        },
        "include": ["web_search_call.action.sources"],
    }

    resp = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=60)
    if resp.status_code >= 400:
        raise RuntimeError(f"OpenAI web search HTTP {resp.status_code}: {resp.text}")

    data = resp.json()

    text = ""
    for item in data.get("output", []):
        for c in item.get("content", []):
            if c.get("type") == "output_text":
                text += str(c.get("text") or "")

    parsed: Dict[str, Any] = {}
    try:
        parsed = json.loads(text) if text else {}
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
            except Exception:
                parsed = {}

    out: List[Dict[str, str]] = []
    raw_results = parsed.get("results") if isinstance(parsed, dict) else None
    if isinstance(raw_results, list):
        for r in raw_results:
            if not isinstance(r, dict):
                continue
            url = str(r.get("url") or "").strip()
            if not url:
                continue
            title = _clean_text(str(r.get("name") or ""))
            snippet = _clean_text(str(r.get("snippet") or ""))
            out.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                }
            )

    if out:
        return out[:per_query_results]

    # Fallback: try to pick URL sources if model output JSON was not parseable.
    for item in data.get("output", []):
        if item.get("type") == "web_search_call":
            action = item.get("action") if isinstance(item.get("action"), dict) else {}
            sources = action.get("sources") if isinstance(action, dict) else None
            if isinstance(sources, list):
                for s in sources:
                    if not isinstance(s, dict):
                        continue
                    url = str(s.get("url") or "").strip()
                    if not url:
                        continue
                    title = _clean_text(str(s.get("title") or ""))
                    snippet = _clean_text(str(s.get("snippet") or ""))
                    out.append(
                        {
                            "title": title,
                            "url": url,
                            "snippet": snippet,
                        }
                    )
    return out[:per_query_results]


def _perplexity_search(query: str, per_query_results: int, api_key: str, model: str) -> List[Dict[str, str]]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    system = (
        "Liefere ausschließlich JSON im Format "
        "{\"results\":[{\"name\":\"\",\"url\":\"\",\"snippet\":\"\"}]}. "
        f"Maximal {int(per_query_results)} Ergebnisse. "
        "Nur reale Wettbewerber/Produkte, keine Platzhalter, keine erfundenen URLs."
    )
    user = f"Finde relevante Wettbewerber für: {query}"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "competitor_search_results",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "results": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "name": {"type": "string"},
                                    "url": {"type": "string"},
                                    "snippet": {"type": "string"},
                                },
                                "required": ["name", "url", "snippet"],
                            },
                        }
                    },
                    "required": ["results"],
                },
            },
        },
    }
    resp = requests.post(PERPLEXITY_URL, headers=headers, json=payload, timeout=60)
    if resp.status_code >= 400:
        raise RuntimeError(f"Perplexity search HTTP {resp.status_code}: {resp.text}")
    data = resp.json()
    text = ""
    try:
        text = str(data["choices"][0]["message"]["content"] or "")
    except Exception:
        text = ""

    parsed: Dict[str, Any] = {}
    try:
        parsed = json.loads(text) if text else {}
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
            except Exception:
                parsed = {}

    out: List[Dict[str, str]] = []
    raw_results = parsed.get("results") if isinstance(parsed, dict) else None
    if isinstance(raw_results, list):
        for r in raw_results:
            if not isinstance(r, dict):
                continue
            url = str(r.get("url") or "").strip()
            if not url:
                continue
            out.append(
                {
                    "title": _clean_text(str(r.get("name") or "")),
                    "url": url,
                    "snippet": _clean_text(str(r.get("snippet") or "")),
                }
            )

    # Best-effort fallback: extract URLs from free text.
    if not out and text:
        url_matches = re.findall(r"https?://[^\s\"'<>]+", text)
        for u in url_matches:
            out.append({"title": "", "url": u.strip(), "snippet": ""})

    # Keep unique URLs only.
    dedup: List[Dict[str, str]] = []
    seen: set[str] = set()
    for r in out:
        u = str(r.get("url") or "").strip()
        if not u:
            continue
        k = u.lower()
        if k in seen:
            continue
        seen.add(k)
        dedup.append(r)
        if len(dedup) >= per_query_results:
            break
    return dedup


def _langsearch_fallback(query: str, per_query_results: int) -> List[Dict[str, str]]:
    result = search_langsearch(query=query, count=per_query_results, summary=False)
    out: List[Dict[str, str]] = []
    for item in result.get("items") or []:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        if not url:
            continue
        out.append(
            {
                "title": _clean_text(str(item.get("title") or "")),
                "url": url,
                "snippet": _clean_text(str(item.get("snippet") or "")),
            }
        )
    return out[:per_query_results]


def _build_profile_text(profile: Dict[str, Any]) -> str:
    category = str(profile.get("product_category") or "").strip()
    claims = [str(c.get("text") or "") for c in (profile.get("claims") or []) if isinstance(c, dict)]
    features = [str(f.get("name") or "") for f in (profile.get("normalized_features") or []) if isinstance(f, dict)]
    segments = _safe_list_str(profile.get("target_segments"))
    use_cases = _safe_list_str(profile.get("use_cases"))
    metadata = profile.get("metadata") if isinstance(profile.get("metadata"), dict) else {}
    product_name = str(metadata.get("product_name") or "").strip()

    lines = [f"category: {category}", f"product: {product_name}"]
    if features:
        lines.append("features: " + ", ".join(features[:20]))
    if claims:
        lines.append("claims: " + " | ".join(claims[:10]))
    if segments:
        lines.append("segments: " + ", ".join(segments[:10]))
    if use_cases:
        lines.append("use_cases: " + ", ".join(use_cases[:10]))
    return "\n".join(lines).strip()


def _score_candidate(
    *,
    profile_text: str,
    market_fit_text: str,
    title: str,
    snippet: str,
    url: str,
    source_query: str,
    dimensions: List[str],
) -> CompetitorCandidate:
    combined = f"{title}\n{snippet}"
    similarity = _cosine_similarity(profile_text, combined)

    low = combined.lower()
    matched = [d for d in dimensions if d.lower() in low]
    dim_score = min(1.0, len(matched) / max(1, len(dimensions)))
    segment_fit = _cosine_similarity(market_fit_text, combined) if market_fit_text else 0.0

    query_score = _cosine_similarity(source_query, combined)
    relevance = 0.42 * similarity + 0.20 * query_score + 0.14 * dim_score + 0.24 * segment_fit
    cluster = _cluster_for_url(url, title, snippet)
    relevance *= _source_quality_multiplier(cluster, url, title, snippet)
    if similarity < 0.06 and dim_score == 0 and query_score < 0.15:
        relevance *= 0.75

    reasons = []
    if similarity >= 0.25:
        reasons.append("Hohe semantische Ähnlichkeit zur Produktbeschreibung")
    if matched:
        reasons.append("Treffer auf Vergleichsdimensionen: " + ", ".join(matched[:3]))
    if not reasons:
        reasons.append("Treffer aus Suchquery mit Basis-Relevanz")

    return CompetitorCandidate(
        name=_pick_name(title, url),
        url=url,
        snippet=snippet,
        source_query=source_query,
        cluster=cluster,
        similarity_score=round(similarity, 4),
        relevance_score=round(relevance, 4),
        matched_dimensions=matched,
        reasons=reasons,
    )


def _iter_queries(analysis_plan: Dict[str, Any], max_queries: int) -> List[str]:
    queries = _safe_list_str(analysis_plan.get("search_queries"))
    if queries:
        return queries[:max_queries]

    terms = []
    for t in analysis_plan.get("search_terms") or []:
        if isinstance(t, dict):
            term = str(t.get("term") or "").strip()
            if term:
                terms.append(term)

    category = str(analysis_plan.get("product_category") or "").strip() or "produkt"
    generated: List[str] = [f"{category} Wettbewerber", f"{category} Alternativen", f"{category} Datenblatt"]
    category_norm = _norm = re.sub(r"\s+", " ", category.strip().lower())
    for t in terms[:10]:
        t_norm = re.sub(r"\s+", " ", t.strip().lower())
        if not t_norm or t_norm == category_norm:
            continue
        if t_norm in category_norm or category_norm in t_norm:
            generated.append(f"{t}")
        else:
            generated.append(f"{category} {t}")

    # dedupe keep order
    out: List[str] = []
    seen: set[str] = set()
    for q in generated:
        k = q.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(q)
    return out[:max_queries]


def _dedupe_candidates(cands: Iterable[CompetitorCandidate]) -> List[CompetitorCandidate]:
    by_key: Dict[str, CompetitorCandidate] = {}
    for c in cands:
        nk = _name_key(c.name)
        sk = _model_stem(c.name)
        fk = _family_key(c.name)
        dk = _domain(c.url)
        if fk and len(fk.split()) >= 2:
            key = fk
        elif sk:
            key = sk
        else:
            key = f"{nk}|{dk}" if nk and dk else (nk or dk or c.url.lower())
        cur = by_key.get(key)
        if cur is None or c.relevance_score > cur.relevance_score:
            by_key[key] = c
    return list(by_key.values())


def _collapse_near_duplicate_names(cands: List[CompetitorCandidate]) -> List[CompetitorCandidate]:
    """
    Merge residual duplicates that differ only by minor name variants
    (e.g., brand legal-form additions) while keeping the higher-ranked item.
    """
    def norm_name_tokens(name: str) -> set[str]:
        toks = [t for t in _tokenize(_norm_compact(name)) if len(t) >= 2 and t not in LEGAL_NAME_TOKENS]
        return set(toks)

    ordered = sorted(cands, key=lambda c: (c.relevance_score, c.similarity_score), reverse=True)
    kept: List[CompetitorCandidate] = []
    for c in ordered:
        ct = norm_name_tokens(c.name)
        if not ct:
            kept.append(c)
            continue
        is_dup = False
        for k in kept:
            kt = norm_name_tokens(k.name)
            if not kt:
                continue
            inter = len(ct & kt)
            union = len(ct | kt)
            jacc = (inter / union) if union else 0.0
            # Strong overlap OR strict subset and same family model stem.
            if jacc >= 0.75:
                is_dup = True
                break
            c_stem = _model_stem(c.name)
            k_stem = _model_stem(k.name)
            if c_stem and k_stem and c_stem == k_stem and (ct <= kt or kt <= ct):
                is_dup = True
                break
        if not is_dup:
            kept.append(c)
    return kept


def identify_competitors(
    *,
    analysis_plan: Optional[Dict[str, Any]],
    analysis_plan_path: Optional[str],
    product_profile: Optional[Dict[str, Any]],
    product_profile_path: Optional[str],
    provider: str = "openai",
    max_queries: int = 8,
    per_query_results: int = 6,
    shortlist_size: int = 10,
    user_root: Path,
    work_root: Path,
) -> CompetitorList:
    plan = _load_json_obj(
        inline_obj=analysis_plan,
        path=analysis_plan_path,
        root_key="analysis_plan",
        user_root=user_root,
        work_root=work_root,
    )
    profile = _load_json_obj(
        inline_obj=product_profile,
        path=product_profile_path,
        root_key="product_profile",
        user_root=user_root,
        work_root=work_root,
    )

    warnings = _safe_list_str(plan.get("extraction_warnings")) + _safe_list_str(profile.get("extraction_warnings"))

    p = str(provider or "openai").strip().lower()
    if p not in {"openai", "ionos", "perplexity"}:
        p = "openai"

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    perplexity_key = os.getenv("PERPLEXITY_API_KEY", "").strip()
    perplexity_model = os.getenv("PERPLEXITY_MODEL", "sonar-pro").strip() or "sonar-pro"

    dimensions = [
        str(d.get("name") or "").strip()
        for d in (plan.get("comparison_dimensions") or [])
        if isinstance(d, dict) and str(d.get("name") or "").strip()
    ]

    profile_text = _build_profile_text(profile)
    market_fit_text = _build_market_fit_text(profile)
    metadata = profile.get("metadata") if isinstance(profile.get("metadata"), dict) else {}
    product_name = str(metadata.get("product_name") or "").strip()
    manufacturer = str(metadata.get("manufacturer") or "").strip()
    main_anchor = _build_main_anchor(product_name, manufacturer)

    queries = _iter_queries(plan, max_queries=max_queries)
    if not queries:
        raise HTTPException(status_code=400, detail="No search queries available in analysis plan")
    if product_name:
        raw_name = _clean_text(product_name)
        spaced_name = raw_name.replace("_", " ").strip()
        normalized: List[str] = []
        for q in queries:
            q2 = q
            if raw_name:
                q2 = q2.replace(raw_name, main_anchor)
            if spaced_name and spaced_name != raw_name:
                q2 = q2.replace(spaced_name, main_anchor)
            normalized.append(q2)
        queries = normalized
    queries = list(dict.fromkeys([_clean_text(q) for q in queries if _clean_text(q)]))

    raw_candidates: List[CompetitorCandidate] = []
    used_provider_web_search = False
    reachable_cache_url: Dict[str, bool] = {}
    reachable_cache_domain: Dict[str, bool] = {}
    url_provenance_by_key: Dict[str, Dict[str, str]] = {}

    def _collect_for_queries(query_list: List[str], per_q: int) -> None:
        nonlocal used_provider_web_search
        for q in query_list:
            results: List[Dict[str, str]] = []
            source_label = "web_search_fallback"
            if p == "openai" and openai_key:
                try:
                    results = _openai_search(q, per_q, api_key=openai_key, model=openai_model)
                    used_provider_web_search = True
                    source_label = "web_search_openai"
                except Exception as exc:
                    warnings.append(f"OpenAI web search failed for query '{q}': {exc}")
            elif p == "perplexity" and perplexity_key:
                try:
                    results = _perplexity_search(q, per_q, api_key=perplexity_key, model=perplexity_model)
                    used_provider_web_search = True
                    source_label = "web_search_perplexity"
                except Exception as exc:
                    warnings.append(f"Perplexity web search failed for query '{q}': {exc}")

            if not results:
                try:
                    results = _langsearch_fallback(q, per_q)
                    source_label = "web_search_fallback"
                except Exception as exc:
                    warnings.append(f"Fallback search failed for query '{q}': {exc}")
                    continue

            for r in results:
                title = _clean_text(str(r.get("title") or ""))
                url = str(r.get("url") or "").strip()
                snippet = _clean_text(str(r.get("snippet") or ""))
                if not url:
                    continue
                if _is_placeholder_candidate(name=title, url=url, snippet=snippet):
                    warnings.append(f"Dropped placeholder/garbled candidate: {title or url}")
                    continue
                if _is_self_or_variant_candidate(
                    candidate_name=title,
                    candidate_url=url,
                    candidate_snippet=snippet,
                    product_name=product_name,
                    manufacturer=manufacturer,
                ):
                    warnings.append(f"Dropped self/variant candidate: {title or url}")
                    continue
                cand = _score_candidate(
                    profile_text=profile_text,
                    market_fit_text=market_fit_text,
                    title=title,
                    snippet=snippet,
                    url=url,
                    source_query=q,
                    dimensions=dimensions,
                )
                raw_candidates.append(cand)
                ck = _candidate_key(cand)
                cu = _clean_url(cand.url)
                if ck and cu:
                    prov = url_provenance_by_key.setdefault(ck, {})
                    prov.setdefault(cu, source_label)

    def _filter_rank(cands: List[CompetitorCandidate], warn_on_drop: bool) -> List[CompetitorCandidate]:
        deduped = _dedupe_candidates(cands)
        filtered: List[CompetitorCandidate] = []
        for c in deduped:
            if _is_low_trust_url(c.url):
                if warn_on_drop:
                    warnings.append(f"Dropped low-trust source candidate: {c.name} ({_domain(c.url)})")
                continue
            if manufacturer and _is_same_brand_candidate(
                candidate_name=c.name,
                candidate_url=c.url,
                manufacturer=manufacturer,
            ):
                if warn_on_drop:
                    warnings.append(f"Dropped same-brand candidate: {c.name}")
                continue
            if c.cluster == "media" and c.relevance_score < 0.24:
                continue
            if c.cluster == "marketplace" and c.relevance_score < 0.28:
                continue
            if c.cluster == "video" and c.relevance_score < 0.32:
                continue
            if c.relevance_score < 0.055 and c.similarity_score < 0.05:
                continue

            c.url = _clean_url(c.url)
            d = _domain(c.url)
            url_key = c.url.lower()
            if url_key not in reachable_cache_url:
                reachable_cache_url[url_key] = _is_url_reachable(c.url)
            if not reachable_cache_url[url_key]:
                if d not in reachable_cache_domain:
                    reachable_cache_domain[d] = _is_domain_root_reachable(d)
                # Only strong-penalize when domain root is also not reachable.
                if not reachable_cache_domain[d]:
                    penalty = 0.85
                    if c.cluster == "media":
                        penalty = 0.75
                    elif c.cluster == "marketplace":
                        penalty = 0.68
                    elif c.cluster == "video":
                        penalty = 0.62
                    c.relevance_score = round(float(c.relevance_score) * penalty, 4)
                    if warn_on_drop:
                        warnings.append(f"Unreachable domain penalized (not dropped): {c.name} ({d})")
                    if c.relevance_score < 0.045 and c.similarity_score < 0.04:
                        continue
                else:
                    # Path likely broken but domain alive: keep with mild penalty.
                    c.relevance_score = round(float(c.relevance_score) * 0.95, 4)
            filtered.append(c)
        return sorted(filtered, key=lambda c: (c.relevance_score, c.similarity_score), reverse=True)

    _collect_for_queries(queries, per_query_results)
    min_comp = int(plan.get("min_competitors") or 5)
    candidates = _filter_rank(raw_candidates, warn_on_drop=True)

    if len(candidates) < min_comp:
        extra_queries = _expand_queries_for_undercoverage(base_queries=queries, plan=plan, profile=profile)
        if extra_queries:
            warnings.append(
                f"Under coverage after first pass ({len(candidates)}/{min_comp}); running second pass with {len(extra_queries)} extra queries."
            )
            _collect_for_queries(extra_queries, min(10, per_query_results + 2))
            queries = list(dict.fromkeys(queries + extra_queries))
            candidates = _filter_rank(raw_candidates, warn_on_drop=False)

    target_count = max(min_comp, min(shortlist_size, 50))
    shortlist = candidates[:target_count]
    shortlist = _collapse_near_duplicate_names(shortlist)
    shortlist = sorted(shortlist, key=lambda c: (c.relevance_score, c.similarity_score), reverse=True)[:target_count]

    # Soft-repair with URL candidate pools:
    # collect 2-3 URL options per competitor and choose the best at the end.
    url_pool_by_key: Dict[str, List[str]] = {}
    ranked_raw = sorted(raw_candidates, key=lambda x: (x.relevance_score, x.similarity_score), reverse=True)
    for rc in ranked_raw:
        k = _candidate_key(rc)
        if not k:
            continue
        cu = _clean_url(rc.url)
        if not cu:
            continue
        pool = url_pool_by_key.setdefault(k, [])
        if cu not in pool:
            pool.append(cu)
        if len(pool) > 6:
            url_pool_by_key[k] = pool[:6]

    usable_cache: Dict[str, bool] = {}

    # Final URL selection pass.
    for c in shortlist:
        k = _candidate_key(c)
        options: List[str] = []
        seen_options: set[str] = set()

        def _add_option(u: str) -> None:
            cu = _clean_url(u)
            if not cu or cu in seen_options:
                return
            if _is_low_trust_url(cu):
                return
            seen_options.add(cu)
            options.append(cu)

        _add_option(c.url)
        for u in url_pool_by_key.get(k, []):
            _add_option(u)
        for u, src in _find_replacement_urls(
            cand=c,
            all_candidates=raw_candidates,
            provider=p,
            openai_key=openai_key,
            openai_model=openai_model,
            perplexity_key=perplexity_key,
            perplexity_model=perplexity_model,
            per_query_results=per_query_results,
            product_name=product_name,
            manufacturer=manufacturer,
        ):
            _add_option(u)
            if k and u:
                prov_map = url_provenance_by_key.setdefault(k, {})
                prov_map.setdefault(_clean_url(u), src)

        # If no clearly usable option is present, enrich via domain-constrained live-search.
        domain = _domain(c.url)
        pre_scored = sorted(
            [(_score_url_option(url=u, candidate=c, reachable_cache=reachable_cache_url, usable_cache=usable_cache), u) for u in options],
            key=lambda t: t[0],
            reverse=True,
        )
        best_pre_score = pre_scored[0][0] if pre_scored else -1.0
        if domain and p in {"openai", "perplexity"} and best_pre_score < 1.0:
            q = f'site:{domain} "{c.name}" technical data specifications datasheet pdf'
            fetched: List[Dict[str, str]] = []
            if p == "openai" and openai_key:
                try:
                    fetched = _openai_search(q, max(4, per_query_results), api_key=openai_key, model=openai_model)
                except Exception:
                    fetched = []
            elif p == "perplexity" and perplexity_key:
                try:
                    fetched = _perplexity_search(q, max(4, per_query_results), api_key=perplexity_key, model=perplexity_model)
                except Exception:
                    fetched = []
            if not fetched:
                try:
                    fetched = _langsearch_fallback(q, max(4, per_query_results))
                except Exception:
                    fetched = []
            for r in fetched:
                ru = _clean_url(str(r.get("url") or "").strip())
                if not ru:
                    continue
                if _domain(ru) != domain:
                    continue
                title = _clean_text(str(r.get("title") or ""))
                snip = _clean_text(str(r.get("snippet") or ""))
                if not _name_url_consistent(c.name, ru, title, snip):
                    continue
                _add_option(ru)
                if k and ru:
                    prov_map = url_provenance_by_key.setdefault(k, {})
                    prov_map.setdefault(_clean_url(ru), "domain_live_search")

        # External safety-net: if same-domain options remain unusable, add up to 2
        # trusted external sources (manufacturer-neutral).
        rescored = sorted(
            [(_score_url_option(url=u, candidate=c, reachable_cache=reachable_cache_url, usable_cache=usable_cache), u) for u in options],
            key=lambda t: t[0],
            reverse=True,
        )
        has_usable_same_domain = False
        for _sc, u in rescored:
            if _domain(u) != domain:
                continue
            u_key = f"{u.lower()}|{_name_key(c.name)}"
            is_usable = usable_cache.get(u_key)
            if is_usable is None:
                is_usable = _is_url_usable_for_candidate(u, c.name)
                usable_cache[u_key] = is_usable
            if is_usable:
                has_usable_same_domain = True
                break

        if not has_usable_same_domain:
            ext_query = f'"{c.name}" technical data specs test'
            ext_results: List[Dict[str, str]] = []
            if p == "openai" and openai_key:
                try:
                    ext_results = _openai_search(ext_query, max(6, per_query_results), api_key=openai_key, model=openai_model)
                except Exception:
                    ext_results = []
            elif p == "perplexity" and perplexity_key:
                try:
                    ext_results = _perplexity_search(ext_query, max(6, per_query_results), api_key=perplexity_key, model=perplexity_model)
                except Exception:
                    ext_results = []
            if not ext_results:
                try:
                    ext_results = _langsearch_fallback(ext_query, max(6, per_query_results))
                except Exception:
                    ext_results = []

            added_external = 0
            for r in ext_results:
                ru = _clean_url(str(r.get("url") or "").strip())
                if not ru:
                    continue
                if not _is_trusted_external_url(ru, domain):
                    continue
                title = _clean_text(str(r.get("title") or ""))
                snip = _clean_text(str(r.get("snippet") or ""))
                if not _name_url_consistent_strict(c.name, ru, title, snip):
                    continue
                _add_option(ru)
                if k:
                    prov_map = url_provenance_by_key.setdefault(k, {})
                    prov_map.setdefault(_clean_url(ru), "external_fallback_search")
                added_external += 1
                if added_external >= 2:
                    break

        scored = sorted(
            [(_score_url_option(url=u, candidate=c, reachable_cache=reachable_cache_url, usable_cache=usable_cache), u) for u in options],
            key=lambda t: t[0],
            reverse=True,
        )
        if scored:
            # Prefer first usable URL; if none, keep best scored fallback.
            best_score, best_url = scored[0]
            preferred_usable: Optional[Tuple[float, str]] = None
            for sc, u in scored:
                u_key = f"{u.lower()}|{_name_key(c.name)}"
                is_usable = usable_cache.get(u_key)
                if is_usable is None:
                    is_usable = _is_url_usable_for_candidate(u, c.name)
                    usable_cache[u_key] = is_usable
                if not is_usable:
                    continue
                # Avoid model drift: require minimum model-token overlap.
                if _url_model_match_score(c.name, u) >= 0.34:
                    preferred_usable = (sc, u)
                    break
                if preferred_usable is None:
                    preferred_usable = (sc, u)
            if preferred_usable is not None:
                best_score, best_url = preferred_usable
            old = c.url
            c.url = best_url
            c.url_candidates = [u for _, u in scored[:3]]
            prov_map = url_provenance_by_key.get(k, {}) if k else {}
            c.url_provenance = {}
            for u in c.url_candidates:
                c.url_provenance[u] = prov_map.get(_clean_url(u), "unknown")
            c.url_provenance[c.url] = prov_map.get(_clean_url(c.url), c.url_provenance.get(c.url, "unknown"))
            if best_url != old:
                c.reasons.append("URL aus Kandidatenpool gewählt")
                warnings.append(f"Selected best URL for '{c.name}': {old} -> {best_url} (score={best_score:.2f})")
        else:
            c.url = _clean_url(c.url)
            c.url_candidates = [c.url] if c.url else []
            c.url_provenance = {c.url: "unknown"} if c.url else {}

    if p in {"openai", "perplexity"} and not used_provider_web_search:
        warnings.append(f"{p.capitalize()} web search was not used; fallback search provider was applied.")

    if len(shortlist) < min_comp:
        warnings.append(
            f"Only {len(shortlist)} competitors found, below target min_competitors={min_comp}."
        )

    warnings = list(dict.fromkeys([w.strip() for w in warnings if str(w).strip()]))

    return CompetitorList(
        provider=p,
        generated_queries=queries,
        min_competitors_target=min_comp,
        competitors=shortlist,
        extraction_warnings=warnings,
    )
