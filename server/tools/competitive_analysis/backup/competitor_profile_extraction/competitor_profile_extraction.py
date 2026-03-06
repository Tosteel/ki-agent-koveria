from __future__ import annotations

import json
import io
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from fastapi import HTTPException

from .models import (
    CompetitorSourceRegistry,
    CompetitorProfile,
    CompetitorProfiles,
    DataQuality,
    MappedFeature,
    PriceInfo,
    SourceRegistryCompetitor,
    SourceRegistryEntry,
    SourceEvidence,
)

OPENAI_URL = os.getenv("OPENAI_URL", "https://api.openai.com/v1/responses").strip() or "https://api.openai.com/v1/responses"
PERPLEXITY_URL = (
    os.getenv("PERPLEXITY_URL", "https://api.perplexity.ai/chat/completions").strip()
    or "https://api.perplexity.ai/chat/completions"
)
_USER_AGENT = "Mozilla/5.0 (compatible; KI-Agent-Koveria/1.0; +https://example.local)"
_DEFAULT_SCHEMA = [
    "Leistung",
    "Kapazität",
    "Druck",
    "Durchmesser",
    "Abmessungen",
    "Gewicht",
    "Temperaturbereich",
    "Effizienz",
    "Compliance",
]

_UNIT_NORMALIZATION = {
    "kw": (1000.0, "W"),
    "w": (1.0, "W"),
    "kg": (1.0, "kg"),
    "g": (0.001, "kg"),
    "m": (1.0, "m"),
    "cm": (0.01, "m"),
    "mm": (0.001, "m"),
    "l": (1.0, "L"),
    "ml": (0.001, "L"),
    "bar": (1.0, "bar"),
    "v": (1.0, "V"),
    "a": (1.0, "A"),
    "hz": (1.0, "Hz"),
}

_FEATURE_HINTS = {
    "leistung": "Leistung",
    "power": "Leistung",
    "kapaz": "Kapazität",
    "capacity": "Kapazität",
    "druck": "Druck",
    "pressure": "Druck",
    "durchmesser": "Durchmesser",
    "diameter": "Durchmesser",
    "maß": "Abmessungen",
    "abmess": "Abmessungen",
    "dimension": "Abmessungen",
    "gewicht": "Gewicht",
    "weight": "Gewicht",
    "temperatur": "Temperaturbereich",
    "temp": "Temperaturbereich",
    "effizienz": "Effizienz",
    "efficiency": "Effizienz",
    "ce": "Compliance",
    "iso": "Compliance",
    "rohs": "Compliance",
    "ul": "Compliance",
    "din": "Compliance",
    "length": "Abmessungen",
    "width": "Abmessungen",
    "height": "Abmessungen",
    "torque": "Leistung",
    "speed": "Leistung",
    "verbrauch": "Effizienz",
    "consumption": "Effizienz",
    "co2": "Effizienz",
    "emission": "Effizienz",
}

_PRICE_RE = re.compile(
    r"(?P<raw>(?:(?:EUR|USD|CHF|GBP)\s*)?[0-9]{1,3}(?:[.,][0-9]{3})*(?:[.,][0-9]{1,2})?\s*(?:EUR|USD|CHF|GBP|€|\$|£))",
    re.IGNORECASE,
)

_DATE_RES = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{2}\.\d{2}\.\d{4}\b"),
    re.compile(r"\b\d{2}/\d{2}/\d{4}\b"),
]

_NON_TECH_KEYS = {
    "tel",
    "telefon",
    "phone",
    "fax",
    "kontakt",
    "contact",
    "impressum",
    "adresse",
    "address",
    "e-mail",
    "email",
    "herausgeber",
    "publisher",
    "lesezeit",
    "copyright",
    "datenschutz",
    "privacy",
    "cookie",
}

_MEDIA_DOMAIN_HINTS = (
    "wikipedia.org",
    "linkedin.com",
    "facebook.com",
    "instagram.com",
    "youtube.com",
    "x.com",
    "twitter.com",
    "news",
    "blog",
    "magazin",
    "heise.de",
    "ikz.de",
)

_PRODUCT_URL_HINTS = (
    "product",
    "produkte",
    "produkt",
    "datasheet",
    "datenblatt",
    "spec",
    "technical",
    "pdf",
    "catalog",
    "katalog",
)

_MIN_USABLE_FEATURES = 5
_MIN_PROFILE_CONFIDENCE = 0.30

_UNUSABLE_PAGE_MARKERS = (
    "site maintenance",
    "oops! something went wrong",
    "please contact your administrator",
    "error code",
    "access denied",
    "forbidden",
    "temporarily unavailable",
    "service unavailable",
)

_NOISY_LINE_MARKERS = (
    "cookie-einstellungen",
    "datenschutz",
    "privacy",
    "newsletter",
    "kontakt",
    "nothilfe",
    "mein adac",
    "tariffinder",
    "routenplanung",
    "mitgliedschaft",
)

_PRICE_CONTEXT_HINTS = (
    "preis",
    "price",
    "ab ",
    "starting",
    "from ",
    "uvp",
    "msrp",
    "list price",
    "basispreis",
    "grundpreis",
    "modellpreis",
)

_PRICE_NOISE_HINTS = (
    "sonderausstattung",
    "option",
    "paket",
    "in verbindung",
    "mitglied",
    "membership",
    "tarif",
    "serviceplan",
)

_COMPETITOR_PRODUCT_HINTS = (
    "e-bike",
    "ebike",
    "pedelec",
    "electro bike",
    "electric bike",
    "faltrad",
    "klapprad",
    "folding bike",
)

_NON_PRODUCT_URL_HINTS = (
    "/news/",
    "/blog/",
    "/forum/",
    "/jobs",
    "/career",
    "/karriere",
    "/impressum",
    "/datenschutz",
    "/privacy",
    "/kontakt",
    "/about",
    "auto-motor-und-sport.de",
)

_BOILERPLATE_LINE_HINTS = (
    "zum inhalt springen",
    "skip to content",
    "mein konto",
    "warenkorb",
    "wishlist",
    "cookie",
    "datenschutz",
    "privacy",
    "newsletter",
    "kontakt",
    "faq",
    "agb",
    "widerruf",
    "lieferung",
    "shipping",
    "zahlung",
    "payment",
    "anmelden",
    "login",
    "register",
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


def _parse_localized_number(num: str) -> float | None:
    s = re.sub(r"[^\d,.\-]", "", str(num or "").strip())
    if not s or s in {"-", ".", ","}:
        return None
    if "," in s and "." in s:
        # Use right-most punctuation as decimal separator, remove the other as thousand sep.
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        # Decimal comma when ending with 1-2 digits, otherwise thousands comma.
        s = s.replace(",", ".") if re.search(r",\d{1,2}$", s) else s.replace(",", "")
    elif "." in s:
        # If dot is thousands separator (groups of 3), remove dots.
        s = s.replace(".", "") if re.search(r"\.\d{3}(\.|$)", s) else s
    try:
        return float(s)
    except Exception:
        return None


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


def _load_competitor_list(
    *,
    competitor_list: Optional[Dict[str, Any]],
    competitor_list_path: Optional[str],
    user_root: Path,
    work_root: Path,
) -> Dict[str, Any]:
    payload: Dict[str, Any]
    if isinstance(competitor_list, dict) and competitor_list:
        payload = competitor_list
    else:
        p = _resolve_input_path(str(competitor_list_path or ""), user_root=user_root.resolve(), work_root=work_root.resolve())
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON in competitor_list_path: {competitor_list_path}") from exc

    if "competitor_list" in payload and isinstance(payload.get("competitor_list"), dict):
        payload = payload["competitor_list"]

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid competitor_list payload")
    return payload


def _load_json_path_dict(path: str, *, user_root: Path, work_root: Path) -> Dict[str, Any]:
    p = _resolve_input_path(path, user_root=user_root.resolve(), work_root=work_root.resolve())
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in path: {path}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail=f"Invalid JSON root for path: {path}")
    return payload


def _normalize_name_key(name: str) -> str:
    n = re.sub(r"[^a-z0-9]+", " ", str(name or "").lower()).strip()
    n = re.sub(r"\s+", " ", n)
    return n


def _load_source_registry(
    *,
    source_registry: Optional[Dict[str, Any]],
    source_registry_path: Optional[str],
    user_root: Path,
    work_root: Path,
) -> Dict[str, Any]:
    payload: Dict[str, Any]
    if isinstance(source_registry, dict) and source_registry:
        payload = source_registry
    elif (source_registry_path or "").strip():
        payload = _load_json_path_dict(source_registry_path or "", user_root=user_root, work_root=work_root)
    else:
        return {}

    if "source_registry" in payload and isinstance(payload.get("source_registry"), dict):
        payload = payload["source_registry"]
    if not isinstance(payload, dict):
        return {}
    return payload


def _extract_registry_urls_for_competitor(
    registry: Dict[str, Any],
    competitor_name: str,
    *,
    active_only: bool = True,
    max_urls: int = 8,
) -> List[str]:
    rows = registry.get("competitors") if isinstance(registry.get("competitors"), list) else []
    key = _normalize_name_key(competitor_name)
    matched: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if _normalize_name_key(str(row.get("name") or "")) == key:
            matched.append(row)

    urls: List[Tuple[int, str]] = []
    for row in matched:
        entries = row.get("entries") if isinstance(row.get("entries"), list) else []
        for ent in entries:
            if isinstance(ent, str):
                u = ent.strip()
                if u.startswith("http"):
                    urls.append((50, u))
                continue
            if not isinstance(ent, dict):
                continue
            u = str(ent.get("url") or "").strip()
            if not u.startswith("http"):
                continue
            if active_only and bool(ent.get("active")) is False:
                continue
            prio = int(ent.get("priority") or 50)
            urls.append((prio, u))

    if not urls:
        return []
    urls_sorted = [u for _p, u in sorted(urls, key=lambda t: t[0])]
    return _dedupe_urls(urls_sorted)[:max_urls]


def _openai_web_search_urls(query: str, api_key: str, model: str, max_results: int) -> List[str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": f"Finde relevante Produktseiten und Datenblätter für: {query}"}],
            }
        ],
        "tools": [{"type": "web_search_preview"}],
        "tool_choice": {"type": "web_search_preview"},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "url_list",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "urls": {
                            "type": "array",
                            "items": {"type": "string"},
                        }
                    },
                    "required": ["urls"],
                },
                "strict": False,
            }
        },
        "include": ["web_search_call.action.sources"],
    }

    r = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"OpenAI web search HTTP {r.status_code}: {r.text}")

    data = r.json()

    urls: List[str] = []
    text = ""
    for item in data.get("output", []):
        for c in item.get("content", []):
            if c.get("type") == "output_text":
                text += str(c.get("text") or "")
    if text:
        try:
            parsed = json.loads(text)
            raw_urls = parsed.get("urls") if isinstance(parsed, dict) else []
            if isinstance(raw_urls, list):
                for u in raw_urls:
                    us = str(u or "").strip()
                    if us.startswith("http"):
                        urls.append(us)
        except Exception:
            pass

    if not urls:
        for item in data.get("output", []):
            if item.get("type") == "web_search_call":
                action = item.get("action") if isinstance(item.get("action"), dict) else {}
                sources = action.get("sources") if isinstance(action, dict) else None
                if isinstance(sources, list):
                    for s in sources:
                        if not isinstance(s, dict):
                            continue
                        u = str(s.get("url") or "").strip()
                        if u.startswith("http"):
                            urls.append(u)

    deduped = []
    seen = set()
    for u in urls:
        k = u.lower()
        if k in seen:
            continue
        seen.add(k)
        deduped.append(u)
    return deduped[:max_results]


def _perplexity_web_search_urls(query: str, api_key: str, model: str, max_results: int) -> List[str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Liefere ausschließlich JSON: "
                    "{\"urls\":[\"https://...\"]}. "
                    "Nur reale Produkt-/Datenblattseiten, keine erfundenen URLs."
                ),
            },
            {"role": "user", "content": f"Finde relevante Produktseiten und Datenblätter für: {query}"},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "url_list",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "urls": {
                            "type": "array",
                            "items": {"type": "string"},
                        }
                    },
                    "required": ["urls"],
                },
            },
        },
    }
    r = requests.post(PERPLEXITY_URL, headers=headers, json=payload, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"Perplexity web search HTTP {r.status_code}: {r.text}")
    data = r.json()
    text = ""
    try:
        text = str(data["choices"][0]["message"]["content"] or "")
    except Exception:
        text = ""

    urls: List[str] = []
    if text:
        try:
            parsed = json.loads(text)
            raw_urls = parsed.get("urls") if isinstance(parsed, dict) else []
            if isinstance(raw_urls, list):
                for u in raw_urls:
                    us = str(u or "").strip()
                    if us.startswith("http"):
                        urls.append(us)
        except Exception:
            pass
    if not urls and text:
        for u in re.findall(r"https?://[^\s\"'<>]+", text):
            urls.append(u.strip())

    deduped: List[str] = []
    seen: set[str] = set()
    for u in urls:
        k = u.lower()
        if k in seen:
            continue
        seen.add(k)
        deduped.append(u)
        if len(deduped) >= max_results:
            break
    return deduped[:max_results]


def _openai_domain_constrained_urls(
    *,
    competitor_name: str,
    domain: str,
    api_key: str,
    model: str,
    max_results: int = 8,
) -> List[str]:
    d = str(domain or "").strip().lower().replace("www.", "")
    if not d:
        return []
    q = f"site:{d} \"{competitor_name}\" technical data specifications datasheet pdf"
    return _openai_web_search_urls(q, api_key=api_key, model=model, max_results=max_results)


def _perplexity_domain_constrained_urls(
    *,
    competitor_name: str,
    domain: str,
    api_key: str,
    model: str,
    max_results: int = 8,
) -> List[str]:
    d = str(domain or "").strip().lower().replace("www.", "")
    if not d:
        return []
    q = f"site:{d} \"{competitor_name}\" technical data specifications datasheet pdf"
    return _perplexity_web_search_urls(q, api_key=api_key, model=model, max_results=max_results)


def _http_get_with_retries(url: str, timeout_s: int = 40, retries: int = 2) -> requests.Response:
    headers = {
        "User-Agent": _USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8",
        "Accept-Language": "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7",
        "Connection": "keep-alive",
    }
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout_s, allow_redirects=True)
            r.raise_for_status()
            return r
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(0.8 * (2 ** attempt))
                continue
            raise last_exc


def _parse_pdf_bytes(pdf_bytes: bytes, max_chars: int = 120000) -> str:
    try:
        from pypdf import PdfReader  # type: ignore

        reader_obj = PdfReader(io.BytesIO(pdf_bytes))
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore

            reader_obj = PdfReader(io.BytesIO(pdf_bytes))
        except Exception:
            return ""

    chunks: list[str] = []
    remaining = max(0, int(max_chars))
    for page in reader_obj.pages:
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        if not txt:
            continue
        if len(txt) > remaining:
            chunks.append(txt[:remaining])
            break
        chunks.append(txt)
        remaining -= len(txt)
        if remaining <= 0:
            break
    return "\n".join(chunks).strip()


def _fetch_page(url: str) -> Tuple[str, str, str, str]:
    r = _http_get_with_retries(url, timeout_s=45, retries=2)
    content_type = str(r.headers.get("Content-Type") or "").lower()
    body_head = r.content[:8] if isinstance(r.content, (bytes, bytearray)) else b""
    is_pdf_bytes = body_head.startswith(b"%PDF-")
    is_pdf_ct = "application/pdf" in content_type
    is_pdf_url = url.lower().endswith(".pdf")

    if is_pdf_ct and not is_pdf_bytes:
        raise RuntimeError("Declared PDF response does not contain valid PDF bytes")

    if is_pdf_bytes or (is_pdf_url and is_pdf_ct):
        text = _parse_pdf_bytes(r.content)
        if text:
            title = Path(urlparse(url).path).name or "PDF"
            return title[:220], text[:120000], "", content_type or "application/pdf"
        # If PDF parsing failed despite PDF-like response, continue as unsupported.
        raise RuntimeError("PDF content could not be parsed")

    # Common case: URL looks like .pdf but returns HTML/error page.
    if is_pdf_url and not is_pdf_ct and not is_pdf_bytes:
        if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
            raise RuntimeError(f"URL looks like PDF but content is not PDF/HTML: {content_type or 'unknown'}")
        # Fall through to HTML parsing.

    if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
        raise RuntimeError(f"Unsupported content-type for crawling: {content_type or 'unknown'}")

    html = r.text
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.string or "").strip() if soup.title and soup.title.string else ""
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    text = re.sub(r"\n{2,}", "\n", text)

    low = (title + "\n" + text[:3000]).lower()
    if any(marker in low for marker in _UNUSABLE_PAGE_MARKERS):
        raise RuntimeError("Page appears to be maintenance/error content")

    return title[:220], text[:120000], html[:200000], content_type


def _fallback_urls_for_competitor(name: str, base_url: str) -> List[str]:
    # Manufacturer-neutral fallback: stay on candidate domain only.
    out: List[str] = []
    bu = str(base_url or "").strip()
    if bu:
        out.append(bu)
        try:
            p = urlparse(bu)
            if p.scheme and p.netloc:
                out.append(f"{p.scheme}://{p.netloc}")
        except Exception:
            pass
    return _dedupe_urls(out)


def _extract_feature_pairs_from_text(text: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for line in (text or "").splitlines():
        ln = line.strip()
        if not ln or len(ln) < 4:
            continue
        if ":" in ln:
            left, right = ln.split(":", 1)
            k = left.strip()
            v = right.strip()
            if k and v and len(k) <= 80 and len(v) <= 200:
                pairs.append((k, v))
    return pairs


def _extract_feature_pairs_from_html_tables(html: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    if not html:
        return pairs
    soup = BeautifulSoup(html, "html.parser")
    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
            if len(cells) >= 2:
                key = cells[0].strip()
                val = cells[1].strip()
                if key and val:
                    pairs.append((key, val))
    return pairs


def _garbled_ratio(s: str) -> float:
    if not s:
        return 1.0
    allowed = 0
    for ch in s:
        o = ord(ch)
        if ch in "\n\r\t" or 32 <= o <= 126 or ch in "äöüÄÖÜß€°²³–—-/%()[],:.;+_&'\"":
            allowed += 1
    return 1.0 - (allowed / max(1, len(s)))


def _is_valid_feature_pair(raw_name: str, raw_val: str) -> bool:
    k = str(raw_name or "").strip()
    v = str(raw_val or "").strip()
    if not k or not v:
        return False
    if len(k) > 80 or len(v) > 220:
        return False
    if len(k) < 2 or len(v) < 1:
        return False
    if _garbled_ratio(k) > 0.25 or _garbled_ratio(v) > 0.30:
        return False
    k_low = k.lower().strip(" :.-")
    if k_low in _NON_TECH_KEYS:
        return False
    if any(tok in k_low for tok in ("impressum", "privacy", "datenschutz", "cookie", "kontakt", "newsletter", "agb")):
        return False
    if "http" in k_low or "www." in k_low or "eprel" in k_low or "qr/" in k_low:
        return False
    if any(tok in k_low for tok in ("telefon", "phone", "tel.", "hotline", "vat", "register", "part number")):
        return False
    if any(tok in str(v).lower() for tok in ("http://", "https://", "error code", "please contact your administrator")):
        return False
    # Avoid full sentence fragments as keys.
    if len(re.findall(r"[A-Za-zÄÖÜäöüß]+", k)) > 8:
        return False
    # Keep primarily technical key/value pairs.
    has_tech_hint = any(h in k_low for h in _FEATURE_HINTS.keys())
    has_numeric = bool(re.search(r"\d", v))
    has_measure_unit = bool(re.search(r"\b(kW|PS|Nm|kg|mm|cm|m|kWh|Wh|bar|°C|km/h|g/km|l/100)\b", v, flags=re.IGNORECASE))
    if not has_tech_hint and not has_numeric:
        return False
    if not has_tech_hint and has_numeric and not has_measure_unit:
        return False
    if re.fullmatch(r"[0-9\W_]+", k):
        return False
    return True


def _extract_numeric_unit(value: str) -> Tuple[str, str, float | int | str | None, str]:
    v = str(value or "").strip()
    m = re.search(r"(-?\d+(?:[\.,]\d+)?)\s*([A-Za-z%°/²³]+)?", v)
    if not m:
        return v, "", None, ""
    raw_num = m.group(1)
    unit = (m.group(2) or "").strip()
    parsed = _parse_localized_number(raw_num)
    if parsed is None:
        return v, unit, None, unit
    numeric: float | int = int(parsed) if float(parsed).is_integer() else parsed

    unit_key = unit.lower()
    unit_key = unit_key.replace("²", "2").replace("³", "3")
    if unit_key in _UNIT_NORMALIZATION:
        factor, norm_unit = _UNIT_NORMALIZATION[unit_key]
        norm = float(numeric) * factor
        normalized = int(norm) if norm.is_integer() else round(norm, 6)
        return v, unit, normalized, norm_unit

    return v, unit, numeric, unit


def _map_to_schema(raw_name: str, target_schema: List[str]) -> str:
    low = str(raw_name or "").lower()
    for hint, mapped in _FEATURE_HINTS.items():
        if hint in low:
            return mapped

    # fallback to best token overlap with target schema
    r_tokens = set(re.findall(r"[a-zA-ZäöüÄÖÜß]+", low))
    best = ""
    best_score = 0
    for sf in target_schema:
        s_low = sf.lower()
        s_tokens = set(re.findall(r"[a-zA-ZäöüÄÖÜß]+", s_low))
        score = len(r_tokens & s_tokens)
        if score > best_score:
            best_score = score
            best = sf
    return best or "Other"


def _url_domain(url: str) -> str:
    return (urlparse(str(url or "")).netloc or "").lower().replace("www.", "")


def _url_path(url: str) -> str:
    return (urlparse(str(url or "")).path or "").lower()


def _url_relevance_score(url: str, competitor_name: str) -> float:
    domain = _url_domain(url)
    path = _url_path(url)
    score = 0.0
    if any(h in domain for h in _MEDIA_DOMAIN_HINTS):
        score -= 1.3
    if any(h in domain for h in ("shop", "store", "amazon", "ebay")):
        score -= 1.4
    if any(h in path for h in _PRODUCT_URL_HINTS):
        score += 1.2
    if "/product" in path or "/produkte" in path or "/products" in path:
        score += 0.8
    if competitor_name:
        name_token = re.sub(r"[^a-z0-9]+", "", competitor_name.lower())
        if name_token and name_token[:8] in re.sub(r"[^a-z0-9]+", "", domain + path):
            score += 0.6
    return score


def _extract_prices(text: str, source_url: str) -> List[PriceInfo]:
    out: List[PriceInfo] = []
    src = text or ""
    for m in _PRICE_RE.finditer(src):
        raw = m.group("raw")
        lower = raw.lower()

        currency = ""
        if "€" in raw or "eur" in lower:
            currency = "EUR"
        elif "$" in raw or "usd" in lower:
            currency = "USD"
        elif "£" in raw or "gbp" in lower:
            currency = "GBP"
        elif "chf" in lower:
            currency = "CHF"

        num_match = re.search(r"[0-9]{1,3}(?:[.,][0-9]{3})*(?:[.,][0-9]{1,2})?|[0-9]+(?:[.,][0-9]{1,2})?", raw)
        val: float | int | None = None
        if num_match:
            parsed = _parse_localized_number(num_match.group(0))
            if parsed is not None:
                f = float(parsed)
                val = int(f) if f.is_integer() else f
            # drop obvious artifacts (single-digit/very tiny prices) for B2B/vehicle context
            if isinstance(val, (int, float)) and val <= 2:
                continue

        # Keep prices with strong context, otherwise require a meaningful magnitude.
        ctx = src[max(0, m.start() - 60) : min(len(src), m.end() + 60)].lower()
        has_price_ctx = any(h in ctx for h in _PRICE_CONTEXT_HINTS)
        has_noise_ctx = any(h in ctx for h in _PRICE_NOISE_HINTS)
        if has_noise_ctx and not has_price_ctx:
            continue
        if isinstance(val, (int, float)):
            if val < 50:
                continue
            if not has_price_ctx and val < 10000:
                continue

        out.append(PriceInfo(raw=raw, value=val, currency=currency, source_url=source_url))

    # dedupe
    dedup: Dict[str, PriceInfo] = {}
    for p in out:
        key = f"{p.raw.lower()}|{p.value}|{p.currency}|{p.source_url}"
        dedup[key] = p
    return list(dedup.values())


def _is_unusable_content(title: str, text: str) -> bool:
    low = f"{title}\n{text[:5000]}".lower()
    if any(m in low for m in _UNUSABLE_PAGE_MARKERS):
        return True

    # Navigation/cookie heavy pages with barely any measurable specs are not useful.
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return True
    noisy = sum(1 for ln in lines[:120] if any(m in ln.lower() for m in _NOISY_LINE_MARKERS))
    numeric_like = sum(1 for ln in lines[:200] if re.search(r"\d", ln) and ":" in ln)
    if noisy >= 8 and numeric_like <= 1:
        return True
    return False


def _detect_freshness_days(text_blobs: Iterable[str]) -> int | None:
    now = datetime.now(timezone.utc).date()
    best: Optional[int] = None
    for text in text_blobs:
        if not text:
            continue
        for rgx in _DATE_RES:
            for m in rgx.finditer(text):
                raw = m.group(0)
                for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y"):
                    try:
                        dt = datetime.strptime(raw, fmt).date()
                    except ValueError:
                        continue
                    days = (now - dt).days
                    if days < 0:
                        continue
                    if best is None or days < best:
                        best = days
    return best


def _build_data_quality(
    *,
    mapped_features: List[MappedFeature],
    prices: List[PriceInfo],
    sources: List[SourceEvidence],
    target_schema: List[str],
    freshness_days: int | None,
) -> DataQuality:
    feature_keys = {f.schema_feature.lower() for f in mapped_features if f.schema_feature}
    target = {s.lower() for s in target_schema if s}
    completeness = 0.0
    if target:
        completeness = min(1.0, len(feature_keys & target) / len(target))

    source_factor = min(1.0, len(sources) / 3.0)
    price_factor = 0.15 if prices else 0.0
    feature_factor = min(1.0, len(mapped_features) / 10.0)

    freshness_factor = 0.0
    notes: List[str] = []
    if freshness_days is None:
        notes.append("Keine belastbare Aktualitätsangabe gefunden.")
    else:
        freshness_factor = 1.0 if freshness_days <= 365 else (0.7 if freshness_days <= 730 else 0.4)

    confidence = 0.45 * feature_factor + 0.25 * source_factor + 0.20 * completeness + 0.10 * freshness_factor + price_factor
    confidence = max(0.0, min(1.0, confidence))

    if len(mapped_features) < 3:
        notes.append("Wenige technische Spezifikationen extrahiert.")
    if not prices:
        notes.append("Keine Preise/Pakete auffindbar.")

    return DataQuality(
        confidence=round(confidence, 4),
        completeness=round(completeness, 4),
        freshness_days=freshness_days,
        notes=notes,
    )


def _dedupe_urls(urls: List[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for u in urls:
        us = str(u or "").strip()
        if not us:
            continue
        k = us.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(us)
    return out


def _usable_feature_count(features: List[MappedFeature]) -> int:
    count = 0
    for f in features:
        if f.schema_feature != "Other":
            count += 1
        elif f.normalized_value is not None:
            count += 1
    return count


def _reduce_noise_features(features: List[MappedFeature], max_other: int = 12) -> List[MappedFeature]:
    other = [f for f in features if f.schema_feature == "Other"]
    core = [f for f in features if f.schema_feature != "Other"]
    # Keep numeric "Other" first; they are often still useful.
    other_sorted = sorted(other, key=lambda f: (f.normalized_value is None, len(f.raw_name or "")))
    return core + other_sorted[:max_other]


def extract_competitor_profiles(
    *,
    competitor_list: Optional[Dict[str, Any]],
    competitor_list_path: Optional[str],
    source_registry: Optional[Dict[str, Any]] = None,
    source_registry_path: Optional[str] = None,
    provider: str = "openai",
    max_competitors: int = 10,
    max_pages_per_competitor: int = 3,
    offset: int = 0,
    limit: Optional[int] = None,
    verbose_progress: bool = True,
    registry_first: bool = True,
    min_active_sources_for_search: int = 2,
    user_root: Path,
    work_root: Path,
) -> CompetitorProfiles:
    cl = _load_competitor_list(
        competitor_list=competitor_list,
        competitor_list_path=competitor_list_path,
        user_root=user_root,
        work_root=work_root,
    )

    p = str(provider or "openai").strip().lower()
    if p not in {"openai", "ionos", "perplexity"}:
        p = "openai"

    warnings = _safe_list_str(cl.get("extraction_warnings"))
    source_registry_payload = _load_source_registry(
        source_registry=source_registry,
        source_registry_path=source_registry_path,
        user_root=user_root,
        work_root=work_root,
    )

    target_schema = _safe_list_str(cl.get("target_feature_schema"))
    if not target_schema:
        target_schema = list(_DEFAULT_SCHEMA)

    all_candidates = cl.get("competitors") if isinstance(cl.get("competitors"), list) else []
    total_candidates = len(all_candidates)
    start = max(0, int(offset))
    if limit is None:
        stop = min(total_candidates, start + int(max_competitors))
    else:
        stop = min(total_candidates, start + int(limit))
    competitors_raw = all_candidates[start:stop]

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    perplexity_key = os.getenv("PERPLEXITY_API_KEY", "").strip()
    perplexity_model = os.getenv("PERPLEXITY_MODEL", "sonar-pro").strip() or "sonar-pro"

    competitor_profiles: List[CompetitorProfile] = []

    processed = 0
    for idx, comp in enumerate(competitors_raw, start=1):
        if not isinstance(comp, dict):
            continue

        name = str(comp.get("name") or "").strip() or "Unknown Competitor"
        base_url = str(comp.get("url") or "").strip()
        url_candidates = _safe_list_str(comp.get("url_candidates"))
        cluster = str(comp.get("cluster") or "unknown").strip() or "unknown"
        if not base_url:
            warnings.append(f"Competitor '{name}' has no URL and was skipped.")
            continue
        processed += 1
        if verbose_progress:
            print(
                f"[competitive_extract_competitor_profiles] {processed}/{len(competitors_raw)} processing: {name}",
                flush=True,
            )

        registry_urls = _extract_registry_urls_for_competitor(
            source_registry_payload,
            name,
            active_only=True,
            max_urls=max(max_pages_per_competitor * 2, 8),
        )
        seed_urls = _dedupe_urls([base_url] + url_candidates)
        candidate_urls = _fallback_urls_for_competitor(name, base_url)
        candidate_urls = seed_urls + candidate_urls
        if registry_first and registry_urls:
            candidate_urls = registry_urls + candidate_urls
        do_web_search = len(registry_urls) < max(0, int(min_active_sources_for_search))
        if p == "openai" and openai_key and do_web_search:
            try:
                query = f"{name} official technical specifications datasheet pdf"
                extra = _openai_web_search_urls(
                    query,
                    api_key=openai_key,
                    model=openai_model,
                    max_results=max(max_pages_per_competitor * 3, 6),
                )
                candidate_urls.extend(extra)
            except Exception as exc:
                warnings.append(f"OpenAI web search failed for '{name}': {exc}")
        elif p == "perplexity" and perplexity_key and do_web_search:
            try:
                query = f"{name} official technical specifications datasheet pdf"
                extra = _perplexity_web_search_urls(
                    query,
                    api_key=perplexity_key,
                    model=perplexity_model,
                    max_results=max(max_pages_per_competitor * 3, 6),
                )
                candidate_urls.extend(extra)
            except Exception as exc:
                warnings.append(f"Perplexity web search failed for '{name}': {exc}")
        elif p == "openai" and not openai_key:
            warnings.append("OPENAI_API_KEY missing; web search enrichment disabled.")
        elif p == "perplexity" and not perplexity_key:
            warnings.append("PERPLEXITY_API_KEY missing; web search enrichment disabled.")

        urls = _dedupe_urls(candidate_urls)
        urls = sorted(urls, key=lambda u: _url_relevance_score(u, name), reverse=True)[: max(max_pages_per_competitor * 2, 6)]

        sources: List[SourceEvidence] = []
        mapped_features: List[MappedFeature] = []
        prices: List[PriceInfo] = []
        packages: List[str] = []
        page_texts: List[str] = []

        now_iso = datetime.now(timezone.utc).isoformat()
        crawl_queue = list(urls)
        queued = set(u.lower() for u in crawl_queue)
        live_domain_search_done = False
        i = 0
        while i < len(crawl_queue) and len(sources) < max_pages_per_competitor:
            u = crawl_queue[i]
            i += 1
            try:
                title, text, html, content_type = _fetch_page(u)
            except Exception as exc:
                warnings.append(f"Crawling failed for '{name}' url={u}: {exc}")
                # Last-resort fallback: live search constrained to candidate domain.
                if (
                    not live_domain_search_done
                    and p in {"openai", "perplexity"}
                    and _url_domain(base_url)
                ):
                    live_domain_search_done = True
                    try:
                        if p == "openai" and openai_key:
                            live_extra = _openai_domain_constrained_urls(
                                competitor_name=name,
                                domain=_url_domain(base_url),
                                api_key=openai_key,
                                model=openai_model,
                                max_results=max(max_pages_per_competitor * 3, 8),
                            )
                        elif p == "perplexity" and perplexity_key:
                            live_extra = _perplexity_domain_constrained_urls(
                                competitor_name=name,
                                domain=_url_domain(base_url),
                                api_key=perplexity_key,
                                model=perplexity_model,
                                max_results=max(max_pages_per_competitor * 3, 8),
                            )
                        else:
                            live_extra = []
                        added = 0
                        for lu in live_extra:
                            key = lu.lower()
                            if key in queued:
                                continue
                            queued.add(key)
                            crawl_queue.append(lu)
                            added += 1
                        if added:
                            warnings.append(
                                f"Domain-constrained live-search added {added} URLs for '{name}' on domain '{_url_domain(base_url)}'."
                            )
                    except Exception as search_exc:
                        warnings.append(f"Domain-constrained live-search failed for '{name}': {search_exc}")
                continue

            if _is_unusable_content(title, text):
                warnings.append(f"Skipped unusable content for '{name}' url={u}")
                continue

            if _garbled_ratio(text) > 0.35:
                warnings.append(f"Skipped garbled content for '{name}' url={u}")
                continue

            page_texts.append(text)
            excerpt = text[:600]
            sources.append(SourceEvidence(url=u, title=title, retrieved_at=now_iso, excerpt=excerpt))

            feature_pairs = _extract_feature_pairs_from_text(text)
            if "html" in content_type and html:
                feature_pairs.extend(_extract_feature_pairs_from_html_tables(html))

            for raw_name, raw_val in feature_pairs:
                if not _is_valid_feature_pair(raw_name, raw_val):
                    continue
                schema_feature = _map_to_schema(raw_name, target_schema)
                raw_value, unit, normalized_value, normalized_unit = _extract_numeric_unit(raw_val)
                if schema_feature == "Other" and normalized_value is None:
                    continue
                mapped_features.append(
                    MappedFeature(
                        schema_feature=schema_feature,
                        raw_name=raw_name,
                        value=raw_value,
                        unit=unit,
                        normalized_value=normalized_value,
                        normalized_unit=normalized_unit,
                        source_url=u,
                        evidence=f"{raw_name}: {raw_val}"[:260],
                    )
                )

            prices.extend(_extract_prices(text, source_url=u))

            # infer package terms
            for line in text.splitlines()[:1200]:
                ln = line.strip()
                if not ln:
                    continue
                if _garbled_ratio(ln) > 0.30:
                    continue
                low = ln.lower()
                if any(k in low for k in ("package", "paket", "bundle", "tarif", "plan")) and len(ln) <= 140:
                    if any(n in low for n in ("tarif", "membership", "mitglied", "nothilfe", "routenplanung")):
                        continue
                    packages.append(ln)

        # Secondary pass for weak profiles with stricter spec query.
        if (_usable_feature_count(mapped_features) < _MIN_USABLE_FEATURES or len(sources) < 1) and p in {"openai", "perplexity"}:
            try:
                query2 = f"{name} technical data sheet specifications pdf official site"
                if p == "openai" and openai_key:
                    extra2 = _openai_web_search_urls(query2, api_key=openai_key, model=openai_model, max_results=8)
                elif p == "perplexity" and perplexity_key:
                    extra2 = _perplexity_web_search_urls(
                        query2,
                        api_key=perplexity_key,
                        model=perplexity_model,
                        max_results=8,
                    )
                else:
                    extra2 = []
                retry_urls = _dedupe_urls([u for u in extra2 if u not in urls])
                retry_urls = sorted(retry_urls, key=lambda u: _url_relevance_score(u, name), reverse=True)[: max_pages_per_competitor]
                for u in retry_urls:
                    try:
                        title, text, html, content_type = _fetch_page(u)
                    except Exception as exc:
                        warnings.append(f"Crawling retry failed for '{name}' url={u}: {exc}")
                        continue
                    if _is_unusable_content(title, text):
                        continue
                    if _garbled_ratio(text) > 0.35:
                        continue
                    page_texts.append(text)
                    excerpt = text[:600]
                    now_iso = datetime.now(timezone.utc).isoformat()
                    sources.append(SourceEvidence(url=u, title=title, retrieved_at=now_iso, excerpt=excerpt))
                    feature_pairs = _extract_feature_pairs_from_text(text)
                    if "html" in content_type and html:
                        feature_pairs.extend(_extract_feature_pairs_from_html_tables(html))
                    for raw_name, raw_val in feature_pairs:
                        if not _is_valid_feature_pair(raw_name, raw_val):
                            continue
                        schema_feature = _map_to_schema(raw_name, target_schema)
                        raw_value, unit, normalized_value, normalized_unit = _extract_numeric_unit(raw_val)
                        if schema_feature == "Other" and normalized_value is None:
                            continue
                        mapped_features.append(
                            MappedFeature(
                                schema_feature=schema_feature,
                                raw_name=raw_name,
                                value=raw_value,
                                unit=unit,
                                normalized_value=normalized_value,
                                normalized_unit=normalized_unit,
                                source_url=u,
                                evidence=f"{raw_name}: {raw_val}"[:260],
                            )
                        )
                    prices.extend(_extract_prices(text, source_url=u))
            except Exception as exc:
                if p == "perplexity":
                    warnings.append(f"Perplexity retry search failed for '{name}': {exc}")
                else:
                    warnings.append(f"OpenAI retry search failed for '{name}': {exc}")

        # dedupe mapped features
        mf_map: Dict[str, MappedFeature] = {}
        for f in mapped_features:
            key = f"{f.schema_feature.lower()}|{f.raw_name.lower()}|{str(f.value).lower()}|{f.source_url.lower()}"
            if key not in mf_map:
                mf_map[key] = f
        mapped_features = list(mf_map.values())[:250]
        mapped_features = _reduce_noise_features(mapped_features, max_other=12)

        # dedupe prices
        p_map: Dict[str, PriceInfo] = {}
        for pr in prices:
            key = f"{pr.raw.lower()}|{pr.value}|{pr.currency}|{pr.source_url.lower()}"
            p_map[key] = pr
        prices = list(p_map.values())[:80]

        # dedupe packages
        pk_set = []
        seen_pk = set()
        for pk in packages:
            k = pk.lower()
            if k in seen_pk:
                continue
            seen_pk.add(k)
            pk_set.append(pk)
        packages = pk_set[:50]

        freshness_days = _detect_freshness_days(page_texts)
        dq = _build_data_quality(
            mapped_features=mapped_features,
            prices=prices,
            sources=sources,
            target_schema=target_schema,
            freshness_days=freshness_days,
        )
        usable_count = _usable_feature_count(mapped_features)
        status = "usable"
        if usable_count < _MIN_USABLE_FEATURES or dq.confidence < _MIN_PROFILE_CONFIDENCE or not sources:
            warnings.append(
                f"Low-quality profile retained '{name}': usable_features={usable_count}, confidence={dq.confidence}, sources={len(sources)}"
            )
            status = "weak"
            if usable_count < _MIN_USABLE_FEATURES:
                dq.notes.append("Profil unter Mindestqualität (wenige nutzbare Merkmale).")
            if dq.confidence < _MIN_PROFILE_CONFIDENCE:
                dq.notes.append("Profil unter Mindestqualität (niedrige Konfidenz).")
            if not sources:
                dq.notes.append("Profil unter Mindestqualität (keine verwertbaren Quellen).")
        if not sources or usable_count == 0:
            status = "empty"

        competitor_profiles.append(
            CompetitorProfile(
                name=name,
                url=base_url,
                cluster=cluster,
                status=status,
                mapped_features=mapped_features,
                prices=prices,
                packages=packages,
                sources=sources,
                data_quality=dq,
            )
        )

    min_target = int(cl.get("min_competitors_target") or 0)
    usable_profiles = sum(1 for cp in competitor_profiles if cp.status == "usable")
    if min_target > 0 and usable_profiles < min_target:
        warnings.append(
            f"Usable competitor profiles below target: {usable_profiles}/{min_target}."
        )

    warnings = list(dict.fromkeys([str(w).strip() for w in warnings if str(w).strip()]))

    return CompetitorProfiles(
        provider=p,
        target_feature_schema=target_schema,
        competitor_profiles=competitor_profiles,
        extraction_warnings=warnings,
        batch_offset=start,
        batch_limit=(stop - start),
        batch_total_candidates=total_candidates,
        processed_count=processed,
    )


def merge_competitor_profiles_parts(
    *,
    part_paths: List[str],
    provider: str = "openai",
    user_root: Path,
    work_root: Path,
) -> CompetitorProfiles:
    merged_profiles: List[CompetitorProfile] = []
    warnings: List[str] = []
    target_schema: List[str] = []
    total_candidates = 0
    processed_count = 0

    for pth in part_paths:
        payload = _load_json_path_dict(pth, user_root=user_root, work_root=work_root)
        root = payload.get("competitor_profiles") if isinstance(payload.get("competitor_profiles"), dict) else payload
        if not isinstance(root, dict):
            continue
        if not target_schema:
            target_schema = _safe_list_str(root.get("target_feature_schema"))
        warnings.extend(_safe_list_str(root.get("extraction_warnings")))
        total_candidates += int(root.get("batch_total_candidates") or 0)
        processed_count += int(root.get("processed_count") or 0)
        rows = root.get("competitor_profiles") if isinstance(root.get("competitor_profiles"), list) else []
        for item in rows:
            if not isinstance(item, dict):
                continue
            try:
                merged_profiles.append(CompetitorProfile(**item))
            except Exception:
                continue

    # Dedupe by normalized name+domain with preference for higher confidence.
    by_key: Dict[str, CompetitorProfile] = {}
    for cp in merged_profiles:
        domain = _url_domain(cp.url)
        key = f"{re.sub(r'[^a-z0-9]+', ' ', cp.name.lower()).strip()}|{domain}"
        cur = by_key.get(key)
        if cur is None:
            by_key[key] = cp
            continue
        if cp.data_quality.confidence > cur.data_quality.confidence:
            by_key[key] = cp

    p = str(provider or "openai").strip().lower()
    if p not in {"openai", "ionos", "perplexity"}:
        p = "openai"

    if not target_schema:
        target_schema = list(_DEFAULT_SCHEMA)
    warnings = list(dict.fromkeys([w.strip() for w in warnings if str(w).strip()]))
    merged = list(by_key.values())

    return CompetitorProfiles(
        provider=p,
        target_feature_schema=target_schema,
        competitor_profiles=merged,
        extraction_warnings=warnings,
        batch_offset=0,
        batch_limit=None,
        batch_total_candidates=total_candidates,
        processed_count=processed_count,
    )


def _probe_source_url(url: str, timeout_seconds: int) -> Tuple[bool, str, str, str]:
    try:
        r = _http_get_with_retries(url, timeout_s=timeout_seconds, retries=1)
        status = str(r.status_code)
        content_type = str(r.headers.get("Content-Type") or "").lower()
        title = ""
        if "pdf" in content_type or str(url).lower().endswith(".pdf"):
            head = r.content[:8] if isinstance(r.content, (bytes, bytearray)) else b""
            if not head.startswith(b"%PDF-"):
                return False, status, "invalid_pdf_header", title
            txt = _parse_pdf_bytes(r.content, max_chars=4000)
            if not txt:
                return False, status, "empty_pdf_text", title
            if _is_unusable_content("", txt):
                return False, status, "unusable_pdf_content", title
            return True, status, "", title

        html = r.text or ""
        soup = BeautifulSoup(html, "html.parser")
        title = (soup.title.string or "").strip() if soup.title and soup.title.string else ""
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        txt = soup.get_text("\n", strip=True)
        if _is_unusable_content(title, txt):
            return False, status, "unusable_html_content", title
        return True, status, "", title
    except Exception as exc:
        return False, "", str(exc), ""


def verify_competitor_source_registry(
    *,
    competitor_list: Optional[Dict[str, Any]],
    competitor_list_path: Optional[str],
    source_registry: Optional[Dict[str, Any]],
    source_registry_path: Optional[str],
    max_urls_per_competitor: int = 6,
    timeout_seconds: int = 25,
    include_fallbacks: bool = True,
    user_root: Path,
    work_root: Path,
) -> CompetitorSourceRegistry:
    warnings: List[str] = []
    cl: Dict[str, Any] = {}
    if competitor_list or (competitor_list_path or "").strip():
        cl = _load_competitor_list(
            competitor_list=competitor_list,
            competitor_list_path=competitor_list_path,
            user_root=user_root,
            work_root=work_root,
        )
    registry_payload = _load_source_registry(
        source_registry=source_registry,
        source_registry_path=source_registry_path,
        user_root=user_root,
        work_root=work_root,
    )

    rows = cl.get("competitors") if isinstance(cl.get("competitors"), list) else []
    from_list: List[Tuple[str, str]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        nm = str(r.get("name") or "").strip()
        u = str(r.get("url") or "").strip()
        if nm:
            from_list.append((nm, u))

    existing_rows = registry_payload.get("competitors") if isinstance(registry_payload.get("competitors"), list) else []
    by_name: Dict[str, Dict[str, Any]] = {}
    for row in existing_rows:
        if not isinstance(row, dict):
            continue
        nm = str(row.get("name") or "").strip()
        if nm:
            by_name[_normalize_name_key(nm)] = row

    competitors_out: List[SourceRegistryCompetitor] = []
    names: List[str] = []
    names.extend([nm for nm, _u in from_list])
    names.extend([str(r.get("name") or "").strip() for r in existing_rows if isinstance(r, dict)])
    ordered_names = []
    seen = set()
    for nm in names:
        k = _normalize_name_key(nm)
        if not k or k in seen:
            continue
        seen.add(k)
        ordered_names.append(nm)

    for nm in ordered_names:
        key = _normalize_name_key(nm)
        base_url = ""
        for n2, u2 in from_list:
            if _normalize_name_key(n2) == key:
                base_url = u2
                break

        row = by_name.get(key) or {"name": nm, "entries": []}
        raw_entries = row.get("entries") if isinstance(row.get("entries"), list) else []
        candidate_urls: List[str] = []
        for ent in raw_entries:
            if isinstance(ent, str):
                if ent.startswith("http"):
                    candidate_urls.append(ent)
                continue
            if isinstance(ent, dict):
                u = str(ent.get("url") or "").strip()
                if u.startswith("http"):
                    candidate_urls.append(u)
        if include_fallbacks:
            candidate_urls.extend(_fallback_urls_for_competitor(nm, base_url))
        candidate_urls = _dedupe_urls(candidate_urls)[:max_urls_per_competitor]

        out_entries: List[SourceRegistryEntry] = []
        for idx, u in enumerate(candidate_urls):
            ok, st, err, title_hint = _probe_source_url(u, timeout_seconds=timeout_seconds)
            kind = "primary" if idx == 0 else "fallback"
            out_entries.append(
                SourceRegistryEntry(
                    url=u,
                    kind=kind,
                    priority=min(99, 10 + idx * 10),
                    active=ok,
                    last_checked_at=datetime.now(timezone.utc).isoformat(),
                    last_status=st,
                    last_error=err[:240],
                    title_hint=title_hint[:180],
                )
            )
            if not ok:
                warnings.append(f"Source inactive for '{nm}': {u} ({err or st})")

        competitors_out.append(SourceRegistryCompetitor(name=nm, entries=out_entries))

    warnings = list(dict.fromkeys([w.strip() for w in warnings if w.strip()]))
    return CompetitorSourceRegistry(schema_version="1.0", competitors=competitors_out, warnings=warnings)
