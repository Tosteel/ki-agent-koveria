from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

from fastapi import HTTPException

from server.services.llm_brave import LlmBrave
from server.services.llm_ionos import IonosLLM
from server.services.llm_openai import LlmOpenai
from server.services.llm_perplexity import LlmPerplexity
from server.tools.pdf import read_pdf_text

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover
    BeautifulSoup = None


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "auf",
    "bei",
    "das",
    "der",
    "die",
    "ein",
    "eine",
    "for",
    "fuer",
    "from",
    "in",
    "is",
    "mit",
    "of",
    "or",
    "the",
    "to",
    "und",
    "von",
    "zu",
}


def clean_text(value: Any) -> str:
    txt = str(value or "")
    txt = txt.replace("\x00", " ")
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def safe_list_str(values: Any, *, limit: int = 200) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
    seen: set[str] = set()
    for v in values:
        s = clean_text(v)
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= limit:
            break
    return out


def resolve_input_path(path: str, *, user_root: Path, work_root: Path) -> Path:
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


def parse_json_strictish(text: str) -> Dict[str, Any]:
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


def _extract_openai_output_text(resp: Dict[str, Any]) -> str:
    out = ""
    for item in resp.get("output", []):
        for c in item.get("content", []):
            if c.get("type") == "output_text":
                out += str(c.get("text") or "")
    return out


def llm_json(
    *,
    provider: str,
    schema_name: str,
    schema: Dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    warnings = warnings if isinstance(warnings, list) else []
    selected = str(provider or "ionos").strip().lower()
    engines = [selected] + [x for x in ("ionos", "openai", "perplexity") if x != selected]

    for engine in engines:
        try:
            if engine == "openai":
                client = LlmOpenai()
                if not client.enabled():
                    continue
                resp = client._call(
                    input_messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    text_format={
                        "type": "json_schema",
                        "name": schema_name,
                        "schema": schema,
                        "strict": False,
                    },
                )
                parsed = parse_json_strictish(_extract_openai_output_text(resp))
                if parsed:
                    return parsed
                continue

            if engine == "perplexity":
                client = LlmPerplexity()
                if not client.enabled():
                    continue
                resp = client._call(
                    input_messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    text_format={
                        "type": "json_schema",
                        "name": schema_name,
                        "schema": schema,
                        "strict": False,
                    },
                )
                parsed = parse_json_strictish(_extract_openai_output_text(resp))
                if parsed:
                    return parsed
                continue

            client = IonosLLM()
            if not client.enabled():
                continue
            comp = client.chat_completions(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "schema": schema,
                        "strict": False,
                    },
                },
            )
            parsed = parse_json_strictish(client.extract_text(comp))
            if parsed:
                return parsed
        except Exception as exc:
            warnings.append(f"llm_json failed ({engine}): {exc}")
            continue

    return {}


def brave_answers_text(
    *,
    query: str,
    enable_research: bool,
    stream: bool,
    language: Optional[str],
    country: Optional[str],
    timeout_s: int = 90,
    warnings: Optional[List[str]] = None,
) -> str:
    warnings = warnings if isinstance(warnings, list) else []
    llm = LlmBrave()
    if not llm.enabled():
        warnings.append("brave not configured")
        return ""
    q = clean_text(query)
    if not q:
        return ""

    try:
        resp = llm.chat_completions(
            messages=[{"role": "user", "content": q}],
            stream=stream,
            enable_research=enable_research,
            timeout_s=timeout_s,
        )
        # Keep Brave output as-is for downstream raw persistence.
        return llm.extract_text(resp)
    except Exception as exc:
        warnings.append(f"brave query failed: {exc}")
        return ""


def tokenize(text: str) -> List[str]:
    raw = re.findall(r"[a-zA-Z0-9\-]{2,}", clean_text(text).lower())
    return [t for t in raw if t not in _STOPWORDS]


def overlap_score(a: str, b: str) -> float:
    ta = set(tokenize(a))
    tb = set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


def clamp_score(value: float) -> float:
    try:
        v = float(value)
    except Exception:
        v = 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return round(v, 4)


def domain_of(url: str) -> str:
    u = clean_text(url).lower()
    u = re.sub(r"^https?://", "", u)
    return u.split("/", 1)[0].replace("www.", "")


def _name_from_url(url: str) -> str:
    d = domain_of(url)
    if not d:
        return ""
    left = d.split(".")[0]
    return left.replace("-", " ").replace("_", " ").strip().title()


def extract_urls(text: str) -> List[str]:
    urls = re.findall(r"https?://[^\s\]\)>,\"]+", str(text or ""))
    out: List[str] = []
    seen: set[str] = set()
    for u in urls:
        clean = u.strip().rstrip(".,;:")
        if clean in seen:
            continue
        seen.add(clean)
        out.append(clean)
    return out


def parse_startup_hits(text: str, *, source_query: str, max_hits: int = 12) -> List[Dict[str, str]]:
    raw = str(text or "").strip()
    if not raw:
        return []

    out: List[Dict[str, str]] = []
    seen: set[str] = set()

    def _push(snippet: str, url: str, source: str) -> None:
        u = clean_text(url)
        d = domain_of(u)
        sn = clean_text(snippet)
        k = (d, sn.lower()[:120])
        if (not u and not sn) or k in seen:
            return
        seen.add(k)
        out.append(
            {
                "snippet": sn,
                "url": u,
                "source": clean_text(source) or "brave_answers",
            }
        )

    # 1) Try JSON payload first.
    parsed = parse_json_strictish(raw)
    if parsed:
        candidates = parsed.get("search_results")
        if not isinstance(candidates, list):
            candidates = parsed.get("results") if isinstance(parsed.get("results"), list) else []
        for item in candidates:
            if not isinstance(item, dict):
                continue
            _push(
                str(item.get("snippet") or item.get("description") or ""),
                str(item.get("url") or item.get("website") or ""),
                str(item.get("source") or "brave_answers"),
            )
            if len(out) >= max_hits:
                return out[:max_hits]

    # 2) URL-based extraction fallback.
    urls = extract_urls(raw)
    if urls:
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        for u in urls:
            line = next((ln for ln in lines if u in ln), "")
            snippet = line.replace(u, " ").strip(" -:|\t") if line else source_query
            _push(snippet or source_query, u, "brave_answers")
            if len(out) >= max_hits:
                return out[:max_hits]

    # 3) If no URL exists, still keep one text-only candidate.
    if not out:
        _push(raw, "", "brave_answers")

    return out[:max_hits]


def dedupe_startup_hits(rows: Sequence[Dict[str, Any]], *, max_items: int = 200) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen: set[Tuple[str, str]] = set()

    for row in rows:
        if not isinstance(row, dict):
            continue
        snippet = clean_text(row.get("snippet") or row.get("description") or "")
        url = clean_text(row.get("url") or row.get("website") or "")
        source = clean_text(row.get("source") or "") or "aggregated"
        key = (domain_of(url), snippet.lower()[:120])
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "snippet": snippet,
                "url": url,
                "source": source,
            }
        )
        if len(out) >= max_items:
            break

    return out


def _read_text_file(path: Path, *, max_chars: int) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            return path.read_text(encoding=enc)[:max_chars]
        except Exception:
            continue
    raise HTTPException(status_code=500, detail=f"Could not decode text file: {path.name}")


def _read_docx_text(path: Path, *, max_chars: int) -> str:
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    try:
        with zipfile.ZipFile(path) as zf:
            xml_bytes = zf.read("word/document.xml")
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Could not read DOCX file") from exc

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise HTTPException(status_code=500, detail="Invalid DOCX XML") from exc

    lines: List[str] = []
    for paragraph in root.findall(".//w:p", ns):
        runs = [t.text or "" for t in paragraph.findall(".//w:t", ns)]
        line = clean_text("".join(runs))
        if line:
            lines.append(line)
    return "\n".join(lines)[:max_chars]


def _read_pptx_text(path: Path, *, max_chars: int) -> str:
    ns = {
        "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    }
    lines: List[str] = []
    try:
        with zipfile.ZipFile(path) as zf:
            slide_names = [
                n
                for n in zf.namelist()
                if n.startswith("ppt/slides/slide") and n.endswith(".xml")
            ]
            for slide_name in sorted(slide_names):
                xml_bytes = zf.read(slide_name)
                try:
                    root = ET.fromstring(xml_bytes)
                except ET.ParseError:
                    continue
                for node in root.findall(".//a:t", ns):
                    txt = clean_text(node.text or "")
                    if txt:
                        lines.append(txt)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Could not read PPTX file") from exc

    return "\n".join(lines)[:max_chars]


def _read_html_text(path: Path, *, max_chars: int) -> str:
    raw = _read_text_file(path, max_chars=max_chars * 2)
    if BeautifulSoup is None:
        return raw[:max_chars]
    soup = BeautifulSoup(raw, "html.parser")
    return clean_text(soup.get_text("\n", strip=True))[:max_chars]


def read_document_text(path: Path, *, max_chars: int = 50000) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        txt, _pages = read_pdf_text(path, max_chars=max_chars)
        return txt
    if ext == ".docx":
        return _read_docx_text(path, max_chars=max_chars)
    if ext == ".pptx":
        return _read_pptx_text(path, max_chars=max_chars)
    if ext in {".html", ".htm"}:
        return _read_html_text(path, max_chars=max_chars)
    return _read_text_file(path, max_chars=max_chars)


def extract_bullets_from_text(text: str, *, limit: int = 200) -> List[str]:
    if not text:
        return []

    candidates: List[str] = []
    for line in str(text).splitlines():
        s = clean_text(line)
        if not s:
            continue
        s = re.sub(r"^[\-\*\u2022\d\.)\s]+", "", s).strip()
        if not s:
            continue
        if len(s) < 3:
            continue
        candidates.append(s)

    if len(candidates) < 8:
        for part in re.split(r"[;\n]", text):
            s = clean_text(part)
            if len(s) >= 3:
                candidates.append(s)

    out: List[str] = []
    seen: set[str] = set()
    for c in candidates:
        k = c.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(c)
        if len(out) >= limit:
            break
    return out


def classify_bullets_heuristic(bullets: Sequence[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {
        "innovation_goals": [],
        "strategic_fields": [],
        "problem_statements": [],
        "technology_interests": [],
        "target_use_cases": [],
    }

    for b in bullets:
        t = clean_text(b).lower()
        if not t:
            continue

        if any(k in t for k in ("ziel", "goal", "objective", "innovation", "wachstum", "expand")):
            out["innovation_goals"].append(b)
        if any(k in t for k in ("strategie", "strateg", "markt", "segment", "business", "partner")):
            out["strategic_fields"].append(b)
        if any(k in t for k in ("problem", "challenge", "pain", "hindernis", "risiko", "bottleneck")):
            out["problem_statements"].append(b)
        if any(k in t for k in ("ki", "ai", "daten", "data", "platform", "tech", "api", "software", "sensor")):
            out["technology_interests"].append(b)
        if any(k in t for k in ("use case", "anwendung", "einsatz", "workflow", "prozess", "pilot")):
            out["target_use_cases"].append(b)

    # Fallback: keep broad coverage instead of empty sections.
    ordered = [clean_text(x) for x in bullets if clean_text(x)]
    if not out["innovation_goals"]:
        out["innovation_goals"] = ordered[:5]
    if not out["strategic_fields"]:
        out["strategic_fields"] = ordered[:5]
    if not out["problem_statements"]:
        out["problem_statements"] = ordered[:5]
    if not out["technology_interests"]:
        out["technology_interests"] = ordered[:5]
    if not out["target_use_cases"]:
        out["target_use_cases"] = ordered[:5]

    for key in list(out.keys()):
        out[key] = safe_list_str(out[key], limit=20)

    return out


def load_json_obj(
    *,
    inline_obj: Optional[Dict[str, Any]],
    path: Optional[str],
    root_key: Optional[str],
    user_root: Path,
    work_root: Path,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}

    if isinstance(inline_obj, dict) and inline_obj:
        payload = inline_obj
    else:
        resolved = resolve_input_path(str(path or ""), user_root=user_root, work_root=work_root)
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON in path: {path}") from exc

    if root_key and isinstance(payload.get(root_key), dict):
        payload = payload[root_key]

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    return payload


def compact_context(values: Iterable[Any], *, max_chars: int = 18000) -> str:
    lines: List[str] = []
    for value in values:
        if isinstance(value, str):
            s = clean_text(value)
            if s:
                lines.append(s)
            continue
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, list):
                    vv = ", ".join(safe_list_str(v, limit=20))
                    lines.append(f"{k}: {vv}")
                else:
                    lines.append(f"{k}: {clean_text(v)}")
    joined = "\n".join(lines)
    return joined[:max_chars]


def pick_year_from_text(text: str) -> str:
    m = re.search(r"\b(19\d{2}|20\d{2})\b", str(text or ""))
    return m.group(1) if m else ""


def location_from_text(text: str) -> str:
    s = clean_text(text)
    patterns = [
        r"\bheadquartered in ([A-Za-z][A-Za-z\s,\-]{2,60})",
        r"\bbased in ([A-Za-z][A-Za-z\s,\-]{2,60})",
        r"\bstandort[:\-]?\s*([A-Za-z][A-Za-z\s,\-]{2,60})",
    ]
    for pattern in patterns:
        m = re.search(pattern, s, flags=re.IGNORECASE)
        if m:
            return clean_text(m.group(1))
    return ""


def normalize_website(url: str) -> str:
    u = clean_text(url)
    if not u:
        return ""
    if not re.match(r"^https?://", u, flags=re.IGNORECASE):
        u = f"https://{u}"
    return u


def infer_company_name_from_domain(url: str) -> str:
    d = domain_of(url)
    if not d:
        return ""
    parts = d.split(".")
    if not parts:
        return ""
    return clean_text(parts[0]).replace("-", " ").title()


def as_json_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def extract_title_host(url: str) -> str:
    try:
        host = urlparse(url).hostname or ""
    except Exception:
        host = ""
    return host.replace("www.", "")
