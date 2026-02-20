from __future__ import annotations

import mimetypes
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional
from xml.etree import ElementTree as ET

from bs4 import BeautifulSoup
from fastapi import HTTPException

from server.tools.pdf import read_pdf_text

from .models import ParsedDocument, ParsedMeasurement, ParsedMetadata, ParsedSection, ParsedTable

_DATE_PATTERNS = [
    r"\b\d{4}-\d{2}-\d{2}\b",
    r"\b\d{2}\.\d{2}\.\d{4}\b",
    r"\b\d{2}/\d{2}/\d{4}\b",
]

_VERSION_RE = re.compile(r"\b(?:v(?:ersion)?\s*)?(\d+(?:\.\d+){0,3})\b", re.IGNORECASE)
_MEASURE_RE = re.compile(
    r"(?P<value>-?\d+(?:[\.,]\d+)?)\s*(?P<unit>mm|cm|m|km|g|kg|mg|ml|l|w|kw|v|kv|a|ma|hz|khz|mhz|ghz|°c|%|bar|mpa|nm|n|wh|kwh)\b",
    re.IGNORECASE,
)

_SECTION_NAME_HINTS = {
    "technical data": "Technische Daten",
    "technische daten": "Technische Daten",
    "specifications": "Technische Daten",
    "features": "Vorteile",
    "benefits": "Vorteile",
    "vorteile": "Vorteile",
    "certificates": "Zertifikate",
    "certifications": "Zertifikate",
    "zertifikate": "Zertifikate",
}


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

    raise HTTPException(status_code=404, detail=f"Document not found: {path}")


def _detect_format(file_path: Path) -> tuple[str, str]:
    suffix = file_path.suffix.lower().lstrip(".")
    mime_type = mimetypes.guess_type(str(file_path))[0] or ""
    detected = suffix or "unknown"
    return detected, mime_type


def _read_text_file(file_path: Path, max_chars: int) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            return file_path.read_text(encoding=enc)[:max_chars]
        except Exception:
            continue
    raise HTTPException(status_code=500, detail="Could not decode text file")


def _extract_html(file_path: Path, max_chars: int) -> tuple[str, List[ParsedTable]]:
    raw_html = _read_text_file(file_path, max_chars=max_chars * 2)
    soup = BeautifulSoup(raw_html, "html.parser")

    tables: List[ParsedTable] = []
    for t_idx, table in enumerate(soup.find_all("table"), start=1):
        rows = []
        headers = []
        header_row = table.find("tr")
        if header_row:
            headers = [h.get_text(" ", strip=True) for h in header_row.find_all(["th", "td"])]
        for tr in table.find_all("tr"):
            row = [td.get_text(" ", strip=True) for td in tr.find_all(["th", "td"])]
            if row:
                rows.append(row)
        tables.append(
            ParsedTable(
                title=f"HTML Table {t_idx}",
                headers=headers,
                rows=rows,
                source="html",
            )
        )

    text = soup.get_text("\n", strip=True)
    return text[:max_chars], tables


def _extract_docx(file_path: Path, max_chars: int) -> tuple[str, List[ParsedTable]]:
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    tables: List[ParsedTable] = []
    lines: List[str] = []

    try:
        with zipfile.ZipFile(file_path) as zf:
            xml_bytes = zf.read("word/document.xml")
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Could not read DOCX document.xml") from exc

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise HTTPException(status_code=500, detail="Invalid DOCX XML structure") from exc

    for paragraph in root.findall(".//w:p", ns):
        runs = [t.text or "" for t in paragraph.findall(".//w:t", ns)]
        line = "".join(runs).strip()
        if line:
            lines.append(line)

    for t_idx, tbl in enumerate(root.findall(".//w:tbl", ns), start=1):
        rows: List[List[str]] = []
        for tr in tbl.findall(".//w:tr", ns):
            row: List[str] = []
            for tc in tr.findall(".//w:tc", ns):
                texts = [t.text or "" for t in tc.findall(".//w:t", ns)]
                row.append("".join(texts).strip())
            if any(cell for cell in row):
                rows.append(row)
        headers = rows[0] if rows else []
        tables.append(
            ParsedTable(
                title=f"DOCX Table {t_idx}",
                headers=headers,
                rows=rows,
                source="docx",
            )
        )

    return "\n".join(lines)[:max_chars], tables


def _extract_sections(lines: Iterable[str]) -> List[ParsedSection]:
    cleaned = [ln.strip() for ln in lines if str(ln).strip()]
    if not cleaned:
        return []

    sections: List[ParsedSection] = []
    current_name = "Allgemein"
    current_start = 1
    current_lines: List[str] = []

    def flush(end_line: int) -> None:
        if not current_lines:
            return
        sections.append(
            ParsedSection(
                name=current_name,
                text="\n".join(current_lines).strip(),
                start_line=current_start,
                end_line=end_line,
            )
        )

    for idx, line in enumerate(cleaned, start=1):
        lower = line.lower()
        mapped = next((target for src, target in _SECTION_NAME_HINTS.items() if src in lower), "")
        is_heading = bool(mapped) or line.endswith(":") or (len(line) <= 70 and line == line.upper())
        if is_heading and current_lines:
            flush(idx - 1)
            current_lines = []
            current_name = mapped or line.rstrip(":").strip().title()
            current_start = idx
            continue
        if is_heading and not current_lines:
            current_name = mapped or line.rstrip(":").strip().title()
            current_start = idx
            continue
        current_lines.append(line)

    flush(len(cleaned))
    return sections


def _extract_measurements(text: str) -> List[ParsedMeasurement]:
    out: List[ParsedMeasurement] = []
    for match in _MEASURE_RE.finditer(text):
        raw = match.group(0)
        raw_value = match.group("value").replace(",", ".")
        try:
            value: float | int
            parsed_float = float(raw_value)
            value = int(parsed_float) if parsed_float.is_integer() else parsed_float
        except Exception:
            continue

        unit = match.group("unit")
        start = max(0, match.start() - 60)
        end = min(len(text), match.end() + 60)
        context = text[start:end].replace("\n", " ").strip()

        prop_start = max(0, text.rfind("\n", 0, match.start()))
        prop_text = text[prop_start:match.start()].strip().split(":")
        property_name = prop_text[-1].strip()[:80] if prop_text else ""

        out.append(
            ParsedMeasurement(
                property_name=property_name,
                value=value,
                unit=unit,
                raw=raw,
                context=context,
            )
        )
    return out


def _extract_metadata(text: str, file_path: Path) -> ParsedMetadata:
    metadata = ParsedMetadata(product_name=file_path.stem)

    manufacturer_match = re.search(r"(?:manufacturer|hersteller)\s*[:\-]\s*([^\n\r]+)", text, re.IGNORECASE)
    if manufacturer_match:
        metadata.manufacturer = manufacturer_match.group(1).strip()[:120]

    version_match = _VERSION_RE.search(text)
    if version_match:
        metadata.document_version = version_match.group(1)

    for pattern in _DATE_PATTERNS:
        date_match = re.search(pattern, text)
        if not date_match:
            continue
        val = date_match.group(0)
        for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y"):
            try:
                metadata.publication_date = datetime.strptime(val, fmt).date().isoformat()
                return metadata
            except ValueError:
                continue

    return metadata


def parse_product_document(*, path: str, max_chars: int, user_root: Path, work_root: Path) -> ParsedDocument:
    file_path = _resolve_input_path(path, user_root=user_root.resolve(), work_root=work_root.resolve())
    detected_format, mime_type = _detect_format(file_path)
    warnings: List[str] = []
    tables: List[ParsedTable] = []
    text = ""
    ocr_used = False

    if detected_format in {"txt", "text", "md", "csv", "json"}:
        text = _read_text_file(file_path, max_chars=max_chars)
    elif detected_format in {"html", "htm"}:
        text, tables = _extract_html(file_path, max_chars=max_chars)
    elif detected_format == "pdf":
        try:
            text, _pages = read_pdf_text(file_path, max_chars=max_chars)
        except HTTPException as exc:
            detail = str(exc.detail or "")
            if "PDF reader not available" in detail:
                warnings.append("PDF reader not available. Install 'pypdf' (or PyPDF2).")
                text = ""
            else:
                raise
        if not text.strip():
            warnings.append("PDF appears to be scanned/image-based; OCR not implemented yet.")
    elif detected_format == "docx":
        text, tables = _extract_docx(file_path, max_chars=max_chars)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {detected_format}")

    lines = text.splitlines()
    sections = _extract_sections(lines)
    measurements = _extract_measurements(text)
    metadata = _extract_metadata(text, file_path)

    if not sections and text.strip():
        sections = [ParsedSection(name="Allgemein", text=text[:max_chars], start_line=1, end_line=len(lines) or 1)]

    return ParsedDocument(
        source_path=path,
        detected_format=detected_format,
        mime_type=mime_type,
        ocr_used=ocr_used,
        metadata=metadata,
        sections=sections,
        tables=tables,
        measurements=measurements,
        raw_text=text[:max_chars],
        extraction_warnings=warnings,
    )
