from __future__ import annotations

import json
import re
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Dict, List

from fastapi import HTTPException
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .models import Step71PdfExportRequest, Step71PdfExportResult


def _clean_text(value: str) -> str:
    return " ".join(str(value or "").split()).strip()


def _resolve_input_path(path: str, *, user_root: Path, work_root: Path) -> Path:
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


def _load_payload(
    *,
    inline_obj: Dict[str, Any] | None,
    path: str | None,
    user_root: Path,
    work_root: Path,
) -> Dict[str, Any]:
    if isinstance(inline_obj, dict) and inline_obj:
        payload = inline_obj
    else:
        resolved = _resolve_input_path(str(path or ""), user_root=user_root, work_root=work_root)
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON in path: {path}") from exc
    if isinstance(payload.get("final_report"), dict):
        payload = payload["final_report"]
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid final_report payload: expected object.")
    return payload


def _resolve_output_path(output_path: str, work_root: Path) -> Path:
    raw = str(output_path or "").strip().lstrip("/")
    p = Path(raw)
    if not raw or p.is_absolute() or ".." in p.parts:
        raise HTTPException(status_code=400, detail=f"Invalid output_path: {output_path}")
    resolved = (work_root.resolve() / p).resolve()
    if work_root.resolve() not in resolved.parents and resolved != work_root.resolve():
        raise HTTPException(status_code=400, detail="output_path must stay inside user work dir")
    if resolved.suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="output_path must end with .pdf")
    return resolved


def _resolve_optional_asset_path(asset_path: str | None, user_root: Path, work_root: Path) -> Path | None:
    raw = str(asset_path or "").strip().lstrip("/")
    if not raw:
        return None
    p = Path(raw)
    if p.is_absolute() or ".." in p.parts:
        return None
    roots = [work_root.resolve(), user_root.resolve()]
    for root in roots:
        candidate = (root / p).resolve()
        if (candidate == root or root in candidate.parents) and candidate.is_file():
            return candidate
    return None


def _chapter_text(report: Dict[str, Any], key: str) -> str:
    chapters = report.get("chapters") if isinstance(report.get("chapters"), dict) else {}
    txt = str(chapters.get(key) or "").strip()
    return txt if txt else "-"


def _para_text(value: str) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        return "-"
    return escape(cleaned).replace("\n", "<br/>")


def _chapter_blocks(text: str) -> tuple[list[str], list[str]]:
    raw = str(text or "").strip()
    if not raw or raw == "-":
        return [], []
    lines = [ln.strip() for ln in raw.replace("\r", "").split("\n") if ln.strip()]
    explicit_bullets = [re.sub(r"^\s*[-*•]\s*", "", ln).strip() for ln in lines if re.match(r"^\s*[-*•]\s+", ln)]
    if explicit_bullets:
        paragraphs = [ln for ln in lines if not re.match(r"^\s*[-*•]\s+", ln)]
        return paragraphs, explicit_bullets

    text_flat = " ".join(lines)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text_flat) if s.strip()]
    if len(sentences) <= 4:
        return sentences, []
    return [" ".join(sentences[:2])], sentences[2:]


def _highlight_metrics(text: str) -> str:
    t = str(text or "")
    t = re.sub(r"\b(\d+[.,]\d+)\s*(?:/5|von 5 Sternen|Sterne?)\b", r"\1 ⭐", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(\d{2,5})\s*Bewertungen\b", r"\1 Bewertungen", t, flags=re.IGNORECASE)
    return t


def _parse_labeled_sections(text: str) -> list[tuple[str, str]]:
    raw = str(text or "").replace("\r", "").strip()
    if not raw or raw == "-":
        return []
    lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
    sections: list[tuple[str, str]] = []
    for line in lines:
        if ":" in line:
            left, right = line.split(":", 1)
            title = _clean_text(left)
            content = _clean_text(right)
            if title and content:
                sections.append((title, content))
    return sections


def _parse_markdown_table(md: str) -> List[List[str]]:
    lines = [ln.strip() for ln in str(md or "").splitlines() if ln.strip().startswith("|")]
    if len(lines) < 3:
        return []
    rows: List[List[str]] = []
    for idx, line in enumerate(lines):
        cells = [c.strip() for c in line.strip("|").split("|")]
        if not cells:
            continue
        if idx == 1 and all(set(c) <= {"-", ":"} for c in cells):
            continue
        rows.append(cells)
    if len(rows) < 2:
        return []
    col_count = max(len(r) for r in rows)
    normalized: List[List[str]] = []
    for row in rows:
        r = row + [""] * (col_count - len(row))
        normalized.append(r[:col_count])
    return normalized


def _parse_float(value: str) -> float | None:
    s = str(value or "").strip().replace(",", ".")
    match = re.search(r"\d+(?:\.\d+)?", s)
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


def _compact_matrix_text(row_label: str, value: str, *, max_len: int = 110) -> str:
    text = _clean_text(str(value or ""))
    if not text:
        return "-"
    if row_label.lower().startswith("trend:"):
        parts = [p.strip() for p in re.split(r"[;|]", text) if p.strip()]
        if parts:
            text = " | ".join(parts[:2])
    if len(text) > max_len:
        return text[: max_len - 1].rstrip() + "…"
    return text


def _format_matrix_cell(row_label: str, value: str) -> str:
    label = row_label.lower()
    raw = str(value or "").strip()
    if not raw:
        return "-"
    if "google bewertung" in label:
        fv = _parse_float(raw)
        if fv is not None:
            return f"{fv:.1f} ⭐"
    if "anzahl bewertungen" in label:
        fv = _parse_float(raw)
        if fv is not None:
            return f"{int(round(fv))} Bewertungen"
    return _highlight_metrics(_compact_matrix_text(row_label, raw))


def _draw_page_chrome(title: str, date_txt: str, toolname: str):
    title_txt = _clean_text(title) or "Competitive Intelligence Report"
    meta_txt = f"Seite {{page}} | {date_txt} | {toolname}"

    def _draw(canvas, doc) -> None:  # reportlab callback signature
        canvas.saveState()
        canvas.setFont("Helvetica-Bold", 9)
        canvas.setFillColor(colors.HexColor("#3e5c76"))
        canvas.drawString(22 * mm, A4[1] - 10 * mm, title_txt)
        canvas.setStrokeColor(colors.HexColor("#d6dde5"))
        canvas.line(22 * mm, A4[1] - 11 * mm, A4[0] - 22 * mm, A4[1] - 11 * mm)

        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(colors.HexColor("#526177"))
        canvas.drawRightString(A4[0] - 22 * mm, 10 * mm, meta_txt.format(page=doc.page))
        canvas.restoreState()

    return _draw


def _build_recommendation_box(
    *,
    title: str,
    paragraphs: list[str],
    bullets: list[str],
    title_style: ParagraphStyle,
    body_style: ParagraphStyle,
    bullet_style: ParagraphStyle,
) -> Table:
    content_flowables: list[Any] = []
    for paragraph in paragraphs:
        content_flowables.append(Paragraph(_para_text(_highlight_metrics(paragraph)), body_style))
    for item in bullets:
        content_flowables.append(Paragraph(_para_text(_highlight_metrics(item)), bullet_style, bulletText="•"))
    if not content_flowables:
        content_flowables.append(Paragraph("-", body_style))

    box = Table(
        [
            [Paragraph(_para_text(title or "Empfehlung"), title_style)],
            [content_flowables],
        ],
        colWidths=[170 * mm],
    )
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1d3557")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#eef4fb")),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#b7c5d3")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return box


def _append_section_header(story: list[Any], title: str, h1_style: ParagraphStyle) -> None:
    story.append(Spacer(1, 6))
    story.append(Paragraph(_para_text(title), h1_style))
    divider = Table([[""]], colWidths=[170 * mm], rowHeights=[3])
    divider.setStyle(
        TableStyle(
            [
                ("LINEBELOW", (0, 0), (-1, -1), 1.0, colors.HexColor("#c6d0db")),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    story.append(divider)
    story.append(Spacer(1, 8))


def _render_report_pdf(
    *,
    report: Dict[str, Any],
    output_file: Path,
    title: str,
    subtitle: str,
    report_year: int | None,
    created_by: str,
    company_logo: Path | None,
    tool_logo: Path | None,
) -> int:
    chapter_1 = _chapter_text(report, "chapter_1_executive_summary")
    chapter_2 = _chapter_text(report, "chapter_2_company_profiles")
    chapter_3 = _chapter_text(report, "chapter_3_company_matrix")
    chapter_4 = _chapter_text(report, "chapter_4_insights")
    chapter_5 = _chapter_text(report, "chapter_5_recommendations")
    chapter_6 = _chapter_text(report, "chapter_6_appendix_trends")
    year_txt = str(report_year or datetime.now().year)
    date_txt = datetime.now().strftime("%d.%m.%Y")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_file),
        pagesize=A4,
        rightMargin=22 * mm,
        leftMargin=22 * mm,
        topMargin=20 * mm,
        bottomMargin=18 * mm,
        title=_clean_text(title) or "Competitive Intelligence Report",
        author=_clean_text(created_by) or "Competitive Intelligence Tool",
        subject="Competitive Intelligence Report",
    )

    styles = getSampleStyleSheet()
    cover_title = ParagraphStyle(
        "CoverTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=28,
        leading=32,
        alignment=1,
        textColor=colors.HexColor("#1d2d44"),
        spaceAfter=10,
    )
    cover_subtitle = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=13,
        leading=18,
        alignment=1,
        textColor=colors.HexColor("#3e5c76"),
        spaceAfter=12,
    )
    cover_meta = ParagraphStyle(
        "CoverMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        alignment=1,
        textColor=colors.HexColor("#2f3e46"),
    )
    h1 = ParagraphStyle(
        "H1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0b2545"),
        spaceBefore=16,
        spaceAfter=10,
    )
    h2 = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#13315c"),
        spaceBefore=14,
        spaceAfter=8,
    )
    box_title = ParagraphStyle(
        "BoxTitle",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.white,
        spaceBefore=0,
        spaceAfter=0,
    )
    body = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=15,
        spaceAfter=8,
    )
    bullet = ParagraphStyle(
        "Bullet",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10.2,
        leading=14,
        leftIndent=10,
        bulletIndent=0,
        spaceAfter=5,
    )
    table_header = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=12,
        textColor=colors.white,
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
    )

    story: List[Any] = []

    logos: List[Any] = []
    for asset in (company_logo, tool_logo):
        if asset is None:
            logos.append(Spacer(1, 1))
            continue
        logos.append(Image(str(asset), width=34 * mm, height=18 * mm, kind="proportional"))
    if any(isinstance(x, Image) for x in logos):
        logo_table = Table([logos], colWidths=[80 * mm, 80 * mm])
        logo_table.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        story.append(Spacer(1, 12 * mm))
        story.append(logo_table)

    story.append(Spacer(1, 25 * mm))
    story.append(Paragraph(_para_text(title) or "Competitive Intelligence Report", cover_title))
    story.append(
        Paragraph(
            _para_text(subtitle) or "Wettbewerbsanalyse und strategische Handlungsempfehlungen",
            cover_subtitle,
        )
    )
    story.append(Spacer(1, 16 * mm))
    story.append(Paragraph(f"<b>Datum:</b> {year_txt}", cover_meta))
    story.append(Paragraph(f"<b>Erstellt durch:</b> {escape(_clean_text(created_by) or 'Competitive Intelligence Tool')}", cover_meta))
    story.append(PageBreak())

    story.append(Paragraph("Inhaltsverzeichnis", h1))
    toc_rows = [
        ["1.", "Zusammenfassung"],
        ["2.", "Unternehmensprofile"],
        ["3.", "Wettbewerbsvergleich"],
        ["4.", "Insights"],
        ["5.", "Handlungsempfehlungen"],
        ["6.", "Anhang: Quellen zur Trenderfassung"],
    ]
    toc_table = Table(
        [[Paragraph(_para_text(x), body) for x in row] for row in toc_rows],
        colWidths=[12 * mm, 150 * mm],
    )
    toc_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(toc_table)
    story.append(PageBreak())

    _append_section_header(story, "1. Zusammenfassung", h1)
    p, b = _chapter_blocks(chapter_1)
    for paragraph in p:
        story.append(Paragraph(_para_text(_highlight_metrics(paragraph)), body))
    if b:
        story.append(Paragraph("Kernaussagen", h2))
        for item in b:
            story.append(Paragraph(_para_text(_highlight_metrics(item)), bullet, bulletText="•"))

    _append_section_header(story, "2. Unternehmensprofile", h1)
    p, b = _chapter_blocks(chapter_2)
    for paragraph in p:
        story.append(Paragraph(_para_text(_highlight_metrics(paragraph)), body))
    if b:
        story.append(Paragraph("Wesentliche Punkte", h2))
        for item in b:
            story.append(Paragraph(_para_text(_highlight_metrics(item)), bullet, bulletText="•"))

    story.append(PageBreak())
    _append_section_header(story, "3. Wettbewerbsvergleich", h1)
    matrix_rows = _parse_markdown_table(chapter_3)
    if matrix_rows:
        col_count = len(matrix_rows[0])
        avail_width = 170 * mm
        first_col = 44 * mm
        other = (avail_width - first_col) / max(1, col_count - 1)
        col_widths = [first_col] + [max(22 * mm, other)] * max(0, col_count - 1)
        rendered_rows: List[List[Paragraph]] = []
        for idx, row in enumerate(matrix_rows):
            if idx == 0:
                rendered_rows.append([Paragraph(_para_text(cell), table_header) for cell in row])
                continue
            row_label = row[0] if row else ""
            current: List[Paragraph] = [Paragraph(_para_text(row_label), table_cell)]
            for cell in row[1:]:
                current.append(Paragraph(_para_text(_format_matrix_cell(row_label, cell)), table_cell))
            rendered_rows.append(current)
        matrix_table = Table(
            rendered_rows,
            colWidths=col_widths,
            repeatRows=1,
            splitByRow=1,
        )
        matrix_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#6b7785")),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b7c5d3")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f3f6")]),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ]
            )
        )
        story.append(matrix_table)
    else:
        story.append(Paragraph(_para_text(_highlight_metrics(chapter_3)), body))

    story.append(PageBreak())
    _append_section_header(story, "4. Insights", h1)
    p, b = _chapter_blocks(chapter_4)
    for paragraph in p:
        story.append(Paragraph(_para_text(_highlight_metrics(paragraph)), body))
    if b:
        story.append(Paragraph("Ableitungen", h2))
        for item in b:
            story.append(Paragraph(_para_text(_highlight_metrics(item)), bullet, bulletText="•"))

    _append_section_header(story, "5. Handlungsempfehlungen", h1)
    labeled_sections = _parse_labeled_sections(chapter_5)
    if labeled_sections:
        for title_txt, content_txt in labeled_sections:
            p, b = _chapter_blocks(content_txt)
            story.append(
                _build_recommendation_box(
                    title=title_txt,
                    paragraphs=p,
                    bullets=b,
                    title_style=box_title,
                    body_style=body,
                    bullet_style=bullet,
                )
            )
            story.append(Spacer(1, 6))
    else:
        p, b = _chapter_blocks(chapter_5)
        story.append(
            _build_recommendation_box(
                title="Handlungsempfehlungen",
                paragraphs=p,
                bullets=b,
                title_style=box_title,
                body_style=body,
                bullet_style=bullet,
            )
        )

    story.append(PageBreak())
    _append_section_header(story, "6. Anhang: Quellen zur Trenderfassung", h1)
    p, b = _chapter_blocks(chapter_6)
    for paragraph in p:
        story.append(Paragraph(_para_text(_highlight_metrics(paragraph)), body))
    for item in b:
        story.append(Paragraph(_para_text(_highlight_metrics(item)), bullet, bulletText="•"))

    page_cb = _draw_page_chrome(
        _clean_text(title) or "Competitive Intelligence Report",
        date_txt,
        _clean_text(created_by) or "Competitive Intelligence Tool",
    )
    doc.build(story, onFirstPage=page_cb, onLaterPages=page_cb)
    return output_file.stat().st_size


def run_step_7_1_pdf_export(
    *,
    req: Step71PdfExportRequest,
    user_root: Path,
    work_root: Path,
) -> Step71PdfExportResult:
    warnings: List[str] = []
    report = _load_payload(
        inline_obj=req.final_report,
        path=req.final_report_path,
        user_root=user_root,
        work_root=work_root,
    )
    out = _resolve_output_path(req.output_path, work_root)
    company_logo = _resolve_optional_asset_path(req.company_logo_path, user_root, work_root)
    tool_logo = _resolve_optional_asset_path(req.tool_logo_path, user_root, work_root)
    bytes_written = _render_report_pdf(
        report=report,
        output_file=out,
        title=req.title,
        subtitle=req.subtitle,
        report_year=req.report_year,
        created_by=req.created_by,
        company_logo=company_logo,
        tool_logo=tool_logo,
    )
    return Step71PdfExportResult(
        provider=_clean_text(str(req.provider or "ionos")) or "ionos",
        output_file=req.output_path,
        bytes_written=bytes_written,
        title=req.title,
        extraction_warnings=warnings,
    )
