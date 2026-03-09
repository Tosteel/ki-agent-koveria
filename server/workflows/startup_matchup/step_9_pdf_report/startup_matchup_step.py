from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from fastapi import HTTPException
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from server.workflows.startup_matchup.common import clean_text, load_json_obj, safe_list_str

from .models import PdfReportResult, StartupMatchupStep9Request


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


def _join_bullets(values: List[str]) -> str:
    if not values:
        return "n/a"
    return "\n".join([f"- {v}" for v in values])


def _recommended_table_data(recommended: List[Dict[str, Any]]) -> List[List[str]]:
    rows: List[List[str]] = [["Rank", "Startup", "Score", "Kurzprofil"]]
    if not recommended:
        rows.append(["-", "Keine empfohlenen Startups verfuegbar.", "-", "-"])
        return rows

    for item in recommended:
        if not isinstance(item, dict):
            continue
        rows.append(
            [
                str(int(item.get("rank") or 0)),
                clean_text(item.get("startup_name") or item.get("name") or ""),
                f"{float(item.get('relevance_score') or 0.0):.4f}",
                clean_text(item.get("short_description") or ""),
            ]
        )
    return rows


def _narrative_or_fallback(narrative: str, fallback: str) -> str:
    n = clean_text(narrative)
    if len(n) < 80:
        return clean_text(fallback)
    return n


def _render_report_pdf(*, report: Dict[str, Any], output_file: Path, title: str) -> int:
    company_profile = report.get("company_profile") if isinstance(report.get("company_profile"), dict) else {}
    innovation_goals = safe_list_str(report.get("innovation_goals"))
    identified_gaps = safe_list_str(report.get("identified_gaps"))
    startup_search_fields = safe_list_str(report.get("startup_search_fields"))
    recommended = report.get("recommended_startups") if isinstance(report.get("recommended_startups"), list) else []
    report_text = report.get("report") if isinstance(report.get("report"), dict) else {}

    output_file.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_file),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        spaceAfter=10,
    )
    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        spaceBefore=8,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=14,
        spaceAfter=6,
    )
    bullet_style = ParagraphStyle(
        "Bullets",
        parent=styles["Code"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        spaceAfter=5,
    )
    startup_title_style = ParagraphStyle(
        "StartupTitle",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=14,
        spaceBefore=4,
        spaceAfter=4,
    )
    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=11,
        alignment=1,
    )
    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=11,
    )
    meta_label_style = ParagraphStyle(
        "MetaLabel",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=12,
    )
    meta_value_style = ParagraphStyle(
        "MetaValue",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=12,
    )

    story = []
    story.append(Paragraph(clean_text(title) or "Startup Matchup Report", title_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("I. Executive Summary", section_style))
    story.append(
        Paragraph(
            _narrative_or_fallback(
                clean_text(report_text.get("executive_summary")),
                "Dieser Bericht dokumentiert die systematische Suche nach Startup-Kooperationspartnern.",
            ),
            body_style,
        )
    )

    story.append(Paragraph("II. Unternehmensprofil", section_style))
    company_lines = [
        f"Unternehmen: {clean_text(company_profile.get('company_name')) or 'n/a'}",
        f"Branche: {clean_text(company_profile.get('industry')) or 'n/a'}",
        f"Kerngeschaeft: {clean_text(company_profile.get('core_business')) or 'n/a'}",
    ]
    story.append(Paragraph("<br/>".join(company_lines), body_style))
    story.append(
        Paragraph(
            _narrative_or_fallback(
                clean_text(report_text.get("company_profile")),
                clean_text(company_profile.get("core_business") or ""),
            ),
            body_style,
        )
    )

    story.append(Paragraph("III. Innovationsziele", section_style))
    story.append(
        Paragraph(
            _narrative_or_fallback(
                clean_text(report_text.get("innovation_goals")),
                "; ".join(innovation_goals),
            ),
            body_style,
        )
    )
    story.append(Paragraph(_join_bullets(innovation_goals).replace("\n", "<br/>"), bullet_style))

    story.append(Paragraph("IV. Gap-Analyse", section_style))
    story.append(
        Paragraph(
            _narrative_or_fallback(
                clean_text(report_text.get("gap_analysis")),
                "; ".join(identified_gaps),
            ),
            body_style,
        )
    )
    story.append(Paragraph(_join_bullets(identified_gaps).replace("\n", "<br/>"), bullet_style))

    story.append(Paragraph("V. Startup-Suchfelder", section_style))
    story.append(
        Paragraph(
            _narrative_or_fallback(
                clean_text(report_text.get("startup_search_fields")),
                "; ".join(startup_search_fields),
            ),
            body_style,
        )
    )
    story.append(Paragraph(_join_bullets(startup_search_fields).replace("\n", "<br/>"), bullet_style))

    story.append(Paragraph("VI. Empfohlene Startups", section_style))
    story.append(
        Paragraph(
            _narrative_or_fallback(
                clean_text(report_text.get("recommended_startups")),
                "Die folgende Tabelle zeigt die priorisierten Startup-Kandidaten.",
            ),
            body_style,
        )
    )

    raw_table_data = _recommended_table_data(recommended)
    table_data: List[List[Any]] = []
    for r_idx, row in enumerate(raw_table_data):
        if r_idx == 0:
            table_data.append(
                [
                    Paragraph(clean_text(row[0]), table_header_style),
                    Paragraph(clean_text(row[1]), table_header_style),
                    Paragraph(clean_text(row[2]), table_header_style),
                    Paragraph(clean_text(row[3]), table_header_style),
                ]
            )
            continue
        table_data.append(
            [
                Paragraph(clean_text(row[0]), table_cell_style),
                Paragraph(clean_text(row[1]), table_cell_style),
                Paragraph(clean_text(row[2]), table_cell_style),
                Paragraph(clean_text(row[3]), table_cell_style),
            ]
        )

    table = Table(table_data, colWidths=[14 * mm, 34 * mm, 18 * mm, 108 * mm], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8edf3")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1b2a41")),
                ("ALIGN", (0, 0), (0, -1), "CENTER"),
                ("ALIGN", (2, 1), (2, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#7a8aa0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fc")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("VII. Detailprofile der Top-Startups", section_style))
    for item in recommended:
        if not isinstance(item, dict):
            continue
        profile = item.get("profile") if isinstance(item.get("profile"), dict) else {}
        name = clean_text(item.get("startup_name") or profile.get("name") or "")
        story.append(Paragraph(name or "Startup", startup_title_style))

        detail_rows: List[List[Any]] = []

        def _add_detail(label: str, value: Any) -> None:
            if isinstance(value, list):
                value_txt = ", ".join(safe_list_str(value))
            else:
                value_txt = clean_text(value)
            if not value_txt:
                return
            detail_rows.append(
                [
                    Paragraph(label, meta_label_style),
                    Paragraph(value_txt, meta_value_style),
                ]
            )

        _add_detail("Gruendungsjahr", profile.get("founding_year"))
        _add_detail("Standort", profile.get("location"))
        _add_detail("Technologiefokus", profile.get("technology_focus"))
        _add_detail("Relevanzbegruendung", profile.get("why_relevant"))
        _add_detail("Webseite", profile.get("website") or profile.get("url"))
        _add_detail("Relevanzscore", f"{float(item.get('relevance_score') or profile.get('relevance_score') or 0.0):.4f}")

        if detail_rows:
            detail_table = Table(detail_rows, colWidths=[34 * mm, 140 * mm])
            detail_table.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LINEBELOW", (0, 0), (-1, -2), 0.25, colors.HexColor("#d4dbe5")),
                        ("LEFTPADDING", (0, 0), (-1, -1), 3),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                        ("TOPPADDING", (0, 0), (-1, -1), 2),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ]
                )
            )
            story.append(detail_table)
        story.append(Spacer(1, 6))

    doc.build(story)
    return output_file.stat().st_size


def run_step_9(*, req: StartupMatchupStep9Request, user_root: Path, work_root: Path) -> PdfReportResult:
    report = load_json_obj(
        inline_obj=req.final_report,
        path=req.final_report_path,
        root_key="final_report",
        user_root=user_root,
        work_root=work_root,
    )

    output = _resolve_output_path(req.output_path, work_root)
    bytes_written = _render_report_pdf(report=report, output_file=output, title=req.title)

    return PdfReportResult(
        output_path=str(req.output_path),
        bytes_written=bytes_written,
        title=req.title,
        text_preview="PDF generated with formatted headings, narrative sections and startup table.",
    )
