from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from fastapi import HTTPException
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

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


def _narrative_or_fallback(narrative: Any, fallback: str) -> str:
    n = clean_text(narrative)
    if len(n) < 80:
        return clean_text(fallback)
    return n


def _score_bar(score: float) -> str:
    s = max(0.0, min(1.0, float(score or 0.0)))
    total = 12
    filled = int(round(s * total))
    return ("#" * filled + "-" * (total - filled)) + f" {s:.2f}"


def _gap_significance(gap: str) -> str:
    g = clean_text(gap).lower()
    if "innovationszykl" in g:
        return "Verzögerung bei der Markteinführung neuer Technologien"
    if "transparenz" in g and "startup" in g:
        return "Erschwerte Identifikation geeigneter Innovationspartner"
    if "prioris" in g and "partner" in g:
        return "Unsichere Entscheidungsprozesse bei Kooperationsvorhaben"
    return "Reduzierte Umsetzungsgeschwindigkeit in der Innovationsarbeit"


def _draw_page_number(canvas, doc) -> None:  # reportlab callback signature
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.HexColor("#526177"))
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Seite {doc.page}")
    canvas.restoreState()


def _render_report_pdf(
    *,
    report: Dict[str, Any],
    output_file: Path,
    title: str,
    subtitle: str,
    company_name_override: str | None,
    report_year: int | None,
    created_by: str,
    company_logo: Path | None,
    tool_logo: Path | None,
) -> int:
    company_profile = report.get("company_profile") if isinstance(report.get("company_profile"), dict) else {}
    innovation_goals = safe_list_str(report.get("innovation_goals"))
    identified_gaps = safe_list_str(report.get("identified_gaps"))
    startup_search_fields = safe_list_str(report.get("startup_search_fields"))
    recommended = report.get("recommended_startups") if isinstance(report.get("recommended_startups"), list) else []
    report_text = report.get("report") if isinstance(report.get("report"), dict) else {}

    company_name = clean_text(company_name_override) or clean_text(company_profile.get("company_name")) or "n/a"
    year_txt = str(report_year or datetime.now().year)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_file),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=clean_text(title) or "Startup Matchup Report",
        author=clean_text(created_by) or "Startup Matchup Tool",
        subject="Startup-Scouting Report",
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
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#0b2545"),
        spaceBefore=8,
        spaceAfter=8,
    )
    h2 = ParagraphStyle(
        "H2",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=14,
        textColor=colors.HexColor("#13315c"),
        spaceBefore=6,
        spaceAfter=4,
    )
    body = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=14,
        spaceAfter=6,
    )
    body_small = ParagraphStyle(
        "BodySmall",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=12,
        spaceAfter=4,
    )
    table_header = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=11,
        textColor=colors.white,
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=11,
    )

    story: List[Any] = []

    # I. Titelblatt
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
    story.append(Paragraph(clean_text(title) or "Startup Matchup Report", cover_title))
    story.append(
        Paragraph(
            clean_text(subtitle) or "Startup-Scouting für strategische Innovationspartnerschaften",
            cover_subtitle,
        )
    )
    story.append(Spacer(1, 16 * mm))
    story.append(Paragraph(f"<b>Unternehmen:</b> {company_name}", cover_meta))
    story.append(Paragraph(f"<b>Datum:</b> {year_txt}", cover_meta))
    story.append(Paragraph(f"<b>Erstellt durch:</b> {clean_text(created_by) or 'Startup Matchup Tool'}", cover_meta))
    story.append(PageBreak())

    top_startups = [
        clean_text(item.get("startup_name") or item.get("name") or "")
        for item in recommended
        if isinstance(item, dict)
    ]
    top_startups = [x for x in top_startups if x][:5]

    # II. Executive Summary
    story.append(Paragraph("I. Executive Summary", h1))
    story.append(
        Paragraph(
            _narrative_or_fallback(
                report_text.get("executive_summary"),
                "Die Analyse identifiziert priorisierte Startup-Kooperationspartner für strategische Innovationsfelder.",
            ),
            body,
        )
    )

    executive_rows = [
        ["Unternehmen", f"{company_name}"],
        [
            "Ziel der Analyse",
            "Identifikation relevanter Startups zur Beschleunigung strategischer Innovationspartnerschaften.",
        ],
        [
            "Wesentliche Innovationsfelder",
            "<br/>".join([f"• {clean_text(x)}" for x in startup_search_fields[:4]]) or "n/a",
        ],
        [
            "Ergebnis",
            "Die Analyse identifizierte mehrere potenzielle Startup-Kooperationspartner. "
            + (
                "Die relevantesten Kandidaten sind: " + ", ".join(top_startups[:2]) + "."
                if top_startups
                else "Es liegen noch keine priorisierten Kandidaten vor."
            ),
        ],
    ]
    executive_table = Table(
        [[Paragraph("Kategorie", table_header), Paragraph("Inhalt", table_header)]]
        + [[Paragraph(clean_text(k), table_cell), Paragraph(clean_text(v), table_cell)] for k, v in executive_rows],
        colWidths=[42 * mm, 128 * mm],
    )
    executive_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1d3557")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b7c5d3")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7fb")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(executive_table)

    # III. Unternehmensprofil
    story.append(Paragraph("II. Unternehmensprofil", h1))
    profile_rows = [
        ["Unternehmen", company_name],
        ["Branche", clean_text(company_profile.get("industry")) or "n/a"],
        ["Kerngeschäft", clean_text(company_profile.get("core_business")) or "n/a"],
        ["Innovationsfokus", ", ".join(safe_list_str(company_profile.get("innovation_focus"))) or "n/a"],
        [
            "Strategische Initiativen",
            ", ".join(safe_list_str(company_profile.get("strategic_objectives"))) or "n/a",
        ],
    ]
    profile_table = Table(
        [[Paragraph("Kategorie", table_header), Paragraph("Inhalt", table_header)]]
        + [[Paragraph(k, table_cell), Paragraph(v, table_cell)] for k, v in profile_rows],
        colWidths=[42 * mm, 128 * mm],
    )
    profile_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1d3557")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b7c5d3")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7fb")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(profile_table)
    story.append(
        Paragraph(
            _narrative_or_fallback(
                report_text.get("company_profile"),
                "Das Unternehmensprofil zeigt ein stark technologiegetriebenes Umfeld mit Fokus auf skalierbare Innovationspartnerschaften.",
            ),
            body,
        )
    )

    # IV. Innovationsziele
    story.append(Paragraph("III. Strategische Innovationsziele", h1))
    story.append(
        Paragraph(
            _narrative_or_fallback(
                report_text.get("innovation_goals"),
                "Die strategischen Innovationsziele bilden die Grundlage für die Startup-Selektion.",
            ),
            body,
        )
    )
    goal_rows = [[Paragraph(f"• {clean_text(g)}", body_small)] for g in (innovation_goals or ["n/a"])]
    goals_box = Table(goal_rows, colWidths=[170 * mm])
    goals_box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eef6ff")),
                ("BOX", (0, 0), (-1, -1), 0.45, colors.HexColor("#7a9cc6")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(goals_box)

    # V. Gap-Analyse
    story.append(Paragraph("IV. Gap-Analyse", h1))
    story.append(
        Paragraph(
            _narrative_or_fallback(
                report_text.get("gap_analysis"),
                "Die Gap-Analyse zeigt priorisierte Handlungsbedarfe für eine wirksame Startup-Kooperation.",
            ),
            body,
        )
    )
    gap_table_rows = [["Identifizierte Herausforderung", "Bedeutung"]]
    for gap in (identified_gaps or ["Keine belastbaren Gaps identifiziert"]):
        gap_table_rows.append([clean_text(gap), _gap_significance(gap)])
    gap_table = Table(
        [[Paragraph(clean_text(r[0]), table_header), Paragraph(clean_text(r[1]), table_header)] for r in gap_table_rows[:1]]
        + [[Paragraph(clean_text(r[0]), table_cell), Paragraph(clean_text(r[1]), table_cell)] for r in gap_table_rows[1:]],
        colWidths=[85 * mm, 85 * mm],
    )
    gap_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1d3557")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b7c5d3")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7fb")]),
            ]
        )
    )
    story.append(gap_table)

    # VI. Strategische Startup-Suchfelder
    story.append(Paragraph("V. Strategische Startup-Suchfelder", h1))
    story.append(
        Paragraph(
            _narrative_or_fallback(
                report_text.get("startup_search_fields"),
                "Die Suchfelder priorisieren die Bereiche mit dem höchsten strategischen Hebel.",
            ),
            body,
        )
    )
    search_box_rows = [[Paragraph(f"• {clean_text(field)}", body_small)] for field in (startup_search_fields or ["n/a"])]
    search_box = Table(search_box_rows, colWidths=[170 * mm])
    search_box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#edf8f2")),
                ("BOX", (0, 0), (-1, -1), 0.45, colors.HexColor("#7aa874")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(search_box)

    # VII. Startup-Ranking
    story.append(Paragraph("VI. Startup-Ranking", h1))
    story.append(
        Paragraph(
            _narrative_or_fallback(
                report_text.get("recommended_startups"),
                "Die folgende Tabelle zeigt die priorisierten Startup-Kandidaten inklusive Relevanzbewertung.",
            ),
            body,
        )
    )

    ranking_rows = [["Rank", "Startup", "Relevanz", "Kurzbeschreibung"]]
    if not recommended:
        ranking_rows.append(["-", "Keine empfohlenen Startups verfügbar", "-", "-"])
    for item in recommended:
        if not isinstance(item, dict):
            continue
        score = float(item.get("relevance_score") or 0.0)
        ranking_rows.append(
            [
                str(int(item.get("rank") or 0)),
                clean_text(item.get("startup_name") or item.get("name") or "n/a"),
                _score_bar(score),
                clean_text(item.get("short_description") or "n/a"),
            ]
        )

    ranking_table = Table(
        [[Paragraph(clean_text(x), table_header) for x in ranking_rows[0]]]
        + [[Paragraph(clean_text(x), table_cell) for x in row] for row in ranking_rows[1:]],
        colWidths=[14 * mm, 28 * mm, 34 * mm, 80 * mm],
        repeatRows=1,
    )
    ranking_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1d3557")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b7c5d3")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (0, -1), "CENTER"),
                ("ALIGN", (2, 1), (2, -1), "LEFT"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7fb")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(ranking_table)
    story.append(Spacer(1, 6))

    # VIII. Detailprofile der Startups
    story.append(Paragraph("VII. Detailprofile der Startups", h1))
    for item in recommended:
        if not isinstance(item, dict):
            continue

        profile = item.get("profile") if isinstance(item.get("profile"), dict) else {}
        name = clean_text(item.get("startup_name") or profile.get("name") or "Startup")

        card_header = Table([[Paragraph(name, ParagraphStyle("CardTitle", parent=h2, fontSize=12.5, textColor=colors.white))]], colWidths=[170 * mm])
        card_header.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#2a4d69")), ("LEFTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
        story.append(card_header)

        website = clean_text(profile.get("website") or profile.get("url") or "")
        website_render = f"<link href='{website}' color='blue'>{website}</link>" if website else "n/a"

        details = [
            ["Gründungsjahr", clean_text(profile.get("founding_year")) or "n/a"],
            ["Standort", clean_text(profile.get("location")) or "n/a"],
            ["Technologiefokus", ", ".join(safe_list_str(profile.get("technology_focus"))) or "n/a"],
            ["Beschreibung", clean_text(profile.get("description") or item.get("short_description")) or "n/a"],
            ["Begründung der Relevanz", clean_text(profile.get("why_relevant")) or "n/a"],
            ["Webseite", website_render],
            ["Relevance Score", f"{float(item.get('relevance_score') or profile.get('relevance_score') or 0.0):.2f}"],
        ]

        detail_table = Table(
            [[Paragraph("Kategorie", table_header), Paragraph("Inhalt", table_header)]]
            + [[Paragraph(clean_text(k), table_cell), Paragraph(v if k == "Webseite" else clean_text(v), table_cell)] for k, v in details],
            colWidths=[45 * mm, 125 * mm],
        )
        detail_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1d3557")),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b7c5d3")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7fb")]),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(detail_table)
        story.append(Spacer(1, 6))

    # IX. Fazit und nächste Schritte
    story.append(Paragraph("VIII. Fazit und nächste Schritte", h1))
    story.append(
        Paragraph(
            _narrative_or_fallback(
                report_text.get("conclusion_next_steps"),
                "Die priorisierten Startups bieten kurzfristige Ansatzpunkte für Pilotprojekte und strategische Kooperationen.",
            ),
            body,
        )
    )
    next_steps = [
        "Kurzprüfung der identifizierten Startups mit Fokus auf strategische Passung und Umsetzbarkeit.",
        "Kontaktaufnahme mit den Top-Kandidaten und Definition eines strukturierten Evaluationsprozesses.",
        "Durchführung von Pilotprojekten in den priorisierten Innovationsfeldern.",
        "Aufbau eines kontinuierlichen Startup-Scouting-Prozesses mit klaren Governance-Regeln.",
    ]
    for idx, step in enumerate(next_steps, start=1):
        story.append(Paragraph(f"{idx}. {step}", body_small))

    doc.build(story, onFirstPage=_draw_page_number, onLaterPages=_draw_page_number)
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
    company_logo = _resolve_optional_asset_path(req.company_logo_path, user_root, work_root)
    tool_logo = _resolve_optional_asset_path(req.tool_logo_path, user_root, work_root)

    bytes_written = _render_report_pdf(
        report=report,
        output_file=output,
        title=req.title,
        subtitle=req.subtitle,
        company_name_override=req.company_name,
        report_year=req.report_year,
        created_by=req.created_by,
        company_logo=company_logo,
        tool_logo=tool_logo,
    )

    return PdfReportResult(
        output_path=str(req.output_path),
        bytes_written=bytes_written,
        title=req.title,
        text_preview="PDF generated with title page, structured sections, professional tables, ranking visuals and final recommendations.",
    )
