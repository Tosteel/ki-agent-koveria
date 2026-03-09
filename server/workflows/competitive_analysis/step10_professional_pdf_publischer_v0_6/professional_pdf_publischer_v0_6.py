from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from reportlab.graphics.shapes import Circle, Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _safe_join(base: Path, rel_path: str) -> Path:
    rel = Path(str(rel_path or "").strip().lstrip("/"))
    if rel.is_absolute() or ".." in rel.parts:
        raise HTTPException(status_code=400, detail=f"Invalid path: {rel_path}")
    p = (base / rel).resolve()
    base_resolved = base.resolve()
    if not str(p).startswith(str(base_resolved)):
        raise HTTPException(status_code=400, detail=f"Invalid path: {rel_path}")
    return p


def _resolve_read_path(user_root: Path, work_root: Path, rel_path: str) -> Path:
    rel = Path(str(rel_path or "").strip().lstrip("/"))
    if rel.is_absolute() or ".." in rel.parts:
        raise HTTPException(status_code=400, detail=f"Invalid path: {rel_path}")
    uploads_root = (user_root / "uploads").resolve()
    uploads_root.mkdir(parents=True, exist_ok=True)

    parts = list(rel.parts)
    candidates: List[Path] = []
    if parts and parts[0] == "uploads":
        candidates.append((uploads_root / Path(*parts[1:])).resolve())
    elif parts and parts[0] == "work":
        candidates.append((work_root / Path(*parts[1:])).resolve())
    else:
        candidates.append((work_root / rel).resolve())
        candidates.append((uploads_root / rel).resolve())

    for c in candidates:
        if c.exists() and c.is_file() and (user_root in c.parents or c == user_root):
            return c
    raise HTTPException(status_code=404, detail=f"File not found: {rel_path}")


def _load_json_from_path(user_root: Path, work_root: Path, rel_path: str) -> Dict[str, Any]:
    p = _resolve_read_path(user_root, work_root, rel_path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in {rel_path}") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail=f"JSON root must be object: {rel_path}")
    return data


def _extract_final_report(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "final_report" in payload and isinstance(payload.get("final_report"), dict):
        return payload["final_report"]

    outputs = payload.get("outputs")
    if isinstance(outputs, list) and outputs:
        first = outputs[0] if isinstance(outputs[0], dict) else {}
        step_payload = first.get("payload") if isinstance(first.get("payload"), dict) else {}
        final_report = step_payload.get("final_report") if isinstance(step_payload.get("final_report"), dict) else None
        if isinstance(final_report, dict):
            return final_report

    return payload


def _load_final_report(
    *,
    final_report: Optional[Dict[str, Any]],
    final_report_path: Optional[str],
    user_root: Path,
    work_root: Path,
) -> Dict[str, Any]:
    if isinstance(final_report, dict) and final_report:
        return _extract_final_report(final_report)
    if not (final_report_path or "").strip():
        raise HTTPException(status_code=400, detail="Missing final_report input")
    payload = _load_json_from_path(user_root, work_root, str(final_report_path))
    core = _extract_final_report(payload)
    if not isinstance(core, dict) or not core:
        raise HTTPException(status_code=400, detail="Could not extract final_report core from input")
    return core


def _load_optional_config(user_root: Path, work_root: Path, report_config_path: Optional[str]) -> Dict[str, Any]:
    if not (report_config_path or "").strip():
        return {}
    try:
        data = _load_json_from_path(user_root, work_root, str(report_config_path))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _cell_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        return ", ".join(str(x).strip() for x in v if str(x or "").strip())
    if isinstance(v, dict):
        return ", ".join(f"{k}: {v2}" for k, v2 in v.items())
    return str(v)


def _escape_para_text(s: str) -> str:
    txt = str(s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return txt


def _as_table_cell(v: Any, style: ParagraphStyle) -> Any:
    if isinstance(v, Paragraph):
        return v
    txt = _escape_para_text(_cell_text(v))
    return Paragraph(txt, style)


def _table(data: List[List[Any]], col_widths: Optional[List[float]] = None) -> Table:
    styles = getSampleStyleSheet()
    hdr_style = ParagraphStyle(
        "TblHeader",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        wordWrap="CJK",
    )
    body_style = ParagraphStyle(
        "TblBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.2,
        wordWrap="CJK",
    )

    cooked: List[List[Any]] = []
    for r_idx, row in enumerate(data):
        row_cells: List[Any] = []
        for cell in row:
            row_cells.append(_as_table_cell(cell, hdr_style if r_idx == 0 else body_style))
        cooked.append(row_cells)

    t = Table(cooked, colWidths=col_widths)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f3f8")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("WORDWRAP", (0, 0), (-1, -1), "CJK"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return t


def _draw_positioning(points: List[Dict[str, Any]], axis_x: str, axis_y: str) -> Drawing:
    w, h = 420, 240
    pad = 40
    d = Drawing(w, h)
    d.add(Rect(0, 0, w, h, strokeColor=colors.HexColor("#9ca3af"), fillColor=colors.white, strokeWidth=0.8))
    d.add(Line(pad, pad, w - pad, pad, strokeColor=colors.black, strokeWidth=1))
    d.add(Line(pad, pad, pad, h - pad, strokeColor=colors.black, strokeWidth=1))
    d.add(String(w / 2, 12, axis_x, fontSize=9))
    d.add(String(5, h / 2, axis_y, fontSize=9))

    def _f(v: Any, d: float = 0.0) -> float:
        try:
            return float(v)
        except Exception:
            return d

    priced_points = []
    missing_price_points = []
    for p in points:
        typ = str(p.get("point_type") or "competitor").lower()
        if typ.endswith("_missing_price"):
            missing_price_points.append(p)
        else:
            priced_points.append(p)

    xs = [_f(p.get("x"), 0.0) for p in priced_points] or [0.0, 1.0]
    ys = [_f(p.get("y"), 0.0) for p in points] or [0.0]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(1e-6, max_x - min_x)
    span_y = max(1e-6, max_y - min_y)

    for p in priced_points[:40]:
        name = str(p.get("name") or "")[:26]
        x = _f(p.get("x"), 0.0)
        y = _f(p.get("y"), 0.0)
        typ = str(p.get("point_type") or "competitor").lower()

        px = pad + ((x - min_x) / span_x) * (w - 2 * pad)
        py = pad + ((y - min_y) / span_y) * (h - 2 * pad)
        col = colors.HexColor("#0f766e") if typ == "own" else colors.HexColor("#1d4ed8")
        d.add(Circle(px, py, 4.0, fillColor=col, strokeColor=col))
        d.add(String(px + 5, py + 3, name, fontSize=7, fillColor=colors.HexColor("#111827")))

    if missing_price_points:
        mx = pad + 6
        d.add(String(mx + 10, h - pad + 10, "Preis n/a", fontSize=7, fillColor=colors.HexColor("#6b7280")))
        for idx, p in enumerate(missing_price_points[:20]):
            name = str(p.get("name") or "")[:22]
            y = _f(p.get("y"), 0.0)
            py = pad + ((y - min_y) / span_y) * (h - 2 * pad)
            py += ((idx % 3) - 1) * 4
            typ = str(p.get("point_type") or "competitor_missing_price").lower()
            col = colors.HexColor("#0f766e") if typ.startswith("own_") else colors.HexColor("#6b7280")
            d.add(Rect(mx - 2, py - 2, 4, 4, fillColor=col, strokeColor=col))
            d.add(String(mx + 6, py + 2, name, fontSize=7, fillColor=colors.HexColor("#374151")))

    return d


def _add_chapter_title(story: List[Any], txt: str, style: ParagraphStyle) -> None:
    story.append(Paragraph(txt, style))
    story.append(Spacer(1, 0.20 * cm))


def publish_competition_pdf_v0_6(
    *,
    final_report: Optional[Dict[str, Any]],
    final_report_path: Optional[str],
    output_path: str,
    logo_path: Optional[str],
    report_config_path: Optional[str],
    chart_paths: List[str],
    include_render_log: bool,
    render_log_path: str,
    user_root: Path,
    work_root: Path,
) -> Dict[str, Any]:
    core = _load_final_report(
        final_report=final_report,
        final_report_path=final_report_path,
        user_root=user_root.resolve(),
        work_root=work_root.resolve(),
    )
    cfg = _load_optional_config(user_root.resolve(), work_root.resolve(), report_config_path)

    warnings: List[str] = []

    out_file = _safe_join(work_root.resolve(), output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=16, leading=19)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=14)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontName="Helvetica", fontSize=9, leading=12)

    title = str(cfg.get("title") or "Competition Analysis Report v0.6")
    subtitle = str(cfg.get("subtitle") or "Strategic Competitive Intelligence")

    product_brief = core.get("product_profile_brief") if isinstance(core.get("product_profile_brief"), dict) else {}
    competitor_overview = core.get("competitor_overview") if isinstance(core.get("competitor_overview"), dict) else {}
    feature_matrix = core.get("feature_matrix_section") if isinstance(core.get("feature_matrix_section"), dict) else {}
    gap_usp = core.get("gap_usp_analysis") if isinstance(core.get("gap_usp_analysis"), dict) else {}
    swot = core.get("swot") if isinstance(core.get("swot"), dict) else {}
    positioning = core.get("positioning_diagram") if isinstance(core.get("positioning_diagram"), dict) else {}
    recommendations = core.get("strategic_recommendations") if isinstance(core.get("strategic_recommendations"), list) else []
    executive_summary = core.get("executive_summary") if isinstance(core.get("executive_summary"), list) else []
    sections = core.get("report_sections_v0_6") if isinstance(core.get("report_sections_v0_6"), dict) else {}

    product_name = str(product_brief.get("product_name") or "Unknown Product").strip() or "Unknown Product"

    story: List[Any] = []

    # Cover
    story.append(Paragraph(title, h1))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(subtitle, styles["Italic"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(f"Produkt: {product_name}", body))
    story.append(Paragraph(f"Datum: {datetime.now().strftime('%Y-%m-%d')}", body))
    if cfg.get("run_id"):
        story.append(Paragraph(f"Run-ID: {cfg.get('run_id')}", body))
    if logo_path:
        try:
            lp = _resolve_read_path(user_root.resolve(), work_root.resolve(), logo_path)
            story.append(Spacer(1, 0.3 * cm))
            story.append(Image(str(lp), width=4.0 * cm, height=2.0 * cm))
        except Exception as exc:
            warnings.append(f"Logo not embedded: {exc}")

    story.append(Spacer(1, 0.6 * cm))

    # TOC
    _add_chapter_title(story, "Inhaltsverzeichnis", h2)
    toc = [
        "1. Executive Summary",
        "2. Produktprofil",
        "3. Wettbewerbsuebersicht",
        "4. Feature Matrix",
        "5. Gap/USP-Analyse",
        "6. SWOT",
        "7. Positionierung (2x2)",
        "8. Strategische Empfehlungen",
    ]
    for t in toc:
        story.append(Paragraph(t, body))
    story.append(Spacer(1, 0.4 * cm))

    # 1 Executive Summary (flow text)
    _add_chapter_title(story, "1. Executive Summary", h2)
    exec_text = str(sections.get("executive_summary_text") or "").strip()
    if not exec_text and executive_summary:
        exec_text = " ".join(str(x).strip() for x in executive_summary if str(x or "").strip())
    story.append(Paragraph(_escape_para_text(exec_text or "Keine Executive Summary verfuegbar."), body))
    story.append(Spacer(1, 0.25 * cm))

    # 2 Produktprofil (German labels with claims in table)
    _add_chapter_title(story, "2. Produktprofil", h2)
    produktprofil_de = sections.get("produktprofil_de") if isinstance(sections.get("produktprofil_de"), dict) else {}
    felder = produktprofil_de.get("felder") if isinstance(produktprofil_de.get("felder"), list) else []

    if not felder:
        felder = [
            {"label": "Produktname", "value": product_brief.get("product_name")},
            {"label": "Hersteller", "value": product_brief.get("manufacturer")},
            {"label": "Kategorie", "value": product_brief.get("category")},
            {"label": "Zielsegmente", "value": product_brief.get("target_segments")},
            {"label": "Use Cases", "value": product_brief.get("use_cases")},
            {"label": "Schluessel-Differenzierer", "value": product_brief.get("key_differentiators")},
            {"label": "Claims", "value": product_brief.get("top_claims")},
            {"label": "Feature-Anzahl", "value": product_brief.get("feature_count")},
        ]

    pp_rows = [["Feld", "Wert"]]
    for f in felder:
        if not isinstance(f, dict):
            continue
        pp_rows.append([_cell_text(f.get("label")), _cell_text(f.get("value"))])
    story.append(_table(pp_rows, col_widths=[5.0 * cm, 11.0 * cm]))
    story.append(Spacer(1, 0.25 * cm))

    # 3 Wettbewerbsuebersicht
    _add_chapter_title(story, "3. Wettbewerbsuebersicht", h2)
    story.append(Paragraph(_escape_para_text(_cell_text(sections.get("competitor_overview_intro_text") or "")), body))
    story.append(Spacer(1, 0.1 * cm))
    comp_table = [["Modell", "Cluster", "Relevance", "Similarity", "Kurzprofil", "URL"]]
    for row in (competitor_overview.get("table") or [])[:30]:
        if not isinstance(row, dict):
            continue
        comp_table.append(
            [
                _cell_text(row.get("name")),
                _cell_text(row.get("cluster")),
                _cell_text(row.get("relevance_score")),
                _cell_text(row.get("similarity_score")),
                _cell_text(row.get("short_profile")),
                _cell_text(row.get("url")),
            ]
        )
    if len(comp_table) == 1:
        comp_table.append(["-", "-", "-", "-", "Keine Wettbewerberdaten", "-"])
    story.append(_table(comp_table, col_widths=[2.9 * cm, 2.0 * cm, 1.3 * cm, 1.3 * cm, 5.5 * cm, 3.6 * cm]))
    story.append(Spacer(1, 0.25 * cm))

    # 4 Feature Matrix
    _add_chapter_title(story, "4. Feature Matrix", h2)
    story.append(Paragraph(_escape_para_text(_cell_text(sections.get("feature_matrix_performance_text"))), body))
    story.append(Spacer(1, 0.05 * cm))
    story.append(Paragraph(_escape_para_text(_cell_text(sections.get("feature_matrix_soft_text"))), body))
    story.append(Spacer(1, 0.05 * cm))
    story.append(Paragraph(_escape_para_text(_cell_text(sections.get("feature_matrix_price_text"))), body))
    story.append(Spacer(1, 0.05 * cm))
    story.append(Paragraph(_escape_para_text(_cell_text(sections.get("feature_matrix_weighting_text"))), body))
    story.append(Spacer(1, 0.1 * cm))

    fm_table = [["Wettbewerber", "Cluster", "Preis", "Value", "Vorhandene Features"]]
    for r in (feature_matrix.get("rows") or [])[:30]:
        if not isinstance(r, dict):
            continue
        fm_table.append(
            [
                _cell_text(r.get("competitor")),
                _cell_text(r.get("cluster")),
                _cell_text(r.get("avg_price")),
                _cell_text(r.get("value_score")),
                _cell_text(r.get("present_features")),
            ]
        )
    if len(fm_table) == 1:
        fm_table.append(["-", "-", "-", "-", "Keine Matrixdaten"])
    story.append(_table(fm_table, col_widths=[3.4 * cm, 2.1 * cm, 2.2 * cm, 1.7 * cm, 6.2 * cm]))
    story.append(Spacer(1, 0.25 * cm))

    # 5 Gap/USP: two flow paragraphs
    _add_chapter_title(story, "5. Gap/USP-Analyse", h2)
    story.append(Paragraph(_escape_para_text(_cell_text(sections.get("gaps_text"))), body))
    story.append(Spacer(1, 0.08 * cm))
    story.append(Paragraph(_escape_para_text(_cell_text(sections.get("usps_text"))), body))
    story.append(Spacer(1, 0.25 * cm))

    # 6 SWOT: four flow paragraphs
    _add_chapter_title(story, "6. SWOT", h2)
    story.append(Paragraph("Staerken", styles["Heading3"]))
    story.append(Paragraph(_escape_para_text(_cell_text(sections.get("swot_strengths_text"))), body))
    story.append(Spacer(1, 0.06 * cm))
    story.append(Paragraph("Schwaechen", styles["Heading3"]))
    story.append(Paragraph(_escape_para_text(_cell_text(sections.get("swot_weaknesses_text"))), body))
    story.append(Spacer(1, 0.06 * cm))
    story.append(Paragraph("Chancen", styles["Heading3"]))
    story.append(Paragraph(_escape_para_text(_cell_text(sections.get("swot_opportunities_text"))), body))
    story.append(Spacer(1, 0.06 * cm))
    story.append(Paragraph("Gefahren", styles["Heading3"]))
    story.append(Paragraph(_escape_para_text(_cell_text(sections.get("swot_threats_text"))), body))
    story.append(Spacer(1, 0.25 * cm))

    # 7 Positionierung
    _add_chapter_title(story, "7. Positionierung (2x2)", h2)
    story.append(Paragraph(_escape_para_text(_cell_text(sections.get("positioning_intro_text"))), body))
    story.append(Spacer(1, 0.1 * cm))

    axis_x = _cell_text(positioning.get("axis_x") or "Preisniveau")
    axis_y = _cell_text(positioning.get("axis_y") or "Leistungs-/Wertbeitrag")
    points = positioning.get("points") if isinstance(positioning.get("points"), list) else []

    used_external_chart = False
    for cp in chart_paths[:2]:
        try:
            chart_file = _resolve_read_path(user_root.resolve(), work_root.resolve(), cp)
            story.append(Image(str(chart_file), width=15.0 * cm, height=7.0 * cm))
            used_external_chart = True
            break
        except Exception:
            continue

    if not used_external_chart:
        d = _draw_positioning([p for p in points if isinstance(p, dict)], axis_x, axis_y)
        story.append(d)

    story.append(Spacer(1, 0.1 * cm))
    story.append(Paragraph(_escape_para_text(_cell_text(sections.get("positioning_insights_text"))), body))
    story.append(Spacer(1, 0.25 * cm))

    # 8 Recommendations: intro text + table unchanged
    _add_chapter_title(story, "8. Strategische Empfehlungen", h2)
    story.append(Paragraph(_escape_para_text(_cell_text(sections.get("recommendations_intro_text"))), body))
    story.append(Spacer(1, 0.1 * cm))

    rec_tbl = [["Prioritaet", "Horizont", "Titel", "Massnahme", "Evidenz"]]
    for r in recommendations[:12]:
        if not isinstance(r, dict):
            continue
        rec_tbl.append(
            [
                _cell_text(r.get("priority")),
                _cell_text(r.get("horizon")),
                _cell_text(r.get("title")),
                _cell_text(r.get("action")),
                _cell_text(r.get("evidence_refs")),
            ]
        )
    if len(rec_tbl) == 1:
        rec_tbl.append(["-", "-", "Keine Empfehlungen", "-", "-"])
    story.append(_table(rec_tbl, col_widths=[1.8 * cm, 2.2 * cm, 3.2 * cm, 6.0 * cm, 3.0 * cm]))

    doc = SimpleDocTemplate(
        str(out_file),
        pagesize=A4,
        leftMargin=1.6 * cm,
        rightMargin=1.6 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title=title,
        author="KI-Agent-Koveria",
    )
    doc.build(story)

    render_log_rel = ""
    if include_render_log:
        rl = _safe_join(work_root.resolve(), render_log_path)
        rl.parent.mkdir(parents=True, exist_ok=True)
        log = {
            "generated_at": datetime.now().isoformat(),
            "warnings": warnings,
            "output_path": output_path,
            "sections": {
                "competitor_rows": len(competitor_overview.get("table") or []),
                "feature_matrix_rows": len(feature_matrix.get("rows") or []),
                "recommendations": len(recommendations),
            },
        }
        rl.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        render_log_rel = render_log_path

    return {
        "ok": True,
        "output_path": output_path,
        "bytes_written": out_file.stat().st_size,
        "render_log_path": render_log_rel,
        "warnings": warnings,
    }
