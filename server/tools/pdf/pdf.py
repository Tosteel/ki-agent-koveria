from __future__ import annotations
from pathlib import Path
from fastapi import HTTPException

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


def export_text_pdf(output_file: Path, title: str, text: str) -> int:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="output_path must end with .pdf")

    c = canvas.Canvas(str(output_file), pagesize=A4)
    width, height = A4

    # einfacher Textlayout
    x = 40
    y = height - 60
    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, title[:120])
    y -= 30

    c.setFont("Helvetica", 10)
    line_height = 14
    max_chars = 110

    for raw_line in (text or "").splitlines():
        line = raw_line
        while len(line) > max_chars:
            chunk, line = line[:max_chars], line[max_chars:]
            if y < 60:
                c.showPage()
                y = height - 60
                c.setFont("Helvetica", 10)
            c.drawString(x, y, chunk)
            y -= line_height

        if y < 60:
            c.showPage()
            y = height - 60
            c.setFont("Helvetica", 10)
        c.drawString(x, y, line)
        y -= line_height

    c.save()
    return output_file.stat().st_size
