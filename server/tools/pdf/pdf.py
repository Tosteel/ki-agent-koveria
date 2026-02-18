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


def read_pdf_text(pdf_file: Path, max_chars: int = 20000) -> tuple[str, int]:
    if pdf_file.suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="path must end with .pdf")
    if not pdf_file.exists() or not pdf_file.is_file():
        raise HTTPException(status_code=404, detail="PDF not found")

    reader_obj = None
    try:
        from pypdf import PdfReader  # type: ignore

        reader_obj = PdfReader(str(pdf_file))
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore

            reader_obj = PdfReader(str(pdf_file))
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail="PDF reader not available. Install 'pypdf' (or PyPDF2).",
            ) from exc

    pages = 0
    chunks: list[str] = []
    remaining = max(0, int(max_chars))
    for page in reader_obj.pages:
        pages += 1
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        if not txt:
            continue
        if len(txt) > remaining:
            chunks.append(txt[:remaining])
            remaining = 0
            break
        chunks.append(txt)
        remaining -= len(txt)
        if remaining <= 0:
            break

    text = "\n".join(chunks).strip()
    return text, pages
