from __future__ import annotations

from typing import Any, Dict, List

from server.agent.policies import BASIC_TOOLS


_TOOL_DESCRIPTIONS: Dict[str, str] = {
    "read_file": "Dateien lesen.",
    "write_file": "Dateien schreiben.",
    "rag_knowledgebase": "Wissensbasis durchsuchen (RAG).",
    "llm_summarize": "Text zusammenfassen.",
    "llm_compose": "Text in eine kohärente Antwort umformulieren.",
    "pdf_export": "Text als PDF exportieren.",
    "ppt_export": "Text als PowerPoint exportieren.",
    "websearch_table": "Webrecherche mit strukturierten Ergebnissen.",
    "search_multitable": "Webrecherche mit strukturierten Ergebnissen (Alias).",
    "websearch_videoanalyzer": "Analysiert Web-/Video-Treffer anhand eines Prompts und ergänzt Spalten.",
    "websearch_videoanalizer": "Analysiert Web-/Video-Treffer anhand eines Prompts und ergänzt Spalten (Alias).",
    "langsearch": "Websuche mit LangSearch (Open/Free).",
    "google_search": "Websuche über Google Custom Search.",
    #"search_ebay": "Produktsuche auf eBay.",
    "view_website": "Inhalte auf einer einzelnen Website-Seite suchen.",
    "browse_website": "Website mit Navigation/Klicks über Unterseiten durchsuchen.",
    "send_mail": "E-Mails versenden (optional mit Anhang).",
    "competitor_search_v0_4": "Wettbewerber über Playwright+BeautifulSoup suchen und per LLM auf direkte Konkurrenz prüfen.",
}


def list_skills(*, include_descriptions: bool = True) -> Dict[str, Any]:
    skill_names: List[str] = sorted(BASIC_TOOLS)
    skills: List[Dict[str, str]] = []
    for name in skill_names:
        desc = _TOOL_DESCRIPTIONS.get(name, "Tool verfügbar.")
        if not include_descriptions:
            desc = ""
        skills.append({"name": name, "description": desc})

    if include_descriptions:
        lines = [f"- {item['name']}: {item['description']}" for item in skills]
    else:
        lines = [f"- {item['name']}" for item in skills]

    text = "Verfügbare Fähigkeiten:\n" + "\n".join(lines)
    return {
        "count": len(skills),
        "skills": skills,
        "text": text,
    }
