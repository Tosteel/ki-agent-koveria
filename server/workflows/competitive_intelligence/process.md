# Process: Competitive Intelligence

## Schritt 1.1: step1_1_market_trends_raw
Input: Marktkontext, optionale Suchquellen
Tool: `step1_1_market_trends_raw`
Output: `step1.1_market_trends_raw.json`

Zweck:
- Fuehrt eine Brave-Websuche als Raw-Retrieval aus.
- Wichtig: Im `messages[0].content` nur die einfache Suchquery (kein Systemprompt, keine Zusatzanweisung).

```json
{
  "steps": [
    {
      "tool": "step1_1_market_trends_raw",
      "args": {
        "market_context": "Kuechentrends 2026 in Baden-Wuerttemberg",
        "search_sources": ["Bekannte Medienhaeuser"],
        "brave_country": "IT",
        "brave_language": "it",
        "brave_enable_entities": true,
        "brave_enable_citations": true,
        "brave_enable_research": false
      }
    },
    {
      "tool": "file_write",
      "args": {
        "path": "step1.1_market_trends_raw.json",
        "content": "{{steps[0].payload.market_trends_raw}}",
        "overwrite": true
      }
    }
  ]
}
```

## Schritt 1.2: step1_2_market_trends_structured
Input: `step1.1_market_trends_raw.json` oder `market_trends_raw` inline, optional `provider` (`ionos`/`openai`)
Tool: `step1_2_market_trends_structured`
Output: `step1.2_market_trends_structured.json`

Zweck:
- Strukturiert den Raw-Text aus Schritt 1.1 pro Quelle.
- Ausgabe pro Quelle:
  `url` (Quelle), `originaltext` (per `web_fetch_page`), `originaltext_raw` (aus Schritt 1.1), `kernaussage` (LLM-Stichpunkte).

```json
{
  "steps": [
    {
      "tool": "step1_2_market_trends_structured",
      "args": {
        "market_trends_raw_path": "step1.1_market_trends_raw.json",
        "provider": "ionos",
        "use_web_fetch_page": true,
        "view_timeout_ms": 20000,
        "max_sources": 12,
        "max_chars_per_source": 120000,
        "summary_bullets": 4
      }
    },
    {
      "tool": "file_write",
      "args": {
        "path": "step1.2_market_trends_structured.json",
        "content": "{{steps[0].payload.market_trends_structured}}",
        "overwrite": true
      }
    }
  ]
}
```

## Schritt 1.3: step1_3_market_trends_summary
Input: `step1.2_market_trends_structured.json`, optional `provider` (`ionos`/`openai`)
Tool: `step1_3_market_trends_summary`
Output: `step1.3_market_trends_summary.json`

Zweck:
- Fasst die Kernaussagen aus Schritt 1.2 zu uebergreifenden Trend-Summaries zusammen.
- Hinterlegt pro Summary alle zugehoerigen Quellen-URLs (bei mehreren Quellen werden alle ausgegeben).

```json
{
  "steps": [
    {
      "tool": "step1_3_market_trends_summary",
      "args": {
        "market_trends_structured_path": "step1.2_market_trends_structured.json",
        "provider": "ionos",
        "similarity_threshold": 0.45,
        "max_summary_items": 12,
        "max_evidence_per_item": 6
      }
    },
    {
      "tool": "file_write",
      "args": {
        "path": "step1.3_market_trends_summary.json",
        "content": "{{steps[0].payload.market_trends_summary}}",
        "overwrite": true
      }
    }
  ]
}
```

## Schritt 2.2: step2_2_competitor_profile_raw
Input: Unternehmensliste
Tool: `step2_2_competitor_profile_raw`
Output: `step2.2_competitor_profile_raw.json`

Zweck:
- Pro Unternehmen 4 Brave-Suchqueries als Raw-Erhebung:
- Unternehmensprofil/Zielgruppe
- Angebote/Aktionen
- Bewertungen/Reichweite/Social Media
- Presse/Berichterstattung

```json
{
  "steps": [
    {
      "tool": "step2_2_competitor_profile_raw",
      "args": {
        "companies_path": "step2_companies.json",
        "provider": "brave",
        "brave_country": "DE",
        "brave_language": "de",
        "brave_enable_entities": true,
        "brave_enable_citations": true,
        "brave_enable_research": false
      }
    },
    {
      "tool": "file_write",
      "args": {
        "path": "step2.2_competitor_profile_raw.json",
        "content": "{{steps[0].payload.competitor_profile_raw}}",
        "overwrite": true
      }
    }
  ]
}
```

## Schritt 2.3: step2_3_competitor_profile_structured
Input: `step2.2_competitor_profile_raw.json`
Tool: `step2_3_competitor_profile_structured`
Output: `step2.3_competitor_profile_structured.json`

Zweck:
- Erzeugt pro Unternehmen ein strukturiertes Profil mit englischen Feldern:
  `company`, `website`, `region`, `company_profile_target_audience`, `offers_actions`, `ratings_reach`, `press_coverage`.
- Fuehrt Quellen (`source_urls`) pro Abschnitt mit.
- Laeuft heuristisch ohne LLM und bereinigt die Textbausteine in `summary`.

```json
{
  "steps": [
    {
      "tool": "step2_3_competitor_profile_structured",
      "args": {
        "competitor_profile_raw_path": "step2.2_competitor_profile_raw.json"
      }
    },
    {
      "tool": "file_write",
      "args": {
        "path": "step2.3_competitor_profile_structured.json",
        "content": "{{steps[0].payload.competitor_profile_structured}}",
        "overwrite": true
      }
    }
  ]
}
```

## Schritt 2.4: step2_4_competitor_trends
Input: `step2.3_competitor_profile_structured.json`, `step1.3_market_trends_summary.json`
Tool: `step2_4_competitor_trends`
Output: `step2.4_competitor_trends.json`

Zweck:
- Prueft pro Unternehmen, ob Trend-Keywords aus Schritt 1.3 auf der Unternehmenswebsite vorkommen.
- Nutzt `web_crawl_site` zum Website-Scan und erstellt `trend_matches` je Profil.
- Struktur bleibt kompatibel zu Schritt 2.3 (plus `trend_matches`), um spaeter leicht zu mergen.

```json
{
  "steps": [
    {
      "tool": "step2_4_competitor_trends",
      "args": {
        "competitor_profile_structured_path": "step2.3_competitor_profile_structured.json",
        "market_trends_summary_path": "step1.3_market_trends_summary.json"
      }
    },
    {
      "tool": "file_write",
      "args": {
        "path": "step2.4_competitor_trends.json",
        "content": "{{steps[0].payload.competitor_trends}}",
        "overwrite": true
      }
    }
  ]
}
```

## Schritt 3.1: step3_1_matrix
Input: `step2.4_competitor_trends.json`
Tool: `step3_1_matrix`
Output: `step3.1_matrix.json`

Zweck:
- Erzeugt pro Unternehmen eine LLM-basierte Stichpunkt-Auswertung je Kategorie als Tabellenbasis.
- Enthalten: Kundensegment, Aktionen, Ratings, Press/Coverage, Google-Wertung/-Anzahl, Social-Reichweite sowie Trends mit Match-Score und Suchbegriffen.

```json
{
  "steps": [
    {
      "tool": "step3_1_matrix",
      "args": {
        "competitor_trends_path": "step2.4_competitor_trends.json",
        "provider": "ionos",
        "max_bullets_per_section": 4,
        "max_trend_bullets": 2
      }
    },
    {
      "tool": "file_write",
      "args": {
        "path": "step3.1_matrix.json",
        "content": "{{steps[0].payload.matrix}}",
        "overwrite": true
      }
    }
  ]
}
```

## Schritt 3: competitor_matrix
Input: `step2.4_competitor_trends.json`
Tool: `competitor_matrix`
Output: `step3_competitor_matrix.json`

Zweck:
- Alle Wettbewerber tabellarisch vergleichen (z. B. Preissegment, Ratings, Trendfokus).
- Optional Rankings, Marktsegmente und Differenzierungsmerkmale erzeugen.

```json
{
  "steps": [
    {
      "tool": "competitor_matrix",
      "args": {
        "competitor_profiles_path": "step2.4_competitor_trends.json"
      }
    },
    {
      "tool": "file_write",
      "args": {
        "path": "step3_competitor_matrix.json",
        "content": "{{steps[0].payload.competitor_matrix}}",
        "overwrite": true
      }
    }
  ]
}
```

## Schritt 4: insights
Input: `step1.3_market_trends_summary.json`, `step2.4_competitor_trends.json`, `step3_competitor_matrix.json`
Tool: `insights`
Output: `step4_insights.json`

Zweck:
- Strategische Erkenntnisse zu Marktstruktur, Trendadoption, Marketingstrategien und Kundenfeedback ableiten.

```json
{
  "steps": [
    {
      "tool": "insights",
      "args": {
        "market_trends_summary_path": "step1.3_market_trends_summary.json",
        "competitor_profiles_path": "step2.4_competitor_trends.json",
        "competitor_matrix_path": "step3_competitor_matrix.json"
      }
    },
    {
      "tool": "file_write",
      "args": {
        "path": "step4_insights.json",
        "content": "{{steps[0].payload.insights}}",
        "overwrite": true
      }
    }
  ]
}
```

## Schritt 4.1: step4_1_insights
Input: `step3.1_matrix.json`
Tool: `step4_1_insights`
Output: `step4.1_insights.json`

Zweck:
- Leitet pro Fragestellung kompakte Insights aus Schritt 3.1 ab.
- Ausgabefelder:
  `customer_segment_insights`, `actions_insights`, `ratings_insights`, `trend_items_insights`, `competitor_comparison_insights`.

```json
{
  "steps": [
    {
      "tool": "step4_1_insights",
      "args": {
        "matrix_path": "step3.1_matrix.json",
        "provider": "ionos"
      }
    },
    {
      "tool": "file_write",
      "args": {
        "path": "step4.1_insights.json",
        "content": "{{steps[0].payload.insights}}",
        "overwrite": true
      }
    }
  ]
}
```

## Schritt 5: recommendations
Input: `step1.3_market_trends_summary.json`, `step3_competitor_matrix.json`, `step4_insights.json`
Tool: `recommendations`
Output: `step5_recommendations.json`

Zweck:
- Konkrete Handlungsempfehlungen in Kategorien wie Marketing, Positionierung, Sortiment und Online-Praesenz ableiten.

```json
{
  "steps": [
    {
      "tool": "recommendations",
      "args": {
        "market_trends_summary_path": "step1.3_market_trends_summary.json",
        "competitor_matrix_path": "step3_competitor_matrix.json",
        "insights_path": "step4_insights.json"
      }
    },
    {
      "tool": "file_write",
      "args": {
        "path": "step5_recommendations.json",
        "content": "{{steps[0].payload.recommendations}}",
        "overwrite": true
      }
    }
  ]
}
```

## Schritt 5.1: step5_1_recommendations
Input: `step3.1_matrix.json`
Tool: `step5_1_recommendations`
Output: `step5.1_recommendations.json`

Zweck:
- Leitet pro Kategorie Empfehlungen fuer das evaluierte Unternehmen ab (immer erstes Unternehmen in `step3.1`).
- Ausgabefelder:
  `customer_segements_recommendations`, `actions_recommendations`, `ratings_recommendations`, `trend_items_recommendations`, `competitor_comparison_recommendations`.

```json
{
  "steps": [
    {
      "tool": "step5_1_recommendations",
      "args": {
        "matrix_path": "step3.1_matrix.json",
        "provider": "ionos"
      }
    },
    {
      "tool": "file_write",
      "args": {
        "path": "step5.1_recommendations.json",
        "content": "{{steps[0].payload.recommendations}}",
        "overwrite": true
      }
    }
  ]
}
```

## Schritt 6: report_generator
Input: `step1.3_market_trends_summary.json`, `step2.4_competitor_trends.json`, `step3_competitor_matrix.json`, `step4_insights.json`, `step5_recommendations.json`
Tool: `report_generator`
Output: `step6_report.md`

Zweck:
- Ergebnisse in einer einheitlichen Berichtsstruktur zusammenfuehren.

```json
{
  "steps": [
    {
      "tool": "report_generator",
      "args": {
        "market_trends_summary_path": "step1.3_market_trends_summary.json",
        "competitor_profiles_path": "step2.4_competitor_trends.json",
        "competitor_matrix_path": "step3_competitor_matrix.json",
        "insights_path": "step4_insights.json",
        "recommendations_path": "step5_recommendations.json"
      }
    },
    {
      "tool": "file_write",
      "args": {
        "path": "step6_report.md",
        "content": "{{steps[0].payload.report}}",
        "overwrite": true
      }
    }
  ]
}
```

## Schritt 6.1: step6_1_final_report
Input:
- `step1.3_market_trends_summary.json`
- `step2.4_competitor_trends.json`
- `step3.1_matrix.json`
- `step4.1_insights.json`
- `step5.1_recommendations.json`
Tool: `step6_1_final_report`
Output: `step6.1_final_report.json`

Zweck:
- Erstellt den Gesamtbericht mit Kapiteln:
  1. Executive Summary (zuletzt erzeugt),
  2. Unternehmensprofile,
  3. Unternehmensmatrix (Unternehmen = Spalten, Variablen = Zeilen),
  4. Insight,
  5. Empfehlungen.

```json
{
  "steps": [
    {
      "tool": "step6_1_final_report",
      "args": {
        "market_trends_summary_path": "step1.3_market_trends_summary.json",
        "competitor_trends_path": "step2.4_competitor_trends.json",
        "matrix_path": "step3.1_matrix.json",
        "insights_path": "step4.1_insights.json",
        "recommendations_path": "step5.1_recommendations.json",
        "provider": "ionos"
      }
    },
    {
      "tool": "file_write",
      "args": {
        "path": "step6.1_final_report.json",
        "content": "{{steps[0].payload.final_report}}",
        "overwrite": true
      }
    }
  ]
}
```

## Schritt 7: pdf_export
Input: `step6_report.md`
Tool: `pdf_export`
Output: `step7_report.pdf`

Zweck:
- Finalen Report als PDF bereitstellen, optional mit Tabellen, Diagrammen und Quellen.

```json
{
  "steps": [
    {
      "tool": "pdf_export",
      "args": {
        "report_path": "step6_report.md"
      }
    },
    {
      "tool": "file_write",
      "args": {
        "path": "step7_report.pdf",
        "content": "{{steps[0].payload.pdf_base64}}",
        "overwrite": true
      }
    }
  ]
}
```

## Schritt 7.1: step7_1_pdf_export
Input: `step6.1_final_report.json`
Tool: `step7_1_pdf_export`
Output: `step7.1_report.pdf`

Zweck:
- Exportiert den finalen Report aus Schritt 6.1 als PDF.

```json
{
  "steps": [
    {
      "tool": "step7_1_pdf_export",
      "args": {
        "final_report_path": "step6.1_final_report.json",
        "output_path": "step7.1_report.pdf",
        "title": "Competitive Intelligence Report"
      }
    }
  ]
}
```

## Reihenfolge
`step1_1_market_trends_raw` -> `step1_2_market_trends_structured` -> `step1_3_market_trends_summary` -> `step2_2_competitor_profile_raw` -> `step2_3_competitor_profile_structured` -> `step2_4_competitor_trends` -> `competitor_matrix` -> `insights` -> `recommendations` -> `report_generator` -> `pdf_export`

## Fail-Fast-Regel
- Wenn ein Pflichtartefakt fehlt oder leer ist, darf der naechste Schritt nicht starten.
