# Process v0.5 (ohne Quality-Gates)

Diese Pipeline nutzt nur aktuelle Tools aus `server.workflows.competitive_analysis` (kein `backup`, keine Quality-Gates).

## Schritt 1: Dokument importieren
Input: Quelldatei (PDF/DOCX/TXT etc.)  
Tool: `competitive_parse_document`  
Output: `step1_parsed_doc.json`

```json
{
  "steps": [
    {
      "tool": "competitive_parse_document",
      "args": {
        "path": "uploads/dein_dokument.pdf",
        "max_chars": 50000
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "step1_parsed_doc.json",
        "content": "{{steps[0].payload}}",
        "overwrite": true
      }
    }
  ]
}
```

## Schritt 2: Product Profile extrahieren
Input: `step1_parsed_doc.json`  
Tool: `competitive_extract_product_profile_v0_2`  
Output: `step2_product_profile.json`

Hinweis:
- Ab v0.5 wird zusätzlich `metric_features` erzeugt (z. B. Maße, Gewicht, physikalische Kennwerte).
- `performance_parameters` bleibt vollständig erhalten; `metric_features` ist eine dedizierte Teilmenge für spätere, wertbasierte Metrik-Analysen.

```json
{
  "steps": [
    {
      "tool": "competitive_extract_product_profile_v0_2",
      "args": {
        "parsed_doc": null,
        "parsed_doc_path": "step1_parsed_doc.json",
        "provider": "openai",
        "max_context_chars": 18000
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "step2_product_profile.json",
        "content": "{{steps[0].payload}}",
        "overwrite": true
      }
    }
  ]
}
```

## Schritt 3: Analyseplan erzeugen
Input: `step2_product_profile.json`  
Tool: `competitive_generate_analysis_plan_v0_2`  
Output: `step3_analysis_plan_v0_2.json`

```json
{
  "steps": [
    {
      "tool": "competitive_generate_analysis_plan_v0_2",
      "args": {
        "product_profile": null,
        "product_profile_path": "step2_product_profile.json",
        "provider": "openai",
        "max_context_chars": 14000
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "step3_analysis_plan_v0_2.json",
        "content": "{{steps[0].payload}}",
        "overwrite": true
      }
    }
  ]
}
```

## Schritt 4: Wettbewerbsprodukte suchen
Input: `step3_analysis_plan_v0_2.json`  
Tool: `competitor_search_v0_5`  
Output: `step4_competitor_search_results_v0_5.json`

```json
{
  "steps": [
    {
      "tool": "competitor_search_v0_5",
      "args": {
        "analysis_plan": null,
        "analysis_plan_path": "step3_analysis_plan_v0_2.json",
        "provider": "openai",
        "max_queries": 16,
        "per_query_results": 8,
        "max_candidates_to_check": 200,
        "verbose_terminal": true,
        "verbose_search_hits": false
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "step4_competitor_search_results_v0_5.json",
        "content": "{{steps[0].payload}}",
        "overwrite": true
      }
    }
  ]
}
```

## Schritt 4.1: Wettbewerbsprodukte verdichten (Top-N + Produktfilter)
Input: `step4_competitor_search_results_v0_5.json`  
Tool: `competitor_product_results_v0_6`  
Output: `step4_1_competitor_product_results_v0_6.json`

Dieser Schritt sortiert nach `relevance_score` (absteigend), filtert per LLM generische/nicht deutsch- oder englischsprachige Treffer heraus und nimmt danach die ersten `top_n` verbleibenden Produkte.

```json
{
  "steps": [
    {
      "tool": "competitor_product_results_v0_6",
      "args": {
        "competitor_search_results": null,
        "competitor_search_results_path": "step4_competitor_search_results_v0_5.json",
        "provider": "openai",
        "top_n": 20,
        "verbose_terminal": true
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "step4_1_competitor_product_results_v0_6.json",
        "content": "{{steps[0].payload}}",
        "overwrite": true
      }
    }
  ]
}
```

## Schritt 5.1: Wettbewerber-Plaintext mit Brave sammeln
Input: `step4_1_competitor_product_results_v0_6.json` + `step2_product_profile.json`  
Tool: `competitor_profile_text_v0_6`  
Output: `step5_1_competitor_profile_text_v0_6.json`

```json
{
  "steps": [
    {
      "tool": "competitor_profile_text_v0_6",
      "args": {
        "competitor_product_results": null,
        "competitor_product_results_path": "step4_1_competitor_product_results_v0_6.json",
        "product_profile": null,
        "product_profile_path": "step2_product_profile.json",
        "provider": "brave",
        "max_competitors": 200,
        "brave_enable_research": false,
        "brave_stream": true,
        "brave_language": "de",
        "brave_country": "DE",
        "verbose_terminal": true
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "step5_1_competitor_profile_text_v0_6.json",
        "content": "{{steps[0].payload}}",
        "overwrite": true
      }
    }
  ]
}
```

## Schritt 5.2: Plaintext in strukturiertes Profil überführen
Input: `step5_1_competitor_profile_text_v0_6.json` + `step2_product_profile.json`  
Tool: `competitor_profile_extraction_v0_6`  
Output: `step5_2_competitor_profile_extraction_v0_6.json`

Hinweis:
- Die Ausgabe enthält ebenfalls `metric_features`, damit Baseline und Wettbewerber dieselbe Feature-Kategorisierung nutzen.

```json
{
  "steps": [
    {
      "tool": "competitor_profile_extraction_v0_6",
      "args": {
        "competitor_profile_text": null,
        "competitor_profile_text_path": "step5_1_competitor_profile_text_v0_6.json",
        "product_profile": null,
        "product_profile_path": "step2_product_profile.json",
        "provider": "openai",
        "max_competitors": 200,
        "verbose_terminal": true
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "step5_2_competitor_profile_extraction_v0_6.json",
        "content": "{{steps[0].payload}}",
        "overwrite": true
      }
    }
  ]
}
```

## Schritt 6: Feature Matrix + Gap/USP
Input: `step2_product_profile.json` + `step5_2_competitor_profile_extraction_v0_6.json`  
Tool: `feature_matrix_gap_analysis_v0_5`  
Output: `step6_feature_matrix_gap_v0_5.json`

Restriktionen/Regeln in `feature_matrix_gap_analysis_v0_5`:

1. Preis-Gap als `missing_data`
- Für Preis/UVP wird bei fehlender Baseline-Erfassung `status="missing_data"` gesetzt (statt `absent`).

2. USP-Top-N Priorisierung
- USPs werden nach Seltenheit im Wettbewerb priorisiert.
- USP-Liste wird auf Top-10 begrenzt.
- `differentiators` werden auf diese priorisierten USPs reduziert.

3. Cluster-Fallback ohne Preis
- Wenn `avg_price` fehlt, aber `value_score` vorhanden ist:
- `>= 0.7` -> `performance_focused`
- `>= 0.4` -> `mainstream`
- `> 0` -> `feature_limited`
- `== 0` -> `data_gap`

4. Metric-Features als eigene Kategorie (kein binäres Gap/USP)
- `metric_features` wird separat in der Matrix geführt (`metric_dimensions`, `metric_features` je Row).
- Für `metric_features` werden keine Gaps/USPs nur aufgrund fehlender Nennung erzeugt.
- Metrik-Gap/USP wird nur wertbasiert berechnet, wenn:
- eine klare Zielrichtung vorliegt (aktuell: `lower-is-better`, z. B. Breite/Höhe/Tiefe/Länge/Gewicht),
- Baseline- und Wettbewerbswerte numerisch vergleichbar sind (inkl. generischer Einheiten-Normalisierung),
- und genügend Wettbewerbswerte vorhanden sind.

```json
{
  "steps": [
    {
      "tool": "feature_matrix_gap_analysis_v0_5",
      "args": {
        "product_profile": null,
        "product_profile_path": "step2_product_profile.json",
        "competitor_profile_results": null,
        "competitor_profile_results_path": "step5_2_competitor_profile_extraction_v0_6.json",
        "provider": "openai"
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "step6_feature_matrix_gap_v0_5.json",
        "content": "{{steps[0].payload}}",
        "overwrite": true
      }
    }
  ]
}
```

## Schritt 7: SWOT + Positionierung
Input: `step6_feature_matrix_gap_v0_5.json`  
Tool: `stratetic_analysis_swot_positioning_v0_5`  
Output: `step7_strategic_analysis_v0_5.json`

```json
{
  "steps": [
    {
      "tool": "stratetic_analysis_swot_positioning_v0_5",
      "args": {
        "feature_matrix_gap": null,
        "feature_matrix_gap_path": "step6_feature_matrix_gap_v0_5.json",
        "comparison_matrix": null,
        "gaps_and_usps": null,
        "evidences": null,
        "provider": "openai"
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "step7_strategic_analysis_v0_5.json",
        "content": "{{steps[0].payload}}",
        "overwrite": true
      }
    }
  ]
}
```

## Schritt 8: Finalen Analysebericht erzeugen
Input-Artefakte:  
- `step2_product_profile.json`  
- `step5_2_competitor_profile_extraction_v0_6.json`  
- `step6_feature_matrix_gap_v0_5.json`  
- `step7_strategic_analysis_v0_5.json`  
Tool: `final_report_generator_v0_5`  
Output: `step8_final_analysis_report_v0_5.json`

Hinweis:
- Die Feature-Matrix-Sektion übernimmt `metric_dimensions` als `metric::...`-Dimensionen.
- `present_features` enthält zusätzlich erkannte `metric_features`.

```json
{
  "steps": [
    {
      "tool": "final_report_generator_v0_5",
      "args": {
        "artifacts": null,
        "artifact_paths": {
          "product_profile": "step2_product_profile.json",
          "competitor_profiles": "step5_2_competitor_profile_extraction_v0_6.json",
          "comparison_matrix": "step6_feature_matrix_gap_v0_5.json",
          "gaps_and_usps": "step6_feature_matrix_gap_v0_5.json",
          "strategic_analysis": "step7_strategic_analysis_v0_5.json",
          "swot": "step7_strategic_analysis_v0_5.json",
          "positioning_data": "step7_strategic_analysis_v0_5.json"
        },
        "provider": "openai",
        "max_chars_per_artifact": 10000
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "step8_final_analysis_report_v0_5.json",
        "content": "{{steps[0].payload}}",
        "overwrite": true
      }
    }
  ]
}
```

## Schritt 9: PDF publizieren
Input: `step8_final_analysis_report_v0_5.json`  
Tool: `competitive_publish_pdf_report`  
Output: `step9_competition_analysis_report_v0_5.pdf`

```json
{
  "steps": [
    {
      "tool": "competitive_publish_pdf_report",
      "args": {
        "final_report": null,
        "final_report_path": "step8_final_analysis_report_v0_5.json",
        "output_path": "step9_competition_analysis_report_v0_5.pdf",
        "logo_path": "",
        "report_config_path": "",
        "chart_paths": [],
        "include_render_log": true,
        "render_log_path": "step9_render_log_v0_5.json"
      }
    }
  ]
}
```
