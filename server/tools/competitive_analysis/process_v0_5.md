# Process v0.5 (ohne Quality-Gates)

Diese Pipeline nutzt nur aktuelle Tools aus `server/tools/competitive_analysis` (kein `backup`, keine Quality-Gates).

## Schritt 1: Dokument importieren
Input: Quelldatei (PDF/DOCX/TXT etc.)  
Tool: `competitive_parse_document`  
Output: `parsed_doc.json`

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
        "path": "parsed_doc.json",
        "content": "{{steps[0].payload}}",
        "overwrite": true
      }
    }
  ]
}
```

## Schritt 2: Product Profile extrahieren
Input: `parsed_doc.json`  
Tool: `competitive_extract_product_profile_v0_2`  
Output: `product_profile.json`

```json
{
  "steps": [
    {
      "tool": "competitive_extract_product_profile_v0_2",
      "args": {
        "parsed_doc": null,
        "parsed_doc_path": "parsed_doc.json",
        "provider": "openai",
        "max_context_chars": 18000
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "product_profile.json",
        "content": "{{steps[0].payload}}",
        "overwrite": true
      }
    }
  ]
}
```

## Schritt 3: Analyseplan erzeugen
Input: `product_profile.json`  
Tool: `competitive_generate_analysis_plan_v0_2`  
Output: `analysis_plan_v0_2.json`

```json
{
  "steps": [
    {
      "tool": "competitive_generate_analysis_plan_v0_2",
      "args": {
        "product_profile": null,
        "product_profile_path": "product_profile.json",
        "provider": "openai",
        "max_context_chars": 14000
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "analysis_plan_v0_2.json",
        "content": "{{steps[0].payload}}",
        "overwrite": true
      }
    }
  ]
}
```

## Schritt 4: Wettbewerbsprodukte suchen
Input: `analysis_plan_v0_2.json`  
Tool: `competitor_search_v0_5`  
Output: `competitor_search_results_v0_5.json`

```json
{
  "steps": [
    {
      "tool": "competitor_search_v0_5",
      "args": {
        "analysis_plan": null,
        "analysis_plan_path": "analysis_plan_v0_2.json",
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
        "path": "competitor_search_results_v0_5.json",
        "content": "{{steps[0].payload}}",
        "overwrite": true
      }
    }
  ]
}
```

## Schritt 5: Wettbewerberprofile anreichern
Input: `competitor_search_results_v0_5.json` + `product_profile.json`  
Tool: `competitor_profile_extraction_v0_5`  
Output: `competitor_profile_results_v0_5.json`

```json
{
  "steps": [
    {
      "tool": "competitor_profile_extraction_v0_5",
      "args": {
        "competitor_search_results": null,
        "competitor_search_results_path": "competitor_search_results_v0_5.json",
        "product_profile": null,
        "product_profile_path": "product_profile.json",
        "provider": "brave",
        "max_competitors": 200,
        "include_page_fetch": true,
        "page_fetch_timeout_s": 8,
        "page_fetch_max_chars": 8000,
        "verbose_terminal": true
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "competitor_profile_results_v0_5.json",
        "content": "{{steps[0].payload}}",
        "overwrite": true
      }
    }
  ]
}
```

## Schritt 6: Feature Matrix + Gap/USP
Input: `product_profile.json` + `competitor_profile_results_v0_5.json`  
Tool: `feature_matrix_gap_analysis_v0_5`  
Output: `feature_matrix_gap_v0_5.json`

```json
{
  "steps": [
    {
      "tool": "feature_matrix_gap_analysis_v0_5",
      "args": {
        "product_profile": null,
        "product_profile_path": "product_profile.json",
        "competitor_profile_results": null,
        "competitor_profile_results_path": "competitor_profile_results_v0_5.json",
        "provider": "openai"
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "feature_matrix_gap_v0_5.json",
        "content": "{{steps[0].payload}}",
        "overwrite": true
      }
    }
  ]
}
```

## Schritt 7: SWOT + Positionierung
Input: `feature_matrix_gap_v0_5.json`  
Tool: `stratetic_analysis_swot_positioning_v0_5`  
Output: `strategic_analysis_v0_5.json`

```json
{
  "steps": [
    {
      "tool": "stratetic_analysis_swot_positioning_v0_5",
      "args": {
        "feature_matrix_gap": null,
        "feature_matrix_gap_path": "feature_matrix_gap_v0_5.json",
        "comparison_matrix": null,
        "gaps_and_usps": null,
        "evidences": null,
        "provider": "openai"
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "strategic_analysis_v0_5.json",
        "content": "{{steps[0].payload}}",
        "overwrite": true
      }
    }
  ]
}
```

## Schritt 8: Finalen Analysebericht erzeugen
Input-Artefakte:  
- `product_profile.json`  
- `competitor_profile_results_v0_5.json`  
- `feature_matrix_gap_v0_5.json`  
- `strategic_analysis_v0_5.json`  
Tool: `final_report_generator_v0_5`  
Output: `final_analysis_report_v0_5.json`

```json
{
  "steps": [
    {
      "tool": "final_report_generator_v0_5",
      "args": {
        "artifacts": null,
        "artifact_paths": {
          "product_profile": "product_profile.json",
          "competitor_profiles": "competitor_profile_results_v0_5.json",
          "comparison_matrix": "feature_matrix_gap_v0_5.json",
          "gaps_and_usps": "feature_matrix_gap_v0_5.json",
          "strategic_analysis": "strategic_analysis_v0_5.json",
          "swot": "strategic_analysis_v0_5.json",
          "positioning_data": "strategic_analysis_v0_5.json"
        },
        "provider": "openai",
        "max_chars_per_artifact": 10000
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "final_analysis_report_v0_5.json",
        "content": "{{steps[0].payload}}",
        "overwrite": true
      }
    }
  ]
}
```

## Schritt 9: PDF publizieren
Input: `final_analysis_report_v0_5.json`  
Tool: `competitive_publish_pdf_report`  
Output: `competition_analysis_report_v0_5.pdf`

```json
{
  "steps": [
    {
      "tool": "competitive_publish_pdf_report",
      "args": {
        "final_report": null,
        "final_report_path": "final_analysis_report_v0_5.json",
        "output_path": "competition_analysis_report_v0_5.pdf",
        "logo_path": "",
        "report_config_path": "",
        "chart_paths": [],
        "include_render_log": true,
        "render_log_path": "render_log_v0_5.json"
      }
    }
  ]
}
```

