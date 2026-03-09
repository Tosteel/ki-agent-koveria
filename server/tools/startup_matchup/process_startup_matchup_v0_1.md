# Startup Matchup Process v0.1

## Schritt 1: Workshop Analyse
Input: Workshop-Dokument (PDF/PPTX/DOCX/TXT) oder Freitext
Tool: `startup_matchup_step_1_workshop_analysis`
Output: `step1_workshop_analysis.json`

```json
{
  "steps": [
    {
      "tool": "startup_matchup_step_1_workshop_analysis",
      "args": {
        "workshop_document_path": "uploads/workshop_notes.pdf",
        "provider": "ionos"
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "step1_workshop_analysis.json",
        "content": "{steps[0].payload}",
        "overwrite": true
      }
    }
  ]
}
```

## Schritt 2: Unternehmensprofil
Input: `step1_workshop_analysis.json` + Firmenname
Tool: `startup_matchup_step_2_company_profile`
Output: `step2_company_profile.json`

```json
{
  "steps": [
    {
      "tool": "startup_matchup_step_2_company_profile",
      "args": {
        "workshop_analysis_path": "step1_workshop_analysis.json",
        "company_name": "Beispiel AG",
        "provider": "ionos",
        "brave_enable_research": false,
        "brave_stream": true
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "step2_company_profile.json",
        "content": "{steps[0].payload}",
        "overwrite": true
      }
    }
  ]
}
```

## Schritt 3: Gap Analyse
Input: `step1_workshop_analysis.json`, `step2_company_profile.json`
Tool: `startup_matchup_step_3_gap_analysis`
Output: `step3_gap_analysis.json`

```json
{
  "steps": [
    {
      "tool": "startup_matchup_step_3_gap_analysis",
      "args": {
        "workshop_analysis_path": "step1_workshop_analysis.json",
        "company_profile_path": "step2_company_profile.json",
        "provider": "ionos",
        "max_queries": 16
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "step3_gap_analysis.json",
        "content": "{steps[0].payload}",
        "overwrite": true
      }
    }
  ]
}
```

## Schritt 4: Startup Recherche
Input: `step3_gap_analysis.json`
Tool: `startup_matchup_step_4_startup_search`
Output: `step4_startup_candidates_raw.json`

```json
{
  "steps": [
    {
      "tool": "startup_matchup_step_4_startup_search",
      "args": {
        "gap_analysis_path": "step3_gap_analysis.json",
        "max_queries": 16,
        "per_query_results": 8,
        "brave_enable_research": false,
        "brave_stream": true
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "step4_startup_candidates_raw.json",
        "content": "{steps[0].payload}",
        "overwrite": true
      }
    }
  ]
}
```

## Schritt 4.1: Strukturierung der Startup-Suchergebnisse
Input: `step4_startup_candidates_raw.json`
Tool: `startup_matchup_step_4_1_startup_structuring`
Output: `step4_1_startup_structured.json`

```json
{
  "steps": [
    {
      "tool": "startup_matchup_step_4_1_startup_structuring",
      "args": {
        "startup_candidates_raw_path": "step4_startup_candidates_raw.json",
        "provider": "ionos"
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "step4_1_startup_structured.json",
        "content": "{steps[0].payload}",
        "overwrite": true
      }
    }
  ]
}
```

`curl`-Beispiel:
```bash
curl -X POST "http://localhost:8000/startup-matchup/step-4-1/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <API_KEY>" \
  -d '{
    "startup_candidates_raw_path": "step4_startup_candidates_raw.json",
    "provider": "ionos",
    "max_context_chars": 6000
  }'
```

## Schritt 5: Ranking und Relevanz
Input: `step4_1_startup_structured.json` (+ optional Step2/Step3)
Tool: `startup_matchup_step_5_startup_ranking`
Output: `step5_startup_ranked_list.json`

```json
{
  "steps": [
    {
      "tool": "startup_matchup_step_5_startup_ranking",
      "args": {
        "startup_structured_list_path": "step4_1_startup_structured.json",
        "company_profile_path": "step2_company_profile.json",
        "gap_analysis_path": "step3_gap_analysis.json",
        "top_k": 25
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "step5_startup_ranked_list.json",
        "content": "{steps[0].payload}",
        "overwrite": true
      }
    }
  ]
}
```

## Schritt 6: Deep Research Top-N
Input: `step5_startup_ranked_list.json`
Tool: `startup_matchup_step_6_startup_deep_research`
Output: `step6_startup_deep_profiles_raw.json`

```json
{
  "steps": [
    {
      "tool": "startup_matchup_step_6_startup_deep_research",
      "args": {
        "startup_ranked_list_path": "step5_startup_ranked_list.json",
        "top_n": 10,
        "brave_enable_research": false,
        "brave_stream": true
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "step6_startup_deep_profiles_raw.json",
        "content": "{steps[0].payload}",
        "overwrite": true
      }
    }
  ]
}
```

## Schritt 7: Strukturierte Startup Profile
Input: `step6_startup_deep_profiles_raw.json` (+ optional Step3/Step5)
Tool: `startup_matchup_step_7_startup_profiles`
Output: `step7_startup_profiles.json`

```json
{
  "steps": [
    {
      "tool": "startup_matchup_step_7_startup_profiles",
      "args": {
        "startup_deep_profiles_raw_path": "step6_startup_deep_profiles_raw.json",
        "gap_analysis_path": "step3_gap_analysis.json",
        "startup_ranked_list_path": "step5_startup_ranked_list.json",
        "provider": "ionos"
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "step7_startup_profiles.json",
        "content": "{steps[0].payload}",
        "overwrite": true
      }
    }
  ]
}
```

## Schritt 8: Finaler JSON Bericht
Input: Step1, Step2, Step3, Step5, Step7
Tool: `startup_matchup_step_8_final_report`
Output: `step8_final_report.json`

```json
{
  "steps": [
    {
      "tool": "startup_matchup_step_8_final_report",
      "args": {
        "workshop_analysis_path": "step1_workshop_analysis.json",
        "company_profile_path": "step2_company_profile.json",
        "gap_analysis_path": "step3_gap_analysis.json",
        "startup_ranked_list_path": "step5_startup_ranked_list.json",
        "startup_profiles_path": "step7_startup_profiles.json",
        "provider": "ionos",
        "top_k": 10
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "step8_final_report.json",
        "content": "{steps[0].payload}",
        "overwrite": true
      }
    }
  ]
}
```

## Schritt 9: PDF Abschlussbericht
Input: `step8_final_report.json`
Tool: `startup_matchup_step_9_pdf_report`
Output: `startup_matchup_report.pdf`

```json
{
  "steps": [
    {
      "tool": "startup_matchup_step_9_pdf_report",
      "args": {
        "final_report_path": "step8_final_report.json",
        "output_path": "startup_matchup_report.pdf",
        "title": "Startup Matchup Report"
      }
    }
  ]
}
```
