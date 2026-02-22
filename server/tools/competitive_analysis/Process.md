# Competitive Analysis Process

## Überblick
Die Pipeline läuft in 10 Schritten. Für Tests wird jeder Schritt als 2-Step-Aufruf ausgeführt:
1. Analyse-Tool
2. `write_file` mit gesamtem Payload (`{steps[0].payload}`)

Optional: Nach jedem Schritt kann `competitive_quality_gate` ausgeführt werden (Modus `validate` oder `validate_and_repair`).

```json
{
  "steps": [
    {
      "tool": "competitive_quality_gate",
      "args": {
        "artifact_path": "product_profile.json",
        "step": 2,
        "mode": "validate_and_repair",
        "provider": "openai",
        "max_context_chars": 18000
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "product_profile_qg.json",
        "content": "{steps[0].payload}",
        "overwrite": true
      }
    }
  ]
}
```

## Generisches Quality Gate (alle Schritte 1-10)
Tool: `competitive_quality_gate`  
Ziel: Schritt-Artefakte step-spezifisch per LLM prüfen und bei Bedarf (Modus `validate_and_repair`) bereinigen/reparieren.

Schritt-zu-Artefakt:
1. `parsed_doc.json`
2. `product_profile.json` (oder `product_profile_qg.json`)
3. `analysis_plan.json`
4. `competitor_list.json`
5. `competitor_profiles.json`
6. `feature_matrix_gap.json`
7. `strategic_analysis.json`
8. `final_report.json`
9. `review_status.json`
10. `pdf_publish_result.json`

Nur prüfen (`mode=validate`):
```json
{
  "steps": [
    {
      "tool": "competitive_quality_gate",
      "args": {
        "artifact_path": "feature_matrix_gap.json",
        "step": 6,
        "mode": "validate",
        "provider": "openai",
        "max_context_chars": 18000
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "feature_matrix_gap_qg.json",
        "content": "{steps[0].payload}",
        "overwrite": true
      }
    }
  ]
}
```

Prüfen + reparieren (`mode=validate_and_repair`):
```json
{
  "steps": [
    {
      "tool": "competitive_quality_gate",
      "args": {
        "artifact_path": "final_report.json",
        "step": 8,
        "mode": "validate_and_repair",
        "provider": "openai",
        "max_context_chars": 18000
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "final_report_qg.json",
        "content": "{steps[0].payload}",
        "overwrite": true
      }
    }
  ]
}
```

## Tool1: Dokumentenimport & Parsing
Input: Produktdatenblatt (PDF/DOCX/HTML/Text)
Prozess: Text- und Tabellenextraktion, OCR falls nötig, Strukturierung in Abschnitte
Output: `parsed_doc.json`

```json
{
  "steps": [
    {
      "tool": "competitive_parse_document",
      "args": {
        "path": "/competitor_analysis/Dreame_X50_Ultra_Complete_v2.pdf",
        "max_chars": 50000
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "parsed_doc.json",
        "content": "{steps[0].payload}",
        "overwrite": true
      }
    }
  ]
}
```

## Tool2: Feature- & Claim-Extraktion
Input: `parsed_doc.json`
Prozess: Kategorie bestimmen, technische Merkmale normalisieren, Claims extrahieren, Zielsegmente ableiten
Output: `product_profile.json`

```json
{
  "steps": [
    {
      "tool": "competitive_extract_product_profile",
      "args": {
        "parsed_doc_path": "parsed_doc.json",
        "provider": "ionos",
        "max_context_chars": 18000
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "product_profile.json",
        "content": "{steps[0].payload}",
        "overwrite": true
      }
    }
  ]
}
```

## Tool2a: Quality Gate für Feature- & Claim-Extraktion
Input: `product_profile.json`
Prozess: JSON evaluieren, unsinnige Feature-Objekte filtern/reparieren, gleiche `product_profile`-Struktur beibehalten
Output: `product_profile_qg.json`

```json
{
  "steps": [
    {
      "tool": "competitive_extract_feature_claim_profile_quality_gate",
      "args": {
        "product_profile_path": "product_profile.json",
        "provider": "openai",
        "max_context_chars": 18000,
        "remove_nonsensical_features": true,
        "repair_feature_names": true
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "product_profile_qg.json",
        "content": "{steps[0].payload}",
        "overwrite": true
      }
    }
  ]
}
```

Hinweis: In den Folgeschritten dann `product_profile_qg.json` statt `product_profile.json` verwenden.

## Tool3: Adaptiver Analyseplan (Strategie-Generator)
Input: `product_profile_qg.json` (oder `product_profile.json` ohne Quality Gate)
Prozess: Vergleichsdimensionen definieren, Suchqueries generieren, Mindestabdeckung festlegen, Feature-Schema erweitern
Output: `analysis_plan.json`

```json
{
  "steps": [
    {
      "tool": "competitive_generate_analysis_plan",
      "args": {
        "product_profile_path": "product_profile_qg.json",
        "provider": "ionos",
        "max_context_chars": 14000
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "analysis_plan.json",
        "content": "{steps[0].payload}",
        "overwrite": true
      }
    }
  ]
}
```

## Tool4: Wettbewerberidentifikation
Input: `analysis_plan.json` + `product_profile_qg.json` (oder `product_profile.json`)
Prozess: Websuche, semantische Ähnlichkeitsanalyse, Deduplizierung, Relevanzranking
Output: `competitor_list.json`

```json
{
  "steps": [
    {
      "tool": "competitive_identify_competitors",
      "args": {
        "analysis_plan_path": "analysis_plan.json",
        "product_profile_path": "product_profile_qg.json",
        "provider": "openai",
        "max_queries": 8,
        "per_query_results": 6,
        "shortlist_size": 12
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "competitor_list.json",
        "content": "{steps[0].payload}",
        "overwrite": true
      }
    }
  ]
}
```

## Tool4b: Wettbewerberliste-Quality-Gate (optional, empfohlen)
Input: `competitor_list.json` (+ optional `product_profile_qg.json`)
Prozess: Entfernt False-Positives aus Tool4 (z. B. generische Vergleichs-/Katalogseiten, schwache `unknown`-Kandidaten, Herstellerknoten ohne Modellsignal), dedupliziert Name+Domain.
Output: `competitor_list_qg.json`

```json
{
  "steps": [
    {
      "tool": "competitor_identification_quality_gate",
      "args": {
        "competitor_list_path": "competitor_list.json",
        "product_profile_path": "product_profile_qg.json",
        "provider": "perplexity",
        "min_relevance_score": 0.06,
        "drop_generic_listing_pages": true,
        "drop_weak_unknown_candidates": true,
        "drop_manufacturer_nodes_without_model_signal": true,
        "dedupe_by_name_and_domain": true,
        "enable_llm_snippet_validation": true,
        "llm_min_keep_confidence": 0.55,
        "max_llm_checks": 20
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "competitor_list_qg.json",
        "content": "{steps[0].payload}",
        "overwrite": true
      }
    }
  ]
}
```

## Tool5: Wettbewerberdaten-Extraktion & Normalisierung
Input: `competitor_list_qg.json` (oder `competitor_list.json`, falls Tool4b nicht genutzt wird)
Prozess: Crawling, Feature-Mapping, Preis-/Leistungsdaten erfassen, Evidenz dokumentieren
Output: `competitor_profiles.json`
- `source_registry_path` ist optional. Nur setzen, wenn die Datei bereits existiert.
- Standardlauf: `offset: 0`, `limit: 10`.

```json
{
  "steps": [
    {
      "tool": "competitive_extract_competitor_profiles",
      "args": {
        "competitor_list_path": "competitor_list.json",
        "registry_first": true,
        "min_active_sources_for_search": 2,
        "provider": "openai",
        "offset": 0,
        "limit": 10,
        "max_pages_per_competitor": 3
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "competitor_profiles.json",
        "content": "{steps[0].payload}",
        "overwrite": true
      }
    }
  ]
}
```

Hinweis: `write_file` nur ausführen, wenn `steps[0].ok == true`, sonst wird `competitor_profiles.json` mit leerem/fehlerhaftem Inhalt überschrieben.

### Tool5a: Feste Quellenliste verifizieren/aufbauen
- Ziel: pro Wettbewerber stabile Quellen (`primary` + `fallback`) prüfen und als Registry speichern.
- Output: `competitor_source_registry.json`

```json
{
  "steps": [
    {
      "tool": "competitive_verify_competitor_source_registry",
      "args": {
        "competitor_list_path": "competitor_list.json",
        "source_registry_path": "competitor_source_registry.json",
        "max_urls_per_competitor": 6,
        "timeout_seconds": 25,
        "include_fallbacks": true
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "competitor_source_registry.json",
        "content": "{steps[0].payload}",
        "overwrite": true
      }
    }
  ]
}
```

### Tool5 Batch-Modus (empfohlen bei langen Läufen)
- Ziel: Wettbewerber nacheinander in kleinen Batches verarbeiten.
- Terminal-Lebenszeichen: Das Tool schreibt pro Wettbewerber Fortschritt (`processing X/Y: <name>`).
- Empfehlung: `shortlist_size` in Tool4 auf `8` setzen.

Beispiel Batch 1 (`offset=0`, `limit=2`):
```json
{
  "steps": [
    {
      "tool": "competitive_extract_competitor_profiles",
      "args": {
        "competitor_list_path": "competitor_list.json",
        "provider": "openai",
        "offset": 0,
        "limit": 2,
        "max_pages_per_competitor": 3,
        "verbose_progress": true
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "competitor_profiles.json",
        "content": "{steps[0].payload}",
        "overwrite": true
      }
    }
  ]
}
```

Merge aller Teile:
```json
{
  "steps": [
    {
      "tool": "competitive_merge_competitor_profiles",
      "args": {
        "provider": "openai",
        "part_paths": [
          "competitor_profiles.json",
          "competitor_profiles_part_02.json",
          "competitor_profiles_part_03.json",
          "competitor_profiles_part_04.json"
        ]
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "competitor_profiles.json",
        "content": "{steps[0].payload}",
        "overwrite": true
      }
    }
  ]
}
```

### Tool5b: Quality Gate für Competitor-Profile
Input: `competitor_profiles.json`
Prozess: Profil-Qualität prüfen, verrauschte `mapped_features` entfernen, nicht-offizielle URLs markieren (ohne `profile.url` zu überschreiben), Preise gegen Modell/Variante verifizieren und bei Bedarf korrigieren.
Output: `competitor_profiles_qg.json`

```json
{
  "steps": [
    {
      "tool": "competitor_profile_extraction_quality_gate",
      "args": {
        "competitor_profiles_path": "competitor_profiles.json",
        "provider": "openai",
        "max_context_chars": 18000,
        "verify_independent_urls": true,
        "cross_validate_features": true,
        "enrich_seller_prices": true,
        "verify_existing_prices": true,
        "remove_noisy_mapped_features": true,
        "verbose_progress": true
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "competitor_profiles_qg.json",
        "content": "{steps[0].payload}",
        "overwrite": true
      }
    }
  ]
}
```

Hinweis: Das Tool gibt wie die anderen Tools ein Payload zurück; `write_file` sollte weiterhin `content: "{steps[0].payload}"` verwenden.

## Tool6: Feature-Matrix & Gap-Analyse
Input: `product_profile_qg.json` (oder `product_profile.json`) + `competitor_profiles_qg.json` (oder `competitor_profiles.json`)
Prozess: Vergleichsmatrix erstellen, Marktstandards identifizieren, Differenzierungsmerkmale und Lücken berechnen
Output: `comparison_matrix.json`, `gaps_and_usps.json`

```json
{
  "steps": [
    {
      "tool": "competitive_feature_matrix_gap_analysis",
      "args": {
        "product_profile_path": "product_profile.json",
        "competitor_profiles_path": "competitor_profiles.json",
        "provider": "openai"
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "feature_matrix_gap.json",
        "content": "{steps[0].payload}",
        "overwrite": true
      }
    }
  ]
}
```

### Tool6b: Feature-Matrix-Gap Quality Gate (Backfill fehlender Wettbewerbs-Features)
Input: `feature_matrix_gap.json`
Prozess: Für `present=false`-Features pro Wettbewerber gezielte Nachrecherche (`<competitor> <feature>`) via Websuche + LLM-Extraktion; bei Treffer wird der Feature-Cell ergänzt und mit Source-Vermerk markiert.
Output: `feature_matrix_gap_qg.json`

```json
{
  "steps": [
    {
      "tool": "competitive_feature_matrix_gap_analysis_quality_gate",
      "args": {
        "feature_matrix_gap_path": "feature_matrix_gap.json",
        "provider": "perplexity",
        "max_missing_features_per_competitor": 30,
        "max_urls_per_feature": 3,
        "max_llm_calls": 400,
        "min_confidence": 0.55,
        "verbose_progress": true
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "feature_matrix_gap_qg.json",
        "content": "{steps[0].payload}",
        "overwrite": true
      }
    }
  ]
}
```

## Tool7: Strategische Analyse (SWOT & Positionierung)
Input: `gaps_and_usps.json` + `comparison_matrix.json`
Prozess: Interne vs. externe Faktoren ableiten, SWOT strukturieren, Positionierungskoordinaten berechnen
Output: `swot.json`, `positioning.json`

```json
{
  "steps": [
    {
      "tool": "competitive_strategic_analysis",
      "args": {
        "gaps_and_usps_path": "feature_matrix_gap.json",
        "evidences": {
          "comparison_matrix_path": "feature_matrix_gap.json"
        },
        "provider": "openai"
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "strategic_analysis.json",
        "content": "{steps[0].payload}",
        "overwrite": true
      }
    }
  ]
}
```

## Tool8: Finaler Analysebericht (Strukturierte Management-Version)
Input: Alle validierten Artefakte aus Tool2-Tool7
Prozess: Inhalte verdichten, Kernaussagen priorisieren, Executive Summary erzeugen, strategische Empfehlungen formulieren
Output: `final_report.json`

```json
{
  "steps": [
    {
      "tool": "competitive_generate_final_report",
      "args": {
        "provider": "ionos",
        "artifact_paths": {
          "product_profile": "product_profile.json",
          "analysis_plan": "analysis_plan.json",
          "competitor_list": "competitor_list.json",
          "competitor_profiles": "competitor_profiles.json",
          "comparison_matrix": "feature_matrix_gap.json",
          "gaps_and_usps": "feature_matrix_gap.json",
          "strategic_analysis": "strategic_analysis.json"
        },
        "max_chars_per_artifact": 10000
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "final_report.json",
        "content": "{steps[0].payload}",
        "overwrite": true
      }
    }
  ]
}
```

## Tool9: Review / Konsistenzprüfung (optional vor Publishing)
Input: `final_report.json`
Prozess: Strukturprüfung, Vollständigkeit, formale Konsistenz
Output: `review_status.json` (`approved` / `needs_refine`)

Hinweis: Ein dediziertes Tool für diesen Schritt ist aktuell noch nicht implementiert.

## Tool10: Professional PDF Publisher
Input: `final_report.json`
Prozess: Kapitelstruktur rendern, Layout anwenden, Tabellen und Diagramme einbetten, PDF erzeugen
Output: `competition_analysis_report.pdf`

```json
{
  "steps": [
    {
      "tool": "competitive_publish_pdf_report",
      "args": {
        "final_report_path": "final_report.json",
        "output_path": "competition_analysis_report.pdf",
        "logo_path": "logo.png",
        "include_render_log": true,
        "render_log_path": "render_log.json"
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "pdf_publish_result.json",
        "content": "{steps[0].payload}",
        "overwrite": true
      }
    }
  ]
}
```

## End-to-End Stepkette (1-10, eine Sequenz)
Die folgende Kette ist ein einzelner Request im `steps`-Format von Schritt 1 bis 10.  
Nach jedem Schritt wird per `write_file` persistiert.

Hinweis zu Schritt 9: Das Review-Tool ist optional/nicht dediziert implementiert; hier wird ein Review-Placeholder geschrieben.

```json
{
  "steps": [
    {
      "tool": "competitive_parse_document",
      "args": {
        "path": "DE DS SH5.0RT 6.0RT 8.0RT 10RT Datenblatt.pdf",
        "max_chars": 50000
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "parsed_doc.json",
        "content": "{steps[0].payload}",
        "overwrite": true
      }
    },
    {
      "tool": "competitive_extract_product_profile",
      "args": {
        "parsed_doc_path": "parsed_doc.json",
        "provider": "perplexity",
        "max_context_chars": 18000
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "product_profile.json",
        "content": "{steps[2].payload}",
        "overwrite": true
      }
    },
    {
      "tool": "competitive_extract_feature_claim_profile_quality_gate",
      "args": {
        "product_profile_path": "product_profile.json",
        "provider": "perplexity",
        "max_context_chars": 18000,
        "remove_nonsensical_features": true,
        "repair_feature_names": true
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "product_profile_qg.json",
        "content": "{steps[4].payload}",
        "overwrite": true
      }
    },
    {
      "tool": "competitive_generate_analysis_plan",
      "args": {
        "product_profile_path": "product_profile_qg.json",
        "provider": "perplexity",
        "max_context_chars": 14000
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "analysis_plan.json",
        "content": "{steps[6].payload}",
        "overwrite": true
      }
    },
    {
      "tool": "competitive_identify_competitors",
      "args": {
        "analysis_plan_path": "analysis_plan.json",
        "product_profile_path": "product_profile_qg.json",
        "provider": "perplexity",
        "max_queries": 20,
        "per_query_results": 10,
        "shortlist_size": 25
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "competitor_list.json",
        "content": "{steps[8].payload}",
        "overwrite": true
      }
    },
    {
      "tool": "competitor_identification_quality_gate",
      "args": {
        "competitor_list_path": "competitor_list.json",
        "product_profile_path": "product_profile_qg.json",
        "provider": "perplexity",
        "min_relevance_score": 0.06,
        "drop_generic_listing_pages": true,
        "drop_weak_unknown_candidates": true,
        "drop_manufacturer_nodes_without_model_signal": true,
        "dedupe_by_name_and_domain": true,
        "enable_llm_snippet_validation": true,
        "llm_min_keep_confidence": 0.55,
        "max_llm_checks": 20
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "competitor_list_qg.json",
        "content": "{steps[10].payload}",
        "overwrite": true
      }
    },
    {
      "tool": "competitive_extract_competitor_profiles",
      "args": {
        "competitor_list_path": "competitor_list_qg.json",
        "provider": "perplexity",
        "offset": 0,
        "limit": 20,
        "max_pages_per_competitor": 3,
        "verbose_progress": true
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "competitor_profiles.json",
        "content": "{steps[12].payload}",
        "overwrite": true
      }
    },
    {
      "tool": "competitor_profile_extraction_quality_gate",
      "args": {
        "competitor_profiles_path": "competitor_profiles.json",
        "provider": "perplexity",
        "max_context_chars": 18000,
        "verify_independent_urls": true,
        "cross_validate_features": true,
        "enrich_seller_prices": true,
        "verify_existing_prices": true,
        "remove_noisy_mapped_features": true,
        "verbose_progress": true
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "competitor_profiles_qg.json",
        "content": "{steps[14].payload}",
        "overwrite": true
      }
    },
    {
      "tool": "competitive_feature_matrix_gap_analysis",
      "args": {
        "product_profile_path": "product_profile_qg.json",
        "competitor_profiles_path": "competitor_profiles_qg.json",
        "provider": "perplexity"
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "feature_matrix_gap.json",
        "content": "{steps[16].payload}",
        "overwrite": true
      }
    },
    {
      "tool": "competitive_feature_matrix_gap_analysis_quality_gate",
      "args": {
        "feature_matrix_gap_path": "feature_matrix_gap.json",
        "provider": "perplexity",
        "max_missing_features_per_competitor": 30,
        "max_urls_per_feature": 3,
        "max_llm_calls": 400,
        "min_confidence": 0.55,
        "verbose_progress": true
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "feature_matrix_gap_qg.json",
        "content": "{steps[18].payload}",
        "overwrite": true
      }
    },
    {
      "tool": "competitive_strategic_analysis",
      "args": {
        "gaps_and_usps_path": "feature_matrix_gap_qg.json",
        "evidences": {
          "comparison_matrix_path": "feature_matrix_gap_qg.json"
        },
        "provider": "perplexity"
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "strategic_analysis.json",
        "content": "{steps[20].payload}",
        "overwrite": true
      }
    },
    {
      "tool": "competitive_generate_final_report",
      "args": {
        "provider": "perplexity",
        "artifact_paths": {
          "product_profile": "product_profile_qg.json",
          "analysis_plan": "analysis_plan.json",
          "competitor_list": "competitor_list_qg.json",
          "competitor_profiles": "competitor_profiles_qg.json",
          "comparison_matrix": "feature_matrix_gap_qg.json",
          "gaps_and_usps": "feature_matrix_gap_qg.json",
          "strategic_analysis": "strategic_analysis.json"
        },
        "max_chars_per_artifact": 10000
      }
    },
    {
      "tool": "write_file",
      "args": {
        "path": "final_report.json",
        "content": "{steps[22].payload}",
        "overwrite": true
      }
    },
    {
      "tool": "competitive_publish_pdf_report",
      "args": {
        "final_report_path": "final_report.json",
        "output_path": "competition_analysis_report.pdf",
        "logo_path": "logo.png",
        "include_render_log": true,
        "render_log_path": "render_log.json"
      }
    }
  ]
}
```
