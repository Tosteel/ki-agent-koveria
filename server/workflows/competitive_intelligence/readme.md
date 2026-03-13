# Competitive Intelligence Workflow

Diese Workflow-Struktur dient zur systematischen Wettbewerbsanalyse von der Trendrecherche bis zum finalen PDF-Report.

## Ziel

Ein reproduzierbarer, debugbarer und versionierbarer End-to-End-Prozess fuer Competitive Intelligence.

## Enthaltene Dateien

- `metadata.json`: Kurzbeschreibung von Zweck, Eingaben und Ausgaben.
- `process.md`: Schrittfolge mit Inputs, Outputs und Artefaktkonvention.

## Tool-Reihenfolge

1. `step1_1_market_trends_raw`
2. `step1_2_market_trends_structured`
3. `step1_3_market_trends_summary`
4. `step2_2_competitor_profile_raw`
5. `step2_3_competitor_profile_structured`
6. `step2_4_competitor_trends`
7. `step3_1_matrix`
8. `step4_1_insights`
9. `step5_1_recommendations`
10. `step6_1_final_report`
11. `step7_1_pdf_export`
12. `competitor_matrix`
13. `insights`
14. `recommendations`
15. `report_generator`
16. `pdf_export`

## Artefakt-Konvention (empfohlen)

- `step1.1_market_trends_raw.json`
- `step1.2_market_trends_structured.json`
- `step1.3_market_trends_summary.json`
- `step2.2_competitor_profile_raw.json`
- `step2.3_competitor_profile_structured.json`
- `step2.4_competitor_trends.json`
- `step3.1_matrix.json`
- `step4.1_insights.json`
- `step5.1_recommendations.json`
- `step6.1_final_report.json`
- `step7.1_report.pdf`
- `step3_competitor_matrix.json`
- `step4_insights.json`
- `step5_recommendations.json`
- `step6_report.md`
- `step7_report.pdf`
