# Best Practices fuer Workflow-Tools

Diese Sammlung kombiniert praktische Erfahrungen aus produktiven Multi-Step-Workflows mit konkreten Patterns fuer Search, Extraktion, Berechnung und Reporting.

## 1. Search und Retrieval

### BRAVE Answers-API

- Umfangreiche Recherche:
  - `research=true` und `stream=true` verwenden, wenn breite Kontextsammlung wichtig ist.
- Schnelle Antworten:
  - `research=false` und `stream=true` verwenden, wenn niedrige Latenz wichtiger als maximale Tiefe ist.
- Prompt-Stil:
  - Keine Meta-Systemrolle in der Suchanfrage verwenden.
  - Direkte Suchintention formulieren, z. B. `Suche nach Alben von Lady Gaga`.
  - Vermeide Formulierungen wie `Du bist ein Recherche-Assistent...` in Search-Requests.

### Websuche mit Perplexity und OpenAI

- Komplexe Recherche in mehrere Einzelqueries zerlegen.
- Queries sequenziell oder kontrolliert parallel laufen lassen.
- Ergebnisse am Ende zusammenfuehren und deduplizieren.
- Erst Recall maximieren, dann filtern/ranken.

### Staged Retrieval (allgemein)

- Phase 1: breit sammeln (Recall).
- Phase 2: Treffer verdichten und unpassende Kandidaten entfernen.
- Phase 3: nur fuer verbleibende Kandidaten teure Tiefenextraktion starten.

## 2. Prompting und Tool-Nutzung

- Prompt-Hygiene:
  - Kurz, konkret, query-orientiert.
  - Keine unnötige Rollenprosa in Suchtools.
- Tool-Rollen trennen:
  - Search-Tool fuer Auffinden.
  - LLM fuer Strukturierung und Interpretation.
  - Deterministische Logik fuer Berechnungen.
- Bei mehreren Providern:
  - Je Step klar definieren, welcher Provider welche Aufgabe uebernimmt.

## 3. Berechnungen und Scores

- Numerik nicht rein LLM-basiert:
  - Scores, Aggregationen, Rankings deterministisch berechnen.
- Formeln dokumentieren:
  - Berechnungslogik im Output oder Log transparent machen.
- LLM fuer Erklaerung nutzen:
  - LLM soll Zahlen interpretieren, nicht erfinden.
- Rechen-Checks:
  - Stichproben gegen Referenzformeln validieren.

## 4. Workflow-Architektur

- Einheitliche Artefaktnamen:
  - Nummeriertes Schema wie `step1_...`, `step2_...`.
- Single Source of Truth:
  - Zentrale Run-Config fuer Parameter und Pfade.
- Klare Step-Verkettung:
  - Jeder Step liest explizit den vorherigen Output.
- Keine stillen Fallbacks:
  - Bei fehlendem Input lieber frueh mit klarer Fehlermeldung stoppen.

## 5. Datenqualitaet und Semantik

- `missing_data` vs `absent` unterscheiden:
  - `missing_data`: nicht erfasst.
  - `absent`: explizit nicht vorhanden.
- Evidenzpflicht:
  - Kritische Claims nur mit nachvollziehbaren `evidence_refs`.
- Einheiten normalisieren:
  - Vergleichbare Metriken auf gemeinsame Einheiten bringen.
- Metrik-Features separat fuehren:
  - Physikalische Groessen als `metric_features` behandeln, nicht als reine Binary-Features.

## 6. Gap/USP-Logik

- USPs priorisieren:
  - Nach Seltenheit/Marktabdeckung ranken.
  - Auf sinnvolle Top-N begrenzen.
- Gaps semantisch korrekt:
  - Nicht jede fehlende Nennung automatisch als Produktluecke werten.
- Metrik-Gap/USP nur wertbasiert:
  - Nur bei numerisch vergleichbaren Werten und klarer Optimierungsrichtung.

## 7. Robustheit und Fehlerbehandlung

- Fail-fast zwischen Steps:
  - Downstream-Step nicht starten, wenn Upstream-Artefakt leer/ungueltig ist.
- Warnungen mit Ursachen:
  - Z. B. `empty_response`, `timeout`, `parse_error`, `provider_error`.
- Timeouts und Retries:
  - Pro Provider bewusst konfigurieren.
- Degradationsstrategie:
  - Wenn optionaler Sub-Step ausfaellt (z. B. Lokalisierung), Kernpipeline stabil weiterfuehren.

## 8. Reporting und Visualisierung

- Leere Sektionen verhindern:
  - Vor Report-Generierung pruefen, ob Pflichtartefakte gefuellt sind.
- Fehlende Werte explizit kennzeichnen:
  - In 2x2/Charts fehlende X- oder Y-Werte getrennt markieren, nicht als echte `0` interpretieren.
- Konsistente Cluster-Labels:
  - Einheitliche Regeln fuer Fallback-Klassifikation dokumentieren.

## 9. Versionierung und Zusammenarbeit

- Breaking Changes nur mit Versionssprung.
- Prozessdoku (`process_*.md`) und reale Step-Config immer synchron halten.
- Golden-Run-Tests:
  - Referenzfaelle mit erwarteten Kernfeldern und Kennzahlen pflegen.
- Vor Merge:
  - Alte Dateinamen/Importe entfernen.
  - Warnings/Logs pruefen.

## 10. Schnelle Checkliste fuer neue Workflows

1. Sind Input/Output-Dateien pro Step eindeutig benannt?
2. Gibt es eine zentrale Run-Config mit allen Parametern?
3. Sind Search, Extraktion, Berechnung und Interpretation sauber getrennt?
4. Sind Scores/Formeln deterministisch und dokumentiert?
5. Gibt es Fail-fast-Regeln zwischen den Steps?
6. Sind `missing_data` und `absent` semantisch getrennt?
7. Werden Metriken als eigene Kategorie behandelt?
8. Sind Report-Sektionen gegen leere Artefakte abgesichert?
9. Sind Warnings ursachenbasiert und nutzbar fuer Debugging?
10. Sind Version, Doku und Tests konsistent?

