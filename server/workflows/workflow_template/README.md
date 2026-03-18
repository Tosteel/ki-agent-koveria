# Workflow Blueprint (Allgemeingültig)

Dieser Ordner dient als Vorlage, um neue Workflow-Ketten schnell, konsistent und wartbar aufzubauen.

## Ziel

Einen beliebigen Mehrschritt-Workflow so definieren, dass er:
- reproduzierbar läuft,
- klar debuggbar ist,
- versionierbar bleibt,
- und leicht auf neue Use-Cases übertragen werden kann.

## Minimale Bausteine eines Workflows

1. `process_*.md`
- Menschlich lesbare Beschreibung der Step-Kette.
- Pro Schritt: Zweck, Input, Tool, Output, Beispielaufruf.

2. Run-Config (`runs/<name>.json`)
- Zentraler Ort für alle variablen Parameter und Artefaktpfade.
- Keine verstreuten Hardcodes in einzelnen Steps.

3. Artefakt-Konvention
- Eindeutige, nummerierte Outputs je Schritt.
- Beispiel: `step1_...`, `step2_...`, `step3_...`.

4. Validierung
- Jeder Schritt hat ein erwartetes Input/Output-Schema.
- Vor und nach Tool-Aufruf prüfen.

## Namenskonvention (empfohlen)

Nutze nummerierte Artefakte mit sprechendem Suffix:
- `step1_<artifact>.json`
- `step2_<artifact>.json`
- `stepN_<artifact>.json`

Für Nicht-JSON:
- `stepN_<artifact>.pdf`
- `stepN_<artifact>.csv`
- `stepN_<artifact>.md`

Regeln:
- Keine Leerzeichen in Dateinamen.
- Keine unversionierten Alias-Dateien im produktiven Run.
- Jeder Folge-Step liest exakt den definierten Vorgänger-Output.

## Generisches Run-Config-Beispiel

```json
{
  "workflow_name": "my_workflow_v1",
  "providers": {
    "llm": "openai",
    "search": "brave"
  },
  "limits": {
    "max_items": 10,
    "max_context_chars": 18000
  },
  "runtime": {
    "stream": true,
    "verbose_terminal": true
  },
  "artifacts": {
    "step1": "step1_input_prepared.json",
    "step2": "step2_enriched_data.json",
    "step3": "step3_analysis.json",
    "step4": "step4_report.json",
    "step5": "step5_report.pdf",
    "step5_log": "step5_render_log.json"
  }
}
```

## Schritt-Template (pro Step in `process_*.md`)

~~~md
## Schritt X: <Titel>
Input: `<datei_a.json>` (+ optional `<datei_b.json>`)
Tool: `<tool_name>`
Output: `<stepX_output.json>`

<kurzer Zweck/Regeln>

```json
{
  "steps": [
    {
      "tool": "<tool_name>",
      "args": {
        "<input_path_arg>": "<stepX-1_output.json>",
        "provider": "openai"
      }
    },
    {
      "tool": "file_write",
      "args": {
        "path": "<stepX_output.json>",
        "content": "{{steps[0].payload}}",
        "overwrite": true
      }
    }
  ]
}
```
~~~

## Qualitätsregeln (für alle Workflows)

1. Fail-fast
- Bei leerem oder ungültigem Kern-Input nächsten Schritt nicht starten.
- Fehler mit Ursache und betroffener Datei loggen.

2. Schema-Stabilität
- Breaking Changes nur mit Versionssprung (`v0_5` -> `v0_6`).
- Alte Versionen nur gezielt und zeitlich begrenzt weiterführen.

3. Import-Konsistenz
- Referenzen in `artifact_paths` müssen mit den tatsächlichen Outputs übereinstimmen.
- Keine Mischformen aus alten und neuen Dateinamen.

4. Beobachtbarkeit
- Pro Schritt Laufzeit, Warnungen, Inputquelle und Anzahl Objekte dokumentieren.

## Troubleshooting (allgemein)

1. Nachgelagerter Step erzeugt nur `null`/leer:
- Vorgänger-Output öffnen und Mindestfelder prüfen.
- Pfad-Argumente auf Tippfehler oder alte Dateinamen prüfen.

2. Ergebnisbericht enthält leere Sektionen:
- Prüfen, ob die Quell-Artefakte tatsächlich befüllt und korrekt importiert wurden.
- Prüfen, ob Mapping-Schlüssel in `artifact_paths` korrekt sind.

3. Unerwartete Darstellung/Charts:
- Prüfen, ob fehlende Werte sauber als "missing" markiert werden.
- Keine impliziten Defaultwerte verwenden, die als echte Messwerte interpretiert werden können.

## Team-Workflow (empfohlen)

1. Änderungen an Step-Inputs/Outputs immer in `process_*.md` und Run-Config nachziehen.
2. Pro Änderung mindestens einen Teil- oder End-to-End-Testlauf dokumentieren.
3. Vor Merge prüfen:
- Keine veralteten Dateireferenzen.
- Versions- und Toolnamen konsistent.
- Logs/Warnungen nachvollziehbar.

## Optional: Domänenspezifische Ableitung

Für konkrete Anwendungsfälle (z. B. Competitive Analysis, Reporting, ETL) kannst du auf Basis dieses Blueprints ein eigenes `process_<domain>_vX.md` anlegen.
Die Domänenregeln gehören in diese spezialisierte Datei, nicht in dieses allgemeine Blueprint-Dokument.
