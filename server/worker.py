"""
Queue + Worker
- Der Server soll im Terminal ausgeben, wenn der Prozess beendet ist. - Die Gui könnte während des Jobs beendet werden:

✅ 1. Server-Änderungen
1.1 Einführung eines Job-Systems (Queue + Worker)

Wir haben aus synchronen LLM-Operationen ein echtes Hintergrund-Job-System gemacht:

JobManager mit:

eigener Queue (queue.Queue)

Worker-Thread (_worker_loop)

persistenten Job-Dateien (server/jobs/<job_id>.json)

Jobs sind:

queued

running

finished

error

1.2 Server führt die OpenAI-Recherche im Hintergrund aus

Die Endpunkte /generate_csv und /complete_csv geben sofort JSON zurück:

{ "job_id": "...", "status": "queued" }


Der tatsächliche LLM-Prozess wird vom Worker ausgeführt, nicht mehr direkt im Request.

1.3 Ergebnisse werden pro User gespeichert

Alle CSV-Ergebnisse werden gespeichert unter:

server/data/<user_id>/generate/<job_id>.csv
server/data/<user_id>/complete/<job_id>.csv


Mehrere Nutzer haben strikt getrennte Datenbereiche.

1.4 Endpunkte für Status und Download

Neu:

GET /job_status/{job_id}
GET /download/{job_id}


Damit kann der Client jederzeit:

prüfen, ob der Job fertig ist

die fertige Datei abrufen

1.5 Terminal-Logging bei Job-Abschluss

Nach jedem Job steht im Terminal eine vollständige Logzeile:

[JOB <id>] beendet – Status=finished, Typ=generate, User=user1, Ergebnis=..., Fehler=None

✅ 2. Client-Änderungen (CLI)
2.1 Polling statt blockierendem Download

Die Clients (client/main.py, client/complete_csv.py) machen:

POST → Server erzeugt Job

Polling via GET /job_status/...

Wenn „finished“ → Download via GET /download/...

Damit blockiert der Client nicht mehr während der LLM-Verarbeitung beim Server.

✅ 3. GUI-Änderungen (PySide6)
3.1 Keine Blockierung der Oberfläche mehr

Die langen Operationen laufen jetzt in eigenen QThread-Workern:

NewSearchWorker

CompleteSearchWorker

Dadurch bleibt die GUI voll bedienbar.

3.2 Buttons während des eigenen Jobs deaktiviert

Nur der entsprechende Start-Button wird deaktiviert:

Recherche starten wird grau

Rest der GUI bleibt aktiv

Mehrere Jobs können nacheinander erzeugt werden

3.3 Gelöschte Threads überleben GUI-Neustart nicht → Lösung

Wir speichern die letzten Job-IDs dauerhaft in config.json:

last_new_job_id
last_complete_job_id

✅ 4. Neue GUI-Features
4.1 „Letzte Recherche laden“ – zwei neue Buttons

Sowohl im Tab Neue Suche als auch im Tab Suche vervollständigen:

Button: „Letzte Recherche laden“

Ruft anhand der gespeicherten Job-ID:

/job_status

/download

Funktioniert auch, wenn die GUI zwischendurch geschlossen wurde

✨ 5. Architekturprinzipien, die du auf andere Projekte übertragen kannst

Diese Punkte sind besonders wertvoll für deine anderen KI-Agent-Projekte:

5.1 Worker + Queue statt blockierende Requests

Pattern:

User sendet Request → Server erzeugt Job → unmittelbares OK

Worker verarbeitet die Anfrage im Hintergrund

Ergebnis asynchron abrufbar

Ideal, wenn:

LLM-Aufrufe lange dauern

mehrere Nutzer gleichzeitig arbeiten

der Server niemals blockieren darf

5.2 Persistente Job-Dateien statt Datenbank

Einfacher, universell einsetzbarer Mechanismus:

jobs/<job_id>.json

inkludiert Status, Fehler, user_id, result_path

sehr robust, kein externer Dienst notwendig

perfekt für andere kleine bis mittelgroße Projekte

5.3 Dateibasierte User-Sandbox

Statt alles in einen Ordner zu schreiben:

data/<user_id>/...


Kannst du überall verwenden (RAG, OCR, PDF, Embedding, etc.)

5.4 GUI: lange Aufgaben in QThread auslagern

Kernprinzip:

GUI bleibt responsiv, Logik läuft im Worker.


Ausgezeichnet für:

LLM-Aufrufe

große Dateiverarbeitung

OCR-Batch-Prozesse

RAG-Index-Building

5.5 Polling + Download statt sofortiges Ergebnis

Wenn das Ergebnis groß oder zeitintensiv ist:

/submit_task

/task_status/{id}

/download/{id}

Ein universelles API-Muster für:

Videorendering

PDF-Generierung

Web-Scraping

ML-Inference
"""