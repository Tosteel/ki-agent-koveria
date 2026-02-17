# Ideenspeicher

## Idee: Gedächtnis verbessern

Guter nächster Schritt. In eurem Stand gibt es schon die Basis: Die Chat-Historie liegt im Browser (`chats[]` in `client/gui_web.html`), wird aber noch nicht an den Bot übergeben.

Ein pragmatischer Weg in 3 Stufen:

1. Kurzzeit-Gedächtnis (sofort)
- Bei `sendMessage()` die letzten z. B. 8-12 Messages des aktiven Chats mitschicken.
- In `client/gui_web.py` daraus einen `context`-Text bauen und vor die neue Frage setzen.

2. Längere Sessions stabil halten
- Wenn Verlauf zu lang wird: ältere Nachrichten in eine kurze Zusammenfassung verdichten (Rolling Summary).
- Dann nur `summary + letzte N Nachrichten + neue Frage` senden.

3. Persistenz (echtes Gedächtnis über Neustarts)
- `chats` in Datei speichern (z. B. `client/chat_memory.json`) und beim Start laden.
- Optional pro Chat eine ID und Timestamp.

Warum das bei euch passt:
- Frontend hat bereits echte Chats/Verläufe in `client/gui_web.html`.
- Backend sendet aktuell nur `{"goal": req.message}` in `client/gui_web.py`, dort fehlt der Kontext-Teil.

Wenn du willst, setze ich dir jetzt direkt Stufe 1 sauber um (inkl. Limitierung, damit Requests nicht zu groß werden).

## Idee: Trigger für automatische Aufgaben (z. B. neue E-Mail)

Ziel: Aufgaben nicht nur manuell über „Neue Aufgabe“ starten, sondern auch automatisch durch Ereignisse.

Pragmatischer Weg in 3 Stufen:

1. Trigger-Quelle anbinden (einfach starten)
- Polling-Ansatz: alle X Minuten `fetch_unanswered_mails` prüfen.
- Später optional: IMAP-IDLE/Webhook statt Polling.

2. Trigger-Regeln auswerten
- Pro neuer Mail Regeln prüfen (Absender, Betreff-Keywords, Priorität).
- Wenn Regel passt: automatisch eine neue Aufgabe im Client/Taskspeicher anlegen (`source=trigger`).

3. Idempotenz und Kontrolle
- Verarbeitete Mail-UIDs speichern, damit keine Doppelaufgaben entstehen.
- Optional Freigabe-Workflow: Aufgabe wird vorgeschlagen, aber erst nach Bestätigung ausgeführt.

Architektur-Vorschlag:
- `server/automation/triggers.py` für Regel-Engine.
- `server/automation/state.json` für zuletzt gesehene/verarbeitete IDs.
- Separater Worker-Loop (nicht im HTTP-Request), z. B. alle 60 Sekunden.
- GUI-Badge „automatisch erstellt“ für Trigger-Aufgaben.

Beispielregel:
- Wenn neue unbeantwortete Mail von `*@kunde.de` und Betreff enthält „Lieferung“:
- Aufgabe erzeugen: „Mail prüfen und Lieferstatus antworten“.
- Optional direkt Tool-Kette vorbereiten: `fetch_unanswered_mails -> answer_mail`.

## Idee: Zielarchitektur für 1000+ Nutzer

Ziel: stabile Multi-User-Plattform mit horizontaler Skalierung.

1. API stateless machen
- Keine User-Zustände im Prozess halten.
- Mehrere API-Instanzen hinter Load-Balancer betreiben.

2. Persistenz zentralisieren
- Chats, Tasks, Trigger in eine zentrale Datenbank (z. B. PostgreSQL) speichern.
- Generierte Dateien (PDF/PPT) in Object Storage (S3-kompatibel) ablegen.

3. Hintergrundjobs einführen
- Längere Abläufe (Planner/Tools/Mail/Trigger) über Job-Queue ausführen.
- API antwortet schnell mit Job-ID; Ergebnis wird asynchron zurückgeliefert.

4. Trigger-Engine entkoppeln
- Trigger nicht im Webserver-Thread ausführen.
- Eigener Worker-Service für Scheduling und Trigger-Execution.

5. Caching und Rate Limits
- Redis für Cache, Session-nahe Daten und Throttling pro Nutzer/API-Key.
- Schutz gegen Lastspitzen und Missbrauch.

6. Skalierbarer Betrieb
- Containerisierung + Orchestrierung (z. B. Kubernetes/ECS).
- Horizontales Autoscaling für API und Worker.

7. Observability
- Zentrales Logging, Metriken, Tracing und Alerts.
- Monitoring auf Latenz, Fehlerquote und Queue-Lag.

8. Sicherheit und Tenant-Isolation
- Strikte Mandantentrennung in allen Datenmodellen/Queries.
- Secrets über Secret-Manager statt Klartext in `.env`.

Vorgeschlagene Umsetzung in Phasen:
- Phase 1: Datenmodell + DB-Migration für Chats/Tasks/Trigger.
- Phase 2: Job-Queue + Worker für asynchrone Tool-Ausführung.
- Phase 3: Load-Balancer + mehrere API-Instanzen + Monitoring/Autoscaling.
