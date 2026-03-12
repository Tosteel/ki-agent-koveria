# ki-agent-koveria

KI-Agent mit FastAPI-Backend, Web-Client, Tool-Orchestrierung und Trigger-Engine.

## Ueberblick

Das Projekt besteht aus zwei Hauptteilen:

- `server/`: Agent-Backend (Planner, Tool-Ausfuehrung, Trigger, APIs)
- `client/`: Web-GUI inkl. Chat, Aufgaben, Trigger-Verwaltung

Kernfunktionen:

- Chat mit `askIonos` / `askOpenAI`
- Explizite Tool-Ausfuehrung via `/agent/run`
- Reine Planung via `/agent/plan`
- Aufgaben speichern, bearbeiten, rerun, als Trigger speichern
- Mail-Tools (senden, antworten, Inbox lesen)
- Trigger (aktuell `manually`, `time_schedule`)

## Projektstruktur

```text
server/
  main.py                  # FastAPI-App und Endpunkte
  auth.py                  # Bearer-Token -> user_id Mapping
  core/                    # Settings, Models, Logging
  agent/                   # Planner, Orchestrator, Policies, Registry
  services/                # LLM Integrationen + Prompt-Definitionen
  tools/                   # Tool-Module (je Tool eigener Ordner)
  triggers/                # Trigger-Module + Runtime + Store

client/
  gui_web.py               # FastAPI fuer Web-GUI (Proxy + Memory APIs)
  gui_web.html             # Frontend
  assets/                  # Logos/Avatare
  data/users/<user>/       # Chat-/Task-Memory je User (lokal im Client)

tests/
ideen.md
requirements.txt
```

## Voraussetzungen

- Python 3.11+
- Virtuelle Umgebung empfohlen
- Abhaengigkeiten aus `requirements.txt`
- Fuer `browse_website` (Playwright): Browser installieren

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m playwright install chromium
```

## Konfiguration

### 1) Server ENV (`server/.env`)

Minimal fuer IONOS:

```env
IONOS_API_BASE=https://openai.inference.de-txl.ionos.com/v1
IONOS_API_KEY=...
IONOS_MODEL=meta-llama/Llama-3.3-70B-Instruct
MAX_TOKENS=400
TEMPERATURE=0
TOP_P=0.1
```

Optional fuer Mail:

```env
SMTP_HOST=
SMTP_PORT=587
SMTP_USERNAME=
SMTP_PASSWORD=
SMTP_FROM=
SMTP_STARTTLS=true
SMTP_USE_SSL=false

IMAP_HOST=
IMAP_PORT=993
IMAP_USERNAME=
IMAP_PASSWORD=
IMAP_USE_SSL=true
```

Optional fuer Tools:

```env
EBAY_APP_ID=
RAG_BASE_URL=http://localhost:8005
SEARCH_BASE_URL=http://localhost:8002
# Runtime fuer Planner/Tool-Dispatch:
# langgraph (Default), langchain, legacy
KOVERIA_RUNTIME=langgraph
```

### 2) Auth-Token

Die API-Keys werden aktuell in `server/auth.py` gepflegt (`API_KEYS`).

## Starten

### Server

```bash
uvicorn server.main:app --host 0.0.0.0 --port 8012 --reload
```

### Web-GUI

```bash
uvicorn client.gui_web:app --host 0.0.0.0 --port 8013 --reload
```

GUI aufrufen:

- `http://localhost:8013/`

## Wichtige Server-Endpunkte

- `GET /health`
- `GET /user`
- `POST /agent/askIonos`
- `POST /agent/askOpenAI`
- `POST /agent/plan` (nur Planung)
- `POST /agent/run` (nur ausfuehren, keine Planung)
- `GET /tasks/memory`
- `POST /tasks/memory/sync`
- `GET /triggers/types`
- `GET /triggers`
- `POST /triggers`
- `PATCH /triggers/{trigger_id}`
- `DELETE /triggers/{trigger_id}`
- `POST /triggers/{trigger_id}/run-now`

Direkt-Tool-Endpunkte:

- `POST /tools/rag/query`
- `POST /tools/search/generate_json`
- `POST /tools/files/read`
- `POST /tools/files/write`
- `POST /tools/pdf/export`
- `POST /tools/ppt/export`
- `POST /tools/mail/send`

## API-Kontrakt: `/agent/plan`

`/agent/plan` plant nur Schritte und fuehrt keine Tools aus.

Request:

```json
{
  "goal": "bitte nach der tagesschau auch auf http://www.zdf.de suchen.",
  "additional_props": {
    "planned_steps": [
      "1. tool=view_website args={\"url\":\"https://www.tagesschau.de/\",\"query\":\"Friedrich Merz\"}",
      "2. tool=llm_compose args={\"text\":\"{steps[0].text}\"}"
    ]
  }
}
```

Response (Beispiel):

```json
{
  "ok": true,
  "goal": "...",
  "normalized_goal": "...",
  "status": "ready",
  "steps": [
    { "tool": "view_website", "args": { "url": "...", "query": "..." } },
    { "tool": "llm_compose", "args": { "text": "{steps[0].text}" } }
  ]
}
```

## Multi-User

- User wird ueber Bearer-Token bestimmt (`Authorization: Bearer <token>`)
- Daten liegen pro User unter `server/data/users/<user_id>/...`
- GUI speichert client-seitige Daten unter `client/data/users/<user_id>/...`

Hinweis:

- Entwicklungsbetrieb mit `--reload` ist fuer lokales Arbeiten gedacht.
- Fuer produktionsnahe Last lieber ohne `--reload` und mit mehreren Workern starten.

Beispiel:

```bash
uvicorn server.main:app --host 0.0.0.0 --port 8012 --workers 2
```

## Tools erweitern

Neue Tools sind modular aufgebaut:

1. Ordner anlegen: `server/tools/<toolname>/`
2. Dateien anlegen:
   - `<toolname>.py`
   - `models.py`
   - `registry.py`
3. `registry.py` muss eine `register(registry)` Funktion bereitstellen
4. Auto-Discovery laedt das Tool ueber `server/tools/loader.py`

Template:

- `server/tools/_template/`

## Trigger erweitern

Neue Trigger analog modular:

1. Ordner anlegen: `server/triggers/<triggername>/`
2. Dateien anlegen:
   - `<triggername>.py`
   - `models.py`
   - `registry.py`
3. `registry.py` mit `register(registry)`
4. Auto-Discovery ueber `server/triggers/loader.py`

Template:

- `server/triggers/_template/`

## Entwicklungs-Hinweise

- Planner- und Final-Ausgaben werden im Server-Terminal geloggt.
- GUI-Replan loggt in `client/gui_web.py` zusaetzlich `REPLAN RAW RESPONSE` und `REPLAN PARSED STEPS`.
- Trigger-Runs schreiben Logs nach `server/data/users/<user>/logs/trigger_runs.jsonl`.

## Sicherheit

- Keine echten Zugangsdaten in Git einchecken (`.env`, SMTP/IMAP, API Keys).
- `server/auth.py` mit statischen Tokens ist fuer Development ok, fuer Produktion auf DB/JWT umstellen.

## Lizenz

Aktuell keine separate Lizenzdatei im Repository hinterlegt.
