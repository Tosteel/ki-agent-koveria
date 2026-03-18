# Mail Assistant

## Zweck
Der `mail_assistant` verarbeitet unbeantwortete E-Mails automatisch:
1. Mail lesen (inkl. optional Thread/Anhänge)
2. Intent klassifizieren (`mail_classify`)
3. Kontext aus RAG/Web holen
4. Antwort entwerfen (`llm_text_compose`)
5. Qualität bewerten (`customer_support_reply_score`)
6. Policy prüfen (`customer_support_policy_check`)
7. Auto-Send oder Human-Review (`customer_support_review_ticket_create`)

## API
- `POST /assistants/mail-assistant/run-once`
- `GET /assistants/mail-assistant/reviews`
- `POST /assistants/mail-assistant/reviews/{review_id}/approve`
- `POST /assistants/mail-assistant/reviews/{review_id}/reject`

## Wichtige Run-Parameter
Siehe `models.py` (`MailAssistantRunRequest`):
- `auto_send_threshold`: Mindestscore für Auto-Send
- `web_sources`: Start-URLs für Web-Recherche
- `web_whitelist_domains`: erlaubte Domains für `web_crawl_site_whitelist`
- `include_thread`: `mail_read_thread` aktiv
- `include_attachments`: `mail_read_attachments` aktiv
- `strict_policy`: strenger `customer_support_policy_check`
- `trace_steps`: Terminal-Step-Logs wie bei `agent/ask`

## Tool-Infos anpassen
### 1) Welche Tools der Assistant nutzt
In `service.py` über `_tool_call(...)` / `_safe_tool_call(...)`.
Hier kannst du Tool-Reihenfolge und Fallbacks ändern.

### 2) Tool-Parameter pro Schritt
Ebenfalls in `service.py`, z. B.:
- `rag_top_k` in `_retrieve_context(...)`
- `max_pages/max_matches` bei `web_crawl_site_whitelist`
- `max_chars` bei `llm_text_compose`
- `require_actionable` bei `customer_support_reply_score`

### 3) Intent-abhängiges Verhalten
In `service.py`:
- `_classify_mail_intent(...)`
- `_intent_policy(...)`

Dort steuerst du z. B.:
- wann immer Human-Review erzwungen wird
- welche Intents strengere Regeln haben
- welche Draft-Hinweise verwendet werden

### 4) Tool-Metadaten (Planner/Policy)
In den jeweiligen Tool-Ordnern `metadata.json`, z. B.:
- `server/tools/mail/metadata.json`
- `server/tools/customer_support/metadata.json`
- `server/tools/web/metadata.json`

Dort kannst du u. a. `capabilities`, `retry_policy`, `fallback`, `side_effect_level`, `quality_signals` pflegen.

## Typische Anpassungen
- Auto-Send konservativer machen:
  - `auto_send_threshold` erhöhen
  - `_intent_policy()` für `beschwerde/eskalation` auf immer manuell
- Mehr Recherchequalität:
  - `rag_top_k` erhöhen
  - zusätzliche `web_sources` setzen
  - `web_whitelist_domains` sauber pflegen (nur Hostnamen)
- Review-Prozess schärfen:
  - Gründe/Ticket-Metadaten in `customer_support_review_ticket_create` erweitern
