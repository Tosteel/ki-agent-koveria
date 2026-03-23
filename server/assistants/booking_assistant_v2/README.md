# Booking Assistant v2

## Ziel
`booking_assistant_v2` ist ein neu aufgesetzter Booking-Flow mit klarerem State-Modell:
- Thread-basierte Cases (`thread_cases`) statt verstreuter Fakten
- `required_fields` je Thread explizit im State
- Angebots-Handshake: erst Angebot senden, dann LLM-basiert Angebotsannahme prüfen, dann Kalender buchen
- Run-Lock + Run-History + Activity-Log für Cron-Betrieb

## Kern-Endpoints
- `POST /assistants/booking-assistant-v2/run-once`
- `GET /assistants/booking-assistant-v2/reviews`
- `GET /assistants/booking-assistant-v2/status-meeting`
- `GET /assistants/booking-assistant-v2/pending/next`
- `POST /assistants/booking-assistant-v2/pending/{review_id}/apply`

## Agent/Ask (Assistenten-Dialog)
Für Dialogsteuerung:
- `assistant_id: booking-assistant-v2`
- erlaubte Tools stehen in `server/assistants/booking_assistant_v2/policies.py`

## State-Struktur (Auszug)
- `processed_mail_ids`
- `thread_cases[]`
  - `thread_id`
  - `status` (`gathering|booked|declined|human_review`)
  - `required_field_names`
  - `required_fields` (Werte aus Extraktion)
  - `missing_required_fields`
  - `facts`
  - `offer` (signature, quote_text, status)
  - `event` (event_id)
  - `history[]`
- `reviews[]`
- `activity_log[]`
- `run_history[]`
- `run_lock`

## Ablauf
1. Profil laden/bootstrappen
2. Unbeantwortete Mails holen
3. Intent-Klassifikation
4. Bei Booking:
   - Fakten aus letzter Mail + Thread extrahieren und mit Thread-Case mergen
   - `required_fields` prüfen
   - Kalender-Precheck sobald Datum bekannt
   - Distanz + Preis berechnen
   - Angebot senden (falls noch nicht bestätigt)
   - Angebotsannahme per LLM prüfen
   - erst danach final buchen (`calendar_create_event`)
5. Policy + Instruction-Check
6. Auto-Send oder Human-Review

## Beispiel Request (run-once)
```json
{
  "provider": "ionos",
  "mailbox": "INBOX",
  "limit": 2,
  "assistant_profile_name": "dj_booking_default",
  "assistant_codename": "DJ Booking Bot v2",
  "profile_bootstrap": true,
  "profile_instructions_add": [
    "Bitte keine Termine nach 16 Uhr ohne explizite Freigabe.",
    "Öffnungszeiten-Anfragen nie automatisch beantworten.",
    "Keine Montage.",
    "Niemals länger als 8 Stunden am Stück."
  ],
  "profile_rules_patch": {
    "booking": {
      "base_address": "Pforzheim, Deutschland",
      "weekend_only": true,
      "max_duration_hours": 8,
      "max_distance_km": 200,
      "overnight_distance_km": 60,
      "overnight_after_hour": 22
    },
    "pricing": {
      "hourly_rate_eur": 120,
      "travel_per_km_eur": 0.7,
      "overnight_flat_eur": 120,
      "setup_flat_eur": 80,
      "teardown_flat_eur": 60,
      "travel_round_trip": true
    },
    "required_fields": [
      "event_date",
      "start_time",
      "duration_hours",
      "location",
      "occasion",
      "client_name",
      "price_confirmed"
    ]
  },
  "auto_send_threshold": 0.82,
  "web_sources": [],
  "web_whitelist_domains": [],
  "rag_top_k": 3,
  "max_context_chars": 12000,
  "include_thread": true,
  "strict_policy": true,
  "trace_steps": true
}
```
