# Booking Assistant

## Zweck
Der `booking_assistant` verarbeitet unbeantwortete Gmail-Anfragen für Buchungen und Terminabsprachen:
1. Mail lesen (`gmail_read_mail`, optional `gmail_read_mail_thread`)
2. Intent klassifizieren (`mail_classify`)
3. Profilregeln laden/bootstrappen (`assistent_profile_get/create/update`)
4. Bei Booking-Anfragen: Fakten extrahieren, Vollständigkeit prüfen, Preis/Distanz/Entscheidung
5. Bei Info-Anfragen: Kontext aus Profil + RAG/Web beziehen und Antwortentwurf erstellen
6. Policy + Score prüfen, dann auto-senden oder Human-Review

## Endpoints
Primär (aktiv empfohlen):
- `POST /assistants/booking-assistant/run-once`
- `GET /assistants/booking-assistant/reviews`
- `POST /assistants/booking-assistant/pending/{review_id}/apply`

Deprecated (weiterhin vorhanden, aber nicht mehr empfohlen):
- `GET /assistants/booking-assistant/status-meeting?since=...`
- `GET /assistants/booking-assistant/pending/next`
- `POST /assistants/booking-assistant/operator-chat`
- `POST /assistants/booking-assistant/reviews/{review_id}/approve`
- `POST /assistants/booking-assistant/reviews/{review_id}/reject`
- `POST /assistants/booking-assistant/reviews/{review_id}/counteroffer`

Hinweis:
- Für Dialogsteuerung bitte `POST /agent/ask` mit `assistant_id: booking-assistant` nutzen.

## Cron/Betrieb
- `run-once` ist der Worker-Entry.
- Der Lauf nutzt `state.run_lock` mit TTL, damit sich Cron-Laeufe nicht ueberlappen.
- Pro Lauf wird ein `run_history`-Eintrag geschrieben (`run_id`, start/end, counts, lock-Status).
- Pro Mail-Aktion wird ein `activity_log`-Eintrag geschrieben (`decision`, `booking_decision`, `review_id`, `event_id`, `reason`).

## Profil-Setup im Dialog
Über `run-once` kann der Operator das Profil initialisieren und laufend erweitern:
- `assistant_profile_name`
- `assistant_codename`
- `profile_bootstrap`
- `profile_instructions_add`
- `profile_rules_patch`

## Kern-Tools
- Booking: `booking_extract_facts`, `booking_validate_completeness`, `booking_decision_engine`, `booking_reply_score`, `booking_instruction_check`
- Pricing/Distance: `pricing_compute_quote`, `distance_check`
- Calendar: `calendar_check_availability`, `calender_hold_event`
- Mail: `mail_compose_clarification`, `gmail_answer_mail`

## Entscheidungslogik
- `need_clarification`: fehlende Pflichtangaben oder Preis nicht bestätigt
- `auto_decline`: harte Regel verletzt (z. B. Distanz > Limit, kein Wochenende)
- `human_review`: Grenzfall/Regelüberschreitung (z. B. Dauer > max)
- `auto_accept`: Regeln erfüllt, Kalender-Hold möglich

## Review-Vorlagen
Jeder Review-Eintrag enthält `action_templates` mit vordefinierten Texten:
- `approve`
- `reject`
- `counteroffer`

Die Endpunkte `/approve`, `/reject` und `/counteroffer` verwenden standardmäßig die jeweilige Vorlage, sofern kein `edited_body` übergeben wird.

## Booking-Loop
- Erst Event-Details vollständig machen.
- Dann Preis berechnen und explizit zur Bestätigung senden.
- Erst nach Preisbestätigung wird reserviert (`calender_hold_event`) und bestätigt.

## Ablaufdiagramm
```text
Booking Assistant (run-once)

[Start]
  |
  v
[Load/Bootstrap Profile]
  - assistent_profile_get
  - optional: create/update/get
  |
  v
[Load Rules]
  - booking_rules
  - pricing_rules
  - required_fields
  |
  v
[Fetch Unanswered Mails]
  |
  v
+-----------------------------+
| For each mail (limit N)     |
+-----------------------------+
  |
  v
[Already processed?] --yes--> [Skip]
  |
 no
  v
[Read Mail + Thread]
  - gmail_read_mail
  - gmail_read_mail_thread
  |
  v
[Classify Intent]
  |
  +--> newsletter --> [Mark skipped] --> (next mail)
  |
  +--> info
  |      |
  |      v
  |   [RAG/Web Context]
  |      |
  |      v
  |   [Compose info reply]
  |      |
  |      v
  |   [Score + Policy + Instruction Check]
  |      |
  |      +--> pass --> [gmail_answer_mail] -> [mark_processed]
  |      |
  |      +--> fail --> [Create review] -> [mark_processed]
  |
  +--> termin/angebot/beschwerde/eskalation
         |
         v
      [Extract Facts]
      - from latest mail
      - from thread
      - merge thread context memory
         |
         v
      [Offer Acceptance Gate (LLM)]
      - offer present in thread?
      - latest reply accepts offer?
         |
         v
      [Calendar Precheck]
         |
         v
      [Validate Completeness]
         |
         +--> incomplete --> [need_clarification draft]
         |
         +--> complete
               |
               v
            [Early Hard Rules]
            (weekend, duration, etc.)
               |
               +--> auto_decline --> [decline draft]
               |
               +--> human_review --> [internal review draft]
               |
               +--> continue
                     |
                     v
                  [Distance + Pricing]
                     |
                     v
                  [Decision Engine Phase 1]
                  require_price_confirmation=false
                     |
                     +--> auto_decline/human_review (as above)
                     |
                     +--> continue
                           |
                           +--> price_confirmed == false
                           |      -> [need_clarification: send quote + ask confirm]
                           |
                           +--> price_confirmed == true
                                  |
                                  v
                               [Decision Engine Phase 2]
                               require_price_confirmation=true
                                  |
                                  +--> auto_accept
                                  |      |
                                  |      v
                                  |   [Final calendar check]
                                  |      |
                                  |      +--> free -> create/update event
                                  |      |
                                  |      +--> busy -> need_clarification
                                  |
                                  +--> auto_decline / need_clarification / human_review
                                  |
                                  v
                               [Compose decision draft]
         |
         v
      [booking_reply_score]
         |
         v
      [policy_check]
         |
         v
      [booking_instruction_check (LLM vs profile instructions)]
         |
         v
      [Auto-send gate]
      - decision erlaubt?
      - score >= threshold + verdict=send (bei auto_accept/info)
      - policy allowed?
      - instruction allowed?
         |
         +--> yes --> [gmail_answer_mail] -> [mark_processed]
         |
         +--> no  --> [Create review + templates] -> [mark_processed]

(end loop)
  |
  v
[save_state + run summary]
```
