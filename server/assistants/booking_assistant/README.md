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
- `POST /assistants/booking-assistant/run-once`
- `GET /assistants/booking-assistant/reviews`
- `POST /assistants/booking-assistant/reviews/{review_id}/approve`
- `POST /assistants/booking-assistant/reviews/{review_id}/reject`

## Profil-Setup im Dialog
Über `run-once` kann der Operator das Profil initialisieren und laufend erweitern:
- `assistant_profile_name`
- `assistant_codename`
- `profile_bootstrap`
- `profile_instructions_add`
- `profile_rules_patch`

## Kern-Tools
- Booking: `booking_extract_facts`, `booking_validate_completeness`, `booking_decision_engine`
- Pricing/Distance: `pricing_compute_quote`, `distance_check`
- Calendar: `calendar_check_availability`, `calender_hold_event`
- Mail: `mail_compose_clarification`, `gmail_answer_mail`

## Entscheidungslogik
- `need_clarification`: fehlende Pflichtangaben oder Preis nicht bestätigt
- `auto_decline`: harte Regel verletzt (z. B. Distanz > Limit, kein Wochenende)
- `human_review`: Grenzfall/Regelüberschreitung (z. B. Dauer > max)
- `auto_accept`: Regeln erfüllt, Kalender-Hold möglich

## Booking-Loop
- Erst Event-Details vollständig machen.
- Dann Preis berechnen und explizit zur Bestätigung senden.
- Erst nach Preisbestätigung wird reserviert (`calender_hold_event`) und bestätigt.
