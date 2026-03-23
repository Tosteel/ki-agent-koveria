# Booking Assistant v3

## Zielprozess
1. Neue Inbox-Mail lesen und klassifizieren.
2. `info` -> automatische Antwort via Whitelist-Web + optional RAG.
3. `newsletter` -> skip.
4. `termin` -> Pflichtfelder extrahieren, Regeln prüfen:
   - `rejection` verletzt -> automatische Absage.
   - `human_review` getroffen -> Human-in-the-loop.
5. Kalender prüfen:
   - Kein kompletter Termin -> offen.
   - Termin frei -> als verfügbar merken.
   - Termin belegt -> automatische Absage (falls aktiviert).
6. Anfrage unvollständig -> fehlende Daten anfragen; falls Dauer bekannt + Slot frei optional Angebot mitsenden.
7. Bei vollständiger Anfrage:
   - Angebot senden (falls noch nicht gesendet)
   - Angebotsannahme per LLM erkennen
   - bei Annahme: Termin unter Vorbehalt bestätigen und `final_confirmation`-Review erzeugen.

## Wenn/Dann Flow
```text
Neue Mail
  -> Klassifikation
     -> newsletter: skip
     -> info: Web+RAG Kontext -> Info-Antwort
     -> termin:
         -> Pflichtfelder extrahieren + Thread-Case aktualisieren
         -> Rejection-Regeln verletzt?
            -> ja: Auto-Absage
            -> nein:
               -> Kalender-Precheck
               -> Slot belegt?
                  -> ja: Auto-Absage (wenn auto_decline_if_busy=true)
                  -> nein:
                     -> Slot frei + Termin vollständig?
                        -> ja: Kalender-Blocker (Hold) setzen
                     -> Human-Review-Regeln verletzt?
                        -> ja: Review(kind=rule_review, Optionen approve/offer/reject)
                        -> nein:
                           -> Pflichtfelder unvollständig?
                              -> ja: Missing-Fields-Mail (+ optional Angebot)
                              -> nein:
                                 -> Angebot schon gesendet?
                                    -> nein: Angebot senden
                                    -> ja:
                                       -> Angebot vom Kunden angenommen (LLM)?
                                          -> nein: erneut Angebotsbestätigung anfordern
                                          -> ja: unter Vorbehalt bestätigen
                                                + Review(kind=final_confirmation, Optionen final_confirmation/final_rejection)
```

## HITL Tools
- `booking_assistant_v3_status`
- `booking_assistant_v3_reviews`
- `booking_assistant_v3_pending_next`
- `booking_assistant_v3_pending_apply`

Hinweis:
- Schreibende Aktionen laufen zentral über `booking_assistant_v3_pending_apply`.
- Legacy-API-Routen unter `/reviews/{review_id}/...` sind nur noch deprecated Alias und delegieren intern auf `pending/apply`.

## Neue Profilbereiche
Unter `rules`:
- `rejection` (neu)
- `human_review` (neu)
- `calendar` (neu)
- plus wie gehabt: `instructions`, `offering`, `required_fields`, `pricing`, `booking`, `mail`
