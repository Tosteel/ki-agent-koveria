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
