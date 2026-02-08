Was du jetzt tatsächlich hast (klar eingeordnet)
Architektur-Status

RAG: eigenständiger, produktionsfähiger Dienst (8005)

Agent-Server: Orchestrator + Tool-Layer (8012)

query_rag: echtes Tool → HTTP → RAG

Contract sauber: Agent kennt nur Tool-Signaturen, nicht RAG-Interna

Das ist exakt die richtige Trennung.

Sinnvolle nächste Schritte (in richtiger Reihenfolge)
1️⃣ Agent „sichtbar intelligent“ machen

Aktuell rufst du Tools explizit auf. Nächster Schritt:

Ein Endpoint wie:

POST /agent/ask
{
  "goal": "Finde alle Rechnungen zu Dachausbau und fasse sie zusammen"
}


Intern:

LLM erzeugt steps

Orchestrator führt sie aus

Ergebnis kommt zurück

👉 Das ist der Moment, wo es sich wie ein Agent anfühlt.

2️⃣ Tool-Output normalisieren

Jetzt sinnvoll:

query_rag → internes, einheitliches Hit-Schema

Agent arbeitet nur mit:

{ "source", "score", "text" }


So bleibt der Agent stabil, auch wenn du das RAG später änderst.

3️⃣ Weitere Phase-1-Tools ergänzen

Sehr naheliegend bei dir:

list_files

download_file

open_document

summarize_hits

Alles trivial, jetzt wo das Fundament steht.