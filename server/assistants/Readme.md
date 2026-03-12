1.
read_mail_thread
Grund: kompletter Thread statt nur Einzelmail.
2.
read_mail_attachments (+ OCR/PDF/Text-Extraktion)
Grund: Antworten hängen oft an Anhängen.
3.
search_web_whitelist
Grund: nur erlaubte Domains sauber und kontrolliert durchsuchen.
4.
score_reply (striktes JSON-Scoring als Tool)
Grund: Bewertung auslagern, reproduzierbar statt implizit im Servicecode.
5.
create_review_ticket / update_review_ticket
Grund: Human-in-the-loop sauber als Workflow-Objekt.
6.
customer_context_lookup (optional CRM/ERP)
Grund: kundenspezifische Antworten statt generisch.
7.
policy_check (Compliance/PII/Ton/Verbote)
Grund: Sicherheits- und Rechtsregeln vor Versand erzwingen.
Was ihr schon habt und nutzen könnt:
1.
fetch_unanswered_mails
2.
read_mail
3.
rag_knowledgebase
4.
view_website / websearch_table / langsearch
5.
llm_compose
6.
answer_mail / send_mail