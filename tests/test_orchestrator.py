import unittest

from pydantic import BaseModel

from server.agent.orchestrator import Orchestrator
from server.agent.tool_registry import ToolContext, ToolRegistry


class QueryArgs(BaseModel):
    query: str


class PdfArgs(BaseModel):
    output_path: str
    title: str
    text: str

class LlmSummaryArgs(BaseModel):
    text: str
    goal: str = ""


class OrchestratorPlaceholderTests(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()

        def rag_knowledgebase(_ctx, _args):
            return {
                "hits": [
                    {
                        "source": "rechnung_123.pdf",
                        "score": 0.99,
                        "text": "Preis netto: 199,00 EUR",
                    }
                ]
            }

        def pdf_export(_ctx, args):
            return {"output_path": args["output_path"], "text": args["text"]}

        def llm_text_summarize(_ctx, args):
            return {"text": args.get("text", ""), "goal": args.get("goal", "")}

        self.registry.register("rag_knowledgebase", rag_knowledgebase, request_model=QueryArgs)
        self.registry.register("pdf_export", pdf_export, request_model=PdfArgs)
        self.registry.register("llm_text_summarize", llm_text_summarize, request_model=LlmSummaryArgs)

        self.orch = Orchestrator(self.registry)
        self.ctx = ToolContext(user_id="u", settings=object(), api_key="k")

    def test_resolves_placeholder_in_following_step(self):
        steps = [
            {"tool": "rag_knowledgebase", "args": {"query": "rechnung"}},
            {
                "tool": "pdf_export",
                "args": {
                    "output_path": "result.pdf",
                    "title": "Rechnung",
                    "text": "Gefundener Preis: {{steps.1.result.hits.0.text}}",
                },
            },
        ]

        outputs = self.orch.run_steps(self.ctx, steps)

        self.assertTrue(outputs[0]["ok"])
        self.assertTrue(outputs[1]["ok"])
        self.assertEqual(
            outputs[1]["result"]["text"],
            "Gefundener Preis: Preis netto: 199,00 EUR",
        )

    def test_full_value_placeholder_returns_native_value(self):
        steps = [
            {"tool": "rag_knowledgebase", "args": {"query": "rechnung"}},
            {
                "tool": "pdf_export",
                "args": {
                    "output_path": "result.pdf",
                    "title": "Rechnung",
                    "text": "{{last.result.hits.0.source}}",
                },
            },
        ]

        outputs = self.orch.run_steps(self.ctx, steps)
        self.assertEqual(outputs[1]["result"]["text"], "rechnung_123.pdf")

    def test_legacy_steps_index_placeholder_is_resolved(self):
        steps = [
            {"tool": "rag_knowledgebase", "args": {"query": "rechnung"}},
            {
                "tool": "pdf_export",
                "args": {
                    "output_path": "result.pdf",
                    "title": "Rechnung",
                    "text": "{steps[0].hits[0].text}",
                },
            },
        ]

        outputs = self.orch.run_steps(self.ctx, steps)
        self.assertEqual(outputs[1]["result"]["text"], "Preis netto: 199,00 EUR")

    def test_dollar_placeholder_is_resolved(self):
        steps = [
            {"tool": "rag_knowledgebase", "args": {"query": "rechnung"}},
            {
                "tool": "pdf_export",
                "args": {
                    "output_path": "result.pdf",
                    "title": "Rechnung",
                    "text": "${steps[0].hits[0].text}",
                },
            },
        ]

        outputs = self.orch.run_steps(self.ctx, steps)
        self.assertEqual(outputs[1]["result"]["text"], "Preis netto: 199,00 EUR")

    def test_goal_is_injected_into_each_step_args(self):
        steps = [{"tool": "llm_text_summarize", "args": {"text": "input"}}]
        self.ctx.goal = "Bitte mit Quelle zusammenfassen"
        outputs = self.orch.run_steps(self.ctx, steps)
        self.assertEqual(outputs[0]["result"]["goal"], "Bitte mit Quelle zusammenfassen")


if __name__ == "__main__":
    unittest.main()
