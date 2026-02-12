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


class OrchestratorPlaceholderTests(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()

        def query_rag(_ctx, _args):
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

        self.registry.register("query_rag", query_rag, request_model=QueryArgs)
        self.registry.register("pdf_export", pdf_export, request_model=PdfArgs)

        self.orch = Orchestrator(self.registry)
        self.ctx = ToolContext(user_id="u", settings=object(), api_key="k")

    def test_resolves_placeholder_in_following_step(self):
        steps = [
            {"tool": "query_rag", "args": {"query": "rechnung"}},
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
            {"tool": "query_rag", "args": {"query": "rechnung"}},
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


if __name__ == "__main__":
    unittest.main()
