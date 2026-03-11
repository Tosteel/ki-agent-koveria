import os
import unittest

from pydantic import BaseModel

from server.agent.orchestrator import Orchestrator
from server.agent.tool_registry import ToolContext, ToolRegistry


class RagArgs(BaseModel):
    query: str


class SearchArgs(BaseModel):
    user_prompt: str


class ComposeArgs(BaseModel):
    text: str
    goal: str = ""


class SendMailArgs(BaseModel):
    to: list[str]
    subject: str
    body: str


class AdaptiveOrchestratorTests(unittest.TestCase):
    def setUp(self):
        self._old_runtime = os.environ.get("KOVERIA_RUNTIME")
        os.environ["KOVERIA_RUNTIME"] = "langgraph"
        self.ctx = ToolContext(user_id="u", settings=object(), api_key="k", goal="Bitte recherchiere und sende die Infos.")

    def tearDown(self):
        if self._old_runtime is None:
            os.environ.pop("KOVERIA_RUNTIME", None)
        else:
            os.environ["KOVERIA_RUNTIME"] = self._old_runtime

    def _base_registry(self) -> ToolRegistry:
        reg = ToolRegistry()
        reg.register("llm_compose", lambda _ctx, args: {"text": str(args.get("text") or "")}, request_model=ComposeArgs)
        reg.register(
            "send_mail",
            lambda _ctx, args: {
                "sent": True,
                "to": list(args.get("to") or []),
                "subject": str(args.get("subject") or ""),
                "body": str(args.get("body") or ""),
            },
            request_model=SendMailArgs,
        )
        return reg

    def test_rag_down_uses_fallback_and_continues(self):
        reg = self._base_registry()

        def rag_down(_ctx, _args):
            raise RuntimeError("Connection refused")

        def websearch_ok(_ctx, _args):
            return {
                "rows": [{"title": "Friedrich Merz", "source": "https://example.com"}],
                "text": "Friedrich Merz ist ein deutscher Politiker mit langjähriger Bundestagserfahrung.",
            }

        reg.register("rag_knowledgebase", rag_down, request_model=RagArgs)
        reg.register("websearch_table", websearch_ok, request_model=SearchArgs)
        orch = Orchestrator(reg)

        steps = [
            {"tool": "rag_knowledgebase", "args": {"query": "Friedrich Merz"}},
            {"tool": "llm_compose", "args": {"text": "{last.text}"}},
            {"tool": "send_mail", "args": {"to": ["x@example.com"], "subject": "Info", "body": "{last.text}"}},
        ]
        out = orch.run_steps(self.ctx, steps)
        tools = [str(x.get("tool") or "") for x in out]
        self.assertIn("websearch_table", tools)
        self.assertEqual(tools[-1], "send_mail")
        self.assertTrue(bool(out[-1].get("ok")))
        self.assertEqual(str(out[0].get("status")), "transient_error")
        self.assertTrue(bool(out[0].get("handled")))

    def test_rag_empty_uses_fallback(self):
        reg = self._base_registry()
        reg.register("rag_knowledgebase", lambda _ctx, _args: {"hits": [], "text": ""}, request_model=RagArgs)
        reg.register(
            "websearch_table",
            lambda _ctx, _args: {"rows": [{"source": "https://example.com"}], "text": "Gefundene Inhalte."},
            request_model=SearchArgs,
        )
        orch = Orchestrator(reg)

        steps = [
            {"tool": "rag_knowledgebase", "args": {"query": "Friedrich Merz"}},
            {"tool": "llm_compose", "args": {"text": "{last.text}"}},
        ]
        out = orch.run_steps(self.ctx, steps)
        self.assertEqual(str(out[0].get("status")), "empty")
        self.assertTrue(bool(out[0].get("handled")))
        self.assertIn("websearch_table", [str(x.get("tool") or "") for x in out])

    def test_rag_low_quality_uses_fallback(self):
        reg = self._base_registry()
        reg.register(
            "rag_knowledgebase",
            lambda _ctx, _args: {"hits": [{"text": "kurz"}], "text": "kurz"},
            request_model=RagArgs,
        )
        reg.register(
            "websearch_table",
            lambda _ctx, _args: {"rows": [{"source": "https://example.com"}], "text": "Ausreichend langer Recherchetext " * 10},
            request_model=SearchArgs,
        )
        orch = Orchestrator(reg)

        steps = [{"tool": "rag_knowledgebase", "args": {"query": "Friedrich Merz"}}]
        out = orch.run_steps(self.ctx, steps)
        self.assertEqual(str(out[0].get("status")), "low_quality")
        self.assertTrue(bool(out[0].get("handled")))
        self.assertIn("websearch_table", [str(x.get("tool") or "") for x in out])

    def test_retry_then_success(self):
        reg = self._base_registry()
        counter = {"n": 0}

        def rag_flaky(_ctx, _args):
            counter["n"] += 1
            if counter["n"] == 1:
                raise RuntimeError("timeout while calling backend")
            return {"hits": [{"source": "doc"}], "text": "Stabiler Ergebnistext " * 10}

        reg.register("rag_knowledgebase", rag_flaky, request_model=RagArgs)
        orch = Orchestrator(reg)

        steps = [{"tool": "rag_knowledgebase", "args": {"query": "Friedrich Merz"}}]
        out = orch.run_steps(self.ctx, steps)
        self.assertGreaterEqual(counter["n"], 2)
        self.assertEqual(str(out[0].get("status")), "transient_error")
        self.assertTrue(bool(out[0].get("handled")))
        self.assertTrue(bool(out[-1].get("ok")))

    def test_mail_gate_blocks_without_research_evidence(self):
        reg = self._base_registry()
        orch = Orchestrator(reg)

        steps = [
            {
                "tool": "send_mail",
                "args": {
                    "to": ["x@example.com"],
                    "subject": "Info",
                    "body": "Dies ist ein langer Text, aber ohne belastbare Recherchebasis im Verlauf.",
                },
            }
        ]
        out = orch.run_steps(self.ctx, steps)
        self.assertEqual(str(out[0].get("status")), "permanent_error")
        self.assertIn("side_effect_gate_blocked", str(out[0].get("error") or ""))
        self.assertEqual(str(out[-1].get("tool")), "__replan__")
        self.assertEqual(str(out[-1].get("status")), "replan_required")

    def test_replan_marker_when_no_fallback_available(self):
        reg = self._base_registry()
        reg.register("rag_knowledgebase", lambda _ctx, _args: {"hits": [], "text": ""}, request_model=RagArgs)
        orch = Orchestrator(reg)

        steps = [{"tool": "rag_knowledgebase", "args": {"query": "Friedrich Merz"}}]
        out = orch.run_steps(self.ctx, steps)
        self.assertEqual(str(out[-1].get("tool")), "__replan__")
        self.assertEqual(str(out[-1].get("status")), "replan_required")

    def test_capability_based_fallback_without_explicit_policy_entry(self):
        class ViewArgs(BaseModel):
            url: str
            query: str

        reg = self._base_registry()
        reg.register(
            "view_website",
            lambda _ctx, _args: {"url": "http://example.com", "query": "x", "matches": [], "text": ""},
            request_model=ViewArgs,
        )
        reg.register(
            "browse_website",
            lambda _ctx, _args: {"url": "http://example.com", "query": "x", "matches": [{"href": "/a"}], "text": "Treffer"},
            request_model=ViewArgs,
        )
        orch = Orchestrator(reg)

        steps = [{"tool": "view_website", "args": {"url": "http://example.com", "query": "x"}}]
        out = orch.run_steps(self.ctx, steps)
        tools = [str(x.get("tool") or "") for x in out]
        self.assertIn("browse_website", tools)
        self.assertTrue(bool(out[-1].get("ok")))

    def test_promotes_unresolved_steps_result_refs_only_after_fallback(self):
        class WebArgs(BaseModel):
            user_prompt: str

        class LangArgs(BaseModel):
            query: str

        class PdfArgs(BaseModel):
            output_path: str
            text: str

        reg = ToolRegistry()

        def web_fail(_ctx, _args):
            raise RuntimeError("Connection refused")

        def multi_fail(_ctx, _args):
            raise RuntimeError("Connection refused")

        def lang_ok(_ctx, _args):
            return {"query": "Olaf Scholz", "text": "Trefferliste Olaf Scholz"}

        reg.register("websearch_table", web_fail, request_model=WebArgs)
        reg.register("search_multitable", multi_fail, request_model=WebArgs)
        reg.register("langsearch", lang_ok, request_model=LangArgs)
        reg.register("llm_compose", lambda _ctx, args: {"text": str(args.get("text") or "")}, request_model=ComposeArgs)
        reg.register(
            "pdf_export",
            lambda _ctx, args: {"output_path": str(args.get("output_path") or ""), "text": str(args.get("text") or "")},
            request_model=PdfArgs,
        )
        orch = Orchestrator(reg)

        steps = [
            {"tool": "websearch_table", "args": {"user_prompt": "Olaf Scholz"}},
            {"tool": "llm_compose", "args": {"text": "{steps[0].result}"}},
            {"tool": "pdf_export", "args": {"output_path": "olaf_scholz.pdf", "text": "{steps[1].result}"}},
        ]
        out = orch.run_steps(self.ctx, steps)

        compose_entries = [x for x in out if str(x.get("tool") or "") == "llm_compose"]
        pdf_entries = [x for x in out if str(x.get("tool") or "") == "pdf_export"]
        self.assertTrue(compose_entries and bool(compose_entries[-1].get("ok")))
        self.assertTrue(pdf_entries and bool(pdf_entries[-1].get("ok")))
        self.assertEqual(str(compose_entries[-1]["result"]["text"]), "Trefferliste Olaf Scholz")
        self.assertEqual(str(pdf_entries[-1]["result"]["text"]), "Trefferliste Olaf Scholz")


if __name__ == "__main__":
    unittest.main()
