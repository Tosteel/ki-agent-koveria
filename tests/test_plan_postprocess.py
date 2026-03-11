import unittest

from server.main import _compact_tool_outputs, _inject_llm_summary_before_pdf, _outputs_for_final_answer


class PlanPostprocessTests(unittest.TestCase):
    def test_injects_llm_compose_before_pdf_when_goal_wants_summary(self):
        steps = [
            {"tool": "query_rag", "args": {"query": "kunden"}},
            {"tool": "pdf_export", "args": {"output_path": "kunden.pdf", "text": "${steps[0].text}"}},
        ]
        out = _inject_llm_summary_before_pdf(steps, "Fasse nur die Kunden zusammen")
        self.assertEqual(out[1]["tool"], "llm_compose")
        self.assertEqual(out[2]["tool"], "pdf_export")
        self.assertEqual(out[2]["args"]["text"], "{last.text}")

    def test_rewrites_existing_llm_summarize_step_to_llm_compose(self):
        steps = [
            {"tool": "query_rag", "args": {"query": "kunden"}},
            {"tool": "llm_summarize", "args": {"text": "{last.text}"}},
            {"tool": "pdf_export", "args": {"output_path": "kunden.pdf", "text": "{last.text}"}},
        ]
        out = _inject_llm_summary_before_pdf(steps, "Fasse nur die Kunden zusammen")
        self.assertEqual(out[1]["tool"], "llm_compose")

    def test_compact_outputs_keep_only_payload_for_success(self):
        full = [
            {"step": 1, "tool": "query_rag", "ok": True, "result": {"x": 1}, "payload": {"_step": 1, "x": 1}},
            {"step": 2, "tool": "pdf_export", "ok": False, "error": "boom", "payload": {"_step": 1, "x": 1}},
        ]
        compact = _compact_tool_outputs(full)
        self.assertNotIn("result", compact[0])
        self.assertEqual(compact[0]["payload"]["x"], 1)
        self.assertEqual(compact[1]["error"], "boom")

    def test_compact_outputs_drop_query_rag_text_when_hits_exist(self):
        full = [
            {
                "step": 1,
                "tool": "query_rag",
                "ok": True,
                "payload": {"_step": 1, "hits": [{"snippet": "x"}], "text": "redundant"},
            }
        ]
        compact = _compact_tool_outputs(full)
        self.assertIn("hits", compact[0]["payload"])
        self.assertNotIn("text", compact[0]["payload"])

    def test_compact_outputs_drop_llm_summary_text_when_summary_exists(self):
        full = [
            {
                "step": 2,
                "tool": "llm_summarize",
                "ok": True,
                "payload": {"_step": 2, "summary": "abc", "text": "abc"},
            }
        ]
        compact = _compact_tool_outputs(full)
        self.assertIn("summary", compact[0]["payload"])
        self.assertNotIn("text", compact[0]["payload"])

    def test_compact_outputs_drop_llm_compose_text_when_composed_exists(self):
        full = [
            {
                "step": 2,
                "tool": "llm_compose",
                "ok": True,
                "payload": {"_step": 2, "composed_text": "abc", "text": "abc"},
            }
        ]
        compact = _compact_tool_outputs(full)
        self.assertIn("composed_text", compact[0]["payload"])
        self.assertNotIn("text", compact[0]["payload"])

    def test_outputs_for_final_answer_is_token_lean(self):
        full = [
            {
                "step": 1,
                "tool": "query_rag",
                "ok": True,
                "payload": {"_step": 1, "hits": [{"snippet": "a"}, {"snippet": "b"}], "text": "x"},
            },
            {
                "step": 2,
                "tool": "llm_summarize",
                "ok": True,
                "payload": {"_step": 2, "summary": "ok", "usage": {"total_tokens": 123}, "model": "m"},
            },
        ]
        lean = _outputs_for_final_answer(full)
        self.assertNotIn("hits", lean[0]["payload"])
        self.assertEqual(lean[0]["payload"]["hit_count"], 2)
        self.assertNotIn("usage", lean[1]["payload"])
        self.assertNotIn("model", lean[1]["payload"])

    def test_outputs_for_final_answer_drop_llm_compose_model_usage(self):
        full = [
            {
                "step": 3,
                "tool": "llm_compose",
                "ok": True,
                "payload": {"_step": 3, "composed_text": "ok", "usage": {"total_tokens": 1}, "model": "m"},
            }
        ]
        lean = _outputs_for_final_answer(full)
        self.assertIn("composed_text", lean[0]["payload"])
        self.assertNotIn("usage", lean[0]["payload"])
        self.assertNotIn("model", lean[0]["payload"])

if __name__ == "__main__":
    unittest.main()
