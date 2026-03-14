import types
import unittest

from server.tools.llm_text import llm_text_summarize


class _FakeLlm:
    def __init__(self, enabled: bool, text: str = "Kurzfassung", model: str = "fake-model"):
        self._enabled = enabled
        self._text = text
        self.cfg = types.SimpleNamespace(model=model)
        self.last_messages = []

    def enabled(self) -> bool:
        return self._enabled

    def chat_completions(self, **kwargs):
        self.last_messages = kwargs.get("messages") or []
        return {"choices": [{"message": {"content": self._text}}], "usage": {"total_tokens": 12}}

    @staticmethod
    def extract_text(completion):
        return completion["choices"][0]["message"]["content"]

    @staticmethod
    def extract_usage(completion):
        return completion.get("usage")


class LlmSummaryToolTests(unittest.TestCase):
    def test_uses_llm_when_enabled(self):
        fake = _FakeLlm(enabled=True, text="Zusammenfassung")
        res = llm_text_summarize(
            "Preis: 60,00 EUR",
            goal="Bitte mit Quelle zusammenfassen",
            llm=fake,
        )
        self.assertEqual(res["summary"], "Zusammenfassung")
        self.assertFalse(res["fallback_used"])
        self.assertEqual(res["model"], "fake-model")
        self.assertIn("Bitte mit Quelle zusammenfassen", fake.last_messages[1]["content"])

    def test_fallback_when_disabled(self):
        res = llm_text_summarize("Zeile 1\nZeile 2", llm=_FakeLlm(enabled=False))
        self.assertTrue(res["fallback_used"])
        self.assertIn("Zeile 1", res["summary"])


if __name__ == "__main__":
    unittest.main()
