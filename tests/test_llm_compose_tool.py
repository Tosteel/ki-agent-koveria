import types
import unittest

from server.tools.llm_compose import llm_compose_text


class _FakeLlm:
    def __init__(self, enabled: bool, text: str = "Ausformulierter Text", model: str = "fake-model"):
        self._enabled = enabled
        self._text = text
        self.cfg = types.SimpleNamespace(model=model)
        self.last_messages = []

    def enabled(self) -> bool:
        return self._enabled

    def chat_completions(self, **kwargs):
        self.last_messages = kwargs.get("messages") or []
        return {"choices": [{"message": {"content": self._text}}], "usage": {"total_tokens": 10}}

    @staticmethod
    def extract_text(completion):
        return completion["choices"][0]["message"]["content"]

    @staticmethod
    def extract_usage(completion):
        return completion.get("usage")


class LlmComposeToolTests(unittest.TestCase):
    def test_uses_llm_when_enabled(self):
        fake = _FakeLlm(enabled=True, text="Kohärenter Text.")
        res = llm_compose_text(
            "- Kunde A\n- Kunde B",
            goal="Formuliere einen zusammenhängenden Text mit Quelle.",
            llm=fake,
        )
        self.assertEqual(res["text"], "Kohärenter Text.")
        self.assertFalse(res["fallback_used"])
        self.assertIn("Formuliere einen zusammenhängenden Text", fake.last_messages[1]["content"])

    def test_fallback_when_disabled(self):
        res = llm_compose_text("- A\n- B", llm=_FakeLlm(enabled=False))
        self.assertTrue(res["fallback_used"])
        self.assertIn("A", res["text"])


if __name__ == "__main__":
    unittest.main()
