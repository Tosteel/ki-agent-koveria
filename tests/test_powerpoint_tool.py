import types
import unittest

from server.tools.powerpoint import (
    _layout_text_for_slides_llm_with_mode,
    layout_text_for_slides,
    layout_text_for_slides_llm,
)


class _FakeLlm:
    def __init__(self, enabled: bool, content: str):
        self._enabled = enabled
        self._content = content
        self.cfg = types.SimpleNamespace(model="fake-model")

    def enabled(self) -> bool:
        return self._enabled

    def chat_completions(self, **_kwargs):
        return {"choices": [{"message": {"content": self._content}}]}

    @staticmethod
    def extract_text(completion):
        return completion["choices"][0]["message"]["content"]


class PowerPointLayoutTests(unittest.TestCase):
    def test_layout_splits_long_text_across_multiple_slides(self):
        text = "\n\n".join([f"Absatz {i}: " + ("Inhalt " * 40) for i in range(1, 12)])
        slides = layout_text_for_slides("Projektbericht", text, max_chars_per_slide=450, max_lines_per_slide=8)
        self.assertGreaterEqual(len(slides), 2)
        self.assertEqual(slides[0]["title"], "Projektbericht")
        self.assertIn("Teil", slides[1]["title"])

    def test_layout_creates_multiple_text_boxes(self):
        text = "- Punkt A " + ("x " * 25) + "\n\n" + "- Punkt B " + ("y " * 25) + "\n\n" + "- Punkt C " + ("z " * 25)
        slides = layout_text_for_slides("Kunden", text, max_chars_per_slide=900, max_lines_per_slide=12, max_lines_per_box=3)
        self.assertGreaterEqual(len(slides), 1)
        self.assertGreaterEqual(len(slides[0]["boxes"]), 2)

    def test_llm_layout_uses_structured_output(self):
        llm_json = (
            '{"slides":['
            '{"title":"Kunden","boxes":["Kunde A mit Quelle X","Kunde B mit Quelle Y"]},'
            '{"title":"Fazit","boxes":["Kurzfazit"]}'
            ']}'
        )
        slides = layout_text_for_slides_llm(
            "Kunden",
            "rohtext",
            llm=_FakeLlm(enabled=True, content=llm_json),
        )
        self.assertEqual(len(slides), 2)
        self.assertEqual(slides[0]["title"], "Kunden")
        self.assertEqual(len(slides[0]["boxes"]), 2)
        slides2, mode, reason = _layout_text_for_slides_llm_with_mode(
            "Kunden",
            "rohtext",
            llm=_FakeLlm(enabled=True, content=llm_json),
        )
        self.assertEqual(mode, "llm")
        self.assertEqual(reason, "")
        self.assertEqual(len(slides2), 2)

    def test_llm_layout_falls_back_when_invalid_json(self):
        slides = layout_text_for_slides_llm(
            "Kunden",
            "Absatz 1\n\nAbsatz 2",
            llm=_FakeLlm(enabled=True, content="kein-json"),
        )
        self.assertGreaterEqual(len(slides), 1)
        _slides2, mode, reason = _layout_text_for_slides_llm_with_mode(
            "Kunden",
            "Absatz 1\n\nAbsatz 2",
            llm=_FakeLlm(enabled=True, content="kein-json"),
        )
        self.assertEqual(mode, "heuristic")
        self.assertTrue(reason)


if __name__ == "__main__":
    unittest.main()
