import unittest
from unittest.mock import patch

from pydantic import BaseModel

from server.agent.tool_registry import ToolRegistry
from server.services.agent_service import run_planner_guard


class ViewWebsiteArgs(BaseModel):
    url: str
    query: str


class PlannerGuardRegistryBindingTests(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        self.registry.register("web_search_page", lambda _ctx, _args: {}, request_model=ViewWebsiteArgs)

    def test_replaces_llm_hallucinated_missing_arg_with_registry_required_arg(self):
        steps = [{"tool": "web_search_page", "args": {"url": "http://www.tagesschau.de"}}]

        with patch(
            "server.services.agent_service._llm_planner_guard",
            return_value={
                "status": "replan",
                "missing": ["missing_arg:web_search_page.search_query"],
                "reasons": ["Step 1: web_search_page benötigt search_query."],
            },
        ):
            gate = run_planner_guard(llm=object(), provider="ionos", goal="Suche nach Trump", steps=steps, registry=self.registry)

        self.assertEqual(gate.get("status"), "replan")
        missing = list(gate.get("missing") or [])
        reasons = list(gate.get("reasons") or [])
        self.assertIn("missing_arg:web_search_page.query", missing)
        self.assertNotIn("missing_arg:web_search_page.search_query", missing)
        self.assertTrue(any("Pflichtfeld 'query' fehlt" in r for r in reasons))
        self.assertFalse(any("search_query" in r for r in reasons))

    def test_hallucinated_missing_arg_does_not_force_replan_when_registry_is_valid(self):
        steps = [{"tool": "web_search_page", "args": {"url": "http://www.tagesschau.de", "query": "Wut auf Trump"}}]

        with patch(
            "server.services.agent_service._llm_planner_guard",
            return_value={
                "status": "replan",
                "missing": ["missing_arg:web_search_page.search_query"],
                "reasons": ["Step 1: web_search_page benötigt search_query."],
            },
        ):
            gate = run_planner_guard(llm=object(), provider="ionos", goal="Suche nach Trump", steps=steps, registry=self.registry)

        self.assertEqual(gate.get("status"), "ready")
        self.assertEqual(list(gate.get("missing") or []), [])
        self.assertEqual(list(gate.get("reasons") or []), [])


if __name__ == "__main__":
    unittest.main()
