import unittest

from pydantic import BaseModel

from server.agent.tool_registry import ToolRegistry


class RagArgs(BaseModel):
    query: str


class RagOut(BaseModel):
    text: str = ""


class ToolRegistryMetadataTests(unittest.TestCase):
    def test_planner_schema_includes_metadata_description_when_available(self):
        reg = ToolRegistry()
        reg.register("rag_knowledgebase", lambda _ctx, _args: {}, request_model=RagArgs, response_model=RagOut)

        schema = reg.planner_schema()
        items = (
            schema.get("schema", {})
            .get("properties", {})
            .get("steps", {})
            .get("items", {})
            .get("oneOf", [])
        )

        rag_item = next(
            (it for it in items if (it.get("properties", {}).get("tool", {}).get("const") == "rag_knowledgebase")),
            {},
        )
        desc = str(rag_item.get("description") or "")
        tool_desc = str(rag_item.get("properties", {}).get("tool", {}).get("description") or "")
        args_schema = rag_item.get("properties", {}).get("args", {})

        self.assertIn("Description:", desc)
        self.assertIn("Input:", desc)
        self.assertIn("Output:", desc)
        self.assertIn("Output fields:", desc)
        self.assertIn("Required output fields:", desc)
        self.assertEqual(tool_desc, "")
        self.assertEqual(args_schema.get("additionalProperties"), False)


if __name__ == "__main__":
    unittest.main()
