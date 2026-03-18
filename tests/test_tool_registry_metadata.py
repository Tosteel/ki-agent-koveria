import unittest

from pydantic import BaseModel

from server.agent.tool_registry import ToolRegistry


class RagArgs(BaseModel):
    query: str


class RagOut(BaseModel):
    text: str = ""


class NestedPayload(BaseModel):
    name: str


class NestedArgs(BaseModel):
    payload: NestedPayload


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

    def test_planner_schema_inlines_local_defs_in_args(self):
        reg = ToolRegistry()
        reg.register("rag_knowledgebase", lambda _ctx, _args: {}, request_model=NestedArgs, response_model=RagOut)

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
        args_schema = rag_item.get("properties", {}).get("args", {})
        payload_schema = args_schema.get("properties", {}).get("payload", {})

        self.assertTrue(isinstance(payload_schema, dict))
        self.assertNotIn("$ref", payload_schema)
        self.assertNotIn("$defs", args_schema)
        self.assertEqual(payload_schema.get("type"), "object")

    def test_tool_map_metadata_is_applied_per_tool(self):
        class MailArgs(BaseModel):
            to: list[str]
            subject: str
            body: str

        reg = ToolRegistry()
        reg.register("mail_send", lambda _ctx, _args: {"sent": True}, request_model=MailArgs)
        meta = reg.tool_metadata("mail_send")
        policy = reg.tool_policy("mail_send")

        self.assertEqual(str(meta.get("name") or ""), "E-Mail senden")
        self.assertEqual(str(meta.get("side_effect_level") or ""), "high")
        self.assertIn("communication:email_send", list(policy.get("capabilities") or []))
        self.assertEqual(str(policy.get("side_effect_level") or ""), "high")


if __name__ == "__main__":
    unittest.main()
