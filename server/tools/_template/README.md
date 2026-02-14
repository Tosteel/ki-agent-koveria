# Tool Template

Use this folder as blueprint for a new tool under `server/tools/<toolname>/`.

## Quick Start

1. Copy folder:

```bash
cp -r server/tools/_template server/tools/<toolname>
```

2. Rename implementation file:

- `tool_template.py` -> `<toolname>.py`
- Update imports in `registry.py`

3. Update models:

- `TemplateToolRequest`
- `TemplateToolResponse`

4. Update registry:

- `TOOL_NAME = "<toolname>"`
- handler internals and imports

5. Allow planner usage (optional):

- Add `<toolname>` to `server/agent/policies.py` `PHASE1_ALLOWED_TOOLS`.

## Notes

- Loader auto-discovers tools by `server/tools/*/registry.py` and calls `register(registry)`.
- Keep request/response models small and explicit.
- Return JSON-serializable payload only.
