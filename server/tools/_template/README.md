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
- Update imports in `api.py`

3. Update models:

- `TemplateToolRequest`
- `TemplateToolResponse`

4. Update registry:

- `TOOL_NAME = "<toolname>"`
- handler internals and imports

5. Update API route file:

- Adjust endpoint path in `api.py`
- Keep `create_router(*, ensure_user_dirs)` signature

6. Allow planner usage (optional):

- Add `<toolname>` to `server/agent/policies.py` `BASIC_TOOLS`.

## Notes

- Loader auto-discovers tools by `server/tools/**/registry.py` and calls `register(registry)`.
- Tool API router auto-discovers `server/tools/**/api.py` and calls `create_router(...)`.
- Keep request/response models small and explicit.
- Return JSON-serializable payload only.
