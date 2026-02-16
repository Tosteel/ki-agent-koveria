Use this folder as blueprint for a new trigger under `server/triggers/<triggername>/`.

Steps:
1. Copy folder:
   `cp -r server/triggers/_template server/triggers/<triggername>`
2. Rename implementation:
   - `trigger_template.py` -> `<triggername>.py`
   - fix imports in `registry.py`
3. Adapt `models.py` for trigger config
4. Register in `registry.py` with unique trigger type name

Loader auto-discovers triggers by `server/triggers/*/registry.py`.

