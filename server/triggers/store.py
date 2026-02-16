from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from server.core.settings import Settings


def user_triggers_path(s: Settings, user_id: str) -> Path:
    return s.user_dir(user_id) / "triggers_config.json"


def user_trigger_runs_path(s: Settings, user_id: str) -> Path:
    return s.user_logs_dir(user_id) / "trigger_runs.jsonl"


def user_tasks_memory_path(s: Settings, user_id: str) -> Path:
    return s.user_dir(user_id) / "tasks_memory.json"


def user_tasks_memory_legacy_client_path(s: Settings, user_id: str) -> Path:
    return s.base_dir.parent / "client" / "data" / "users" / user_id / "tasks_memory.json"


def load_user_triggers(s: Settings, user_id: str) -> Dict[str, Any]:
    path = user_triggers_path(s, user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return {"version": 1, "triggers": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "triggers": []}
    if not isinstance(data, dict):
        return {"version": 1, "triggers": []}
    triggers = data.get("triggers")
    if not isinstance(triggers, list):
        triggers = []
    return {"version": 1, "triggers": triggers}


def save_user_triggers(s: Settings, user_id: str, triggers: List[Dict[str, Any]]) -> None:
    path = user_triggers_path(s, user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "triggers": triggers}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def list_users_with_triggers(s: Settings) -> List[str]:
    users_dir = s.data_dir / "users"
    if not users_dir.exists():
        return []
    users: List[str] = []
    for child in sorted(users_dir.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        if (child / "triggers_config.json").exists():
            users.append(child.name)
    return users


def load_task_by_id(s: Settings, user_id: str, task_id: int) -> Optional[Dict[str, Any]]:
    candidate_paths = [
        user_tasks_memory_path(s, user_id),
        user_tasks_memory_legacy_client_path(s, user_id),
    ]
    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        tasks = data.get("tasks")
        if not isinstance(tasks, list):
            continue
        for task in tasks:
            if not isinstance(task, dict):
                continue
            if int(task.get("id") or 0) == int(task_id):
                return task
    return None


def append_trigger_run_log(s: Settings, user_id: str, item: Dict[str, Any]) -> None:
    path = user_trigger_runs_path(s, user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
