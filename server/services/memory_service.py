from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from server.core.settings import Settings


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def ensure_user_dirs(s: Settings, user_id: str) -> None:
    s.user_work_dir(user_id).mkdir(parents=True, exist_ok=True)
    s.user_rag_dir(user_id).mkdir(parents=True, exist_ok=True)
    s.user_logs_dir(user_id).mkdir(parents=True, exist_ok=True)


def user_tasks_memory_path(s: Settings, user_id: str) -> Path:
    return s.user_dir(user_id) / 'tasks_memory.json'


def user_agents_memory_path(s: Settings, user_id: str) -> Path:
    return s.user_dir(user_id) / 'template_config.json'


def normalize_tasks_payload(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    tasks: List[Dict[str, Any]] = []
    for t in raw:
        if not isinstance(t, dict):
            continue
        task_id = int(t.get('id') or 0)
        text = str(t.get('text') or '').strip()
        if task_id <= 0 or not text:
            continue
        reruns_raw = t.get('reruns')
        reruns: List[Dict[str, Any]] = []
        if isinstance(reruns_raw, list):
            for r in reruns_raw:
                if not isinstance(r, dict):
                    continue
                answer = str(r.get('answer') or '').strip()
                if not answer:
                    continue
                reruns.append({'answer': answer, 'created_at': str(r.get('created_at') or '')})
        dialog_raw = t.get('dialog')
        dialog: List[Dict[str, Any]] = []
        if isinstance(dialog_raw, list):
            for m in dialog_raw:
                if not isinstance(m, dict):
                    continue
                msg_text = str(m.get('text') or '').strip()
                if not msg_text:
                    continue
                dialog.append(
                    {
                        'role': 'user' if str(m.get('role') or '').strip().lower() == 'user' else 'bot',
                        'text': msg_text,
                        'plannedSteps': [str(s).strip() for s in (m.get('plannedSteps') or []) if str(s).strip()],
                        'options': [
                            {
                                'type': str(o.get('type') or '').strip(),
                                'taskId': int(o.get('taskId') or 0),
                                'created': str(o.get('created') or '').strip(),
                                'plannedSteps': [str(s).strip() for s in (o.get('plannedSteps') or []) if str(s).strip()],
                            }
                            for o in (m.get('options') or [])
                            if isinstance(o, dict) and str(o.get('type') or '').strip()
                        ],
                        'timestamp': str(m.get('timestamp') or ''),
                    }
                )
        tasks.append(
            {
                'id': task_id,
                'title': str(t.get('title') or '').strip(),
                'text': text,
                'planned_steps': [str(s).strip() for s in (t.get('planned_steps') or []) if str(s).strip()],
                'planned_steps_text': str(t.get('planned_steps_text') or '').strip(),
                'created_at': str(t.get('created_at') or ''),
                'dialog': dialog,
                'reruns': reruns,
            }
        )
    return tasks


def load_tasks_memory_for_user(s: Settings, user_id: str) -> Dict[str, Any]:
    path = user_tasks_memory_path(s, user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return {'tasks': []}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {'tasks': []}
    if not isinstance(data, dict):
        return {'tasks': []}
    return {'tasks': normalize_tasks_payload(data.get('tasks'))}


def save_tasks_memory_for_user(s: Settings, user_id: str, tasks: List[Dict[str, Any]]) -> None:
    path = user_tasks_memory_path(s, user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'version': 1,
        'updated_at': now_iso(),
        'tasks': normalize_tasks_payload(tasks),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def normalize_agents_payload(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    agents: List[Dict[str, Any]] = []
    for a in raw:
        if not isinstance(a, dict):
            continue
        agent_id = int(a.get('id') or 0)
        text = str(a.get('text') or '').strip()
        if agent_id <= 0 or not text:
            continue
        dialog_raw = a.get('dialog')
        dialog: List[Dict[str, Any]] = []
        if isinstance(dialog_raw, list):
            for m in dialog_raw:
                if not isinstance(m, dict):
                    continue
                msg_text = str(m.get('text') or '').strip()
                if not msg_text:
                    continue
                dialog.append(
                    {
                        'role': 'user' if str(m.get('role') or '').strip().lower() == 'user' else 'bot',
                        'text': msg_text,
                        'plannedSteps': [str(s).strip() for s in (m.get('plannedSteps') or []) if str(s).strip()],
                        'timestamp': str(m.get('timestamp') or ''),
                    }
                )
        placeholders_raw = a.get('placeholders')
        placeholders: List[Dict[str, Any]] = []
        if isinstance(placeholders_raw, list):
            for p in placeholders_raw:
                if not isinstance(p, dict):
                    continue
                name = str(p.get('name') or '').strip().lower()
                if not name:
                    continue
                placeholders.append(
                    {
                        'name': name,
                        'type': str(p.get('type') or 'string').strip().lower() or 'string',
                        'required': bool(p.get('required', True)),
                        'description': str(p.get('description') or '').strip(),
                        'used_in': [str(u).strip() for u in (p.get('used_in') or []) if str(u).strip()],
                    }
                )
        agents.append(
            {
                'id': agent_id,
                'title': str(a.get('title') or '').strip(),
                'text': text,
                'planned_steps': [str(s).strip() for s in (a.get('planned_steps') or []) if str(s).strip()],
                'created_at': str(a.get('created_at') or ''),
                'source_task_id': int(a.get('source_task_id') or 0),
                'placeholders': placeholders,
                'dialog': dialog,
            }
        )
    return agents


def load_agents_memory_for_user(s: Settings, user_id: str) -> Dict[str, Any]:
    path = user_agents_memory_path(s, user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return {'agents': []}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {'agents': []}
    if not isinstance(data, dict):
        return {'agents': []}
    return {'agents': normalize_agents_payload(data.get('agents'))}


def save_agents_memory_for_user(s: Settings, user_id: str, agents: List[Dict[str, Any]]) -> None:
    path = user_agents_memory_path(s, user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'version': 1,
        'updated_at': now_iso(),
        'agents': normalize_agents_payload(agents),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
