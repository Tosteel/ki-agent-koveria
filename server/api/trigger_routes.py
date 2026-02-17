from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from server.core.settings import Settings
from server.deps import get_current_user, settings as dep_settings


class TriggerCreateRequest(BaseModel):
    name: str = Field(..., min_length=1)
    trigger_type: str = Field(..., min_length=1)
    task_id: int = Field(..., ge=1)
    config: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class TriggerUpdateRequest(BaseModel):
    name: Optional[str] = None
    trigger_type: Optional[str] = None
    task_id: Optional[int] = Field(default=None, ge=1)
    config: Optional[Dict[str, Any]] = None
    enabled: Optional[bool] = None


def create_trigger_router(
    *,
    ensure_user_dirs: Callable[[Settings, str], None],
    build_trigger_registry: Callable[[], Any],
    load_user_triggers: Callable[[Settings, str], Dict[str, Any]],
    save_user_triggers: Callable[[Settings, str, List[Dict[str, Any]]], None],
    now_iso: Callable[[], str],
    get_trigger_runtime: Callable[[], Any],
) -> APIRouter:
    router = APIRouter()

    @router.get('/triggers/types')
    def trigger_types() -> Dict[str, Any]:
        reg = build_trigger_registry()
        return {'types': reg.available_types()}

    @router.get('/triggers')
    def list_triggers(
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> Dict[str, Any]:
        ensure_user_dirs(s, user_id)
        payload = load_user_triggers(s, user_id)
        triggers = payload.get('triggers') if isinstance(payload.get('triggers'), list) else []
        return {'user_id': user_id, 'triggers': triggers}

    @router.post('/triggers')
    def create_trigger(
        req: TriggerCreateRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> Dict[str, Any]:
        ensure_user_dirs(s, user_id)
        reg = build_trigger_registry()
        reg.create_instance(req.trigger_type.strip(), req.config or {})

        payload = load_user_triggers(s, user_id)
        triggers = payload.get('triggers') if isinstance(payload.get('triggers'), list) else []
        trigger_id = str(uuid4())
        item = {
            'id': trigger_id,
            'name': req.name.strip(),
            'trigger_type': req.trigger_type.strip(),
            'task_id': int(req.task_id),
            'config': req.config or {},
            'enabled': bool(req.enabled),
            'created_at': now_iso(),
            'last_fired_at': '',
            'last_error': '',
        }
        triggers.append(item)
        save_user_triggers(s, user_id, triggers)
        return {'ok': True, 'trigger': item}

    @router.patch('/triggers/{trigger_id}')
    def update_trigger(
        trigger_id: str,
        req: TriggerUpdateRequest,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> Dict[str, Any]:
        ensure_user_dirs(s, user_id)
        payload = load_user_triggers(s, user_id)
        triggers = payload.get('triggers') if isinstance(payload.get('triggers'), list) else []
        updated: Optional[Dict[str, Any]] = None
        for t in triggers:
            if str(t.get('id') or '') != trigger_id:
                continue
            if req.name is not None:
                t['name'] = req.name.strip()
            if req.task_id is not None:
                t['task_id'] = int(req.task_id)
            if req.trigger_type is not None:
                trigger_type = req.trigger_type.strip()
                reg = build_trigger_registry()
                cfg_to_validate = req.config if req.config is not None else (t.get('config') if isinstance(t.get('config'), dict) else {})
                reg.create_instance(trigger_type, cfg_to_validate or {})
                t['trigger_type'] = trigger_type
                t['config'] = cfg_to_validate or {}
            if req.config is not None:
                reg = build_trigger_registry()
                reg.create_instance(str(t.get('trigger_type') or ''), req.config or {})
                t['config'] = req.config or {}
            if req.enabled is not None:
                t['enabled'] = bool(req.enabled)
            updated = t
            break
        if updated is None:
            return {'ok': False, 'error': 'trigger_not_found'}

        save_user_triggers(s, user_id, triggers)
        return {'ok': True, 'trigger': updated}

    @router.delete('/triggers/{trigger_id}')
    def delete_trigger(
        trigger_id: str,
        user_id: str = Depends(get_current_user),
        s: Settings = Depends(dep_settings),
    ) -> Dict[str, Any]:
        ensure_user_dirs(s, user_id)
        payload = load_user_triggers(s, user_id)
        triggers = payload.get('triggers') if isinstance(payload.get('triggers'), list) else []
        kept: List[Dict[str, Any]] = []
        deleted = False
        for t in triggers:
            if str(t.get('id') or '') == trigger_id:
                deleted = True
                continue
            kept.append(t)
        if not deleted:
            return {'ok': False, 'error': 'trigger_not_found'}
        save_user_triggers(s, user_id, kept)
        return {'ok': True, 'trigger_id': trigger_id}

    @router.post('/triggers/{trigger_id}/run-now')
    def run_trigger_now(
        trigger_id: str,
        user_id: str = Depends(get_current_user),
    ) -> Dict[str, Any]:
        runtime = get_trigger_runtime()
        try:
            result = runtime.run_trigger_now(user_id=user_id, trigger_id=trigger_id)
        except ValueError as exc:
            return {'ok': False, 'error': str(exc)}
        return {'ok': bool(result.get('ok')), 'result': result}

    return router
