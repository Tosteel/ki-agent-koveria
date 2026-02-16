from __future__ import annotations

import json
import ast
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from server.core.settings import Settings

from .store import append_trigger_run_log, list_users_with_triggers, load_task_by_id, load_user_triggers, save_user_triggers
from .trigger_registry import TriggerRegistry, TriggerInstance


StepExecutor = Callable[[str, List[Dict[str, Any]], str], List[Dict[str, Any]]]


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_planned_steps(steps: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for raw in steps:
        line = str(raw or "").strip()
        if not line:
            continue
        # Format from client: "1. tool=name args={...}"
        marker = "tool="
        args_marker = " args="
        i = line.find(marker)
        j = line.find(args_marker)
        if i < 0 or j < 0 or j <= i:
            continue
        tool = line[i + len(marker) : j].strip()
        args_raw = line[j + len(args_marker) :].strip()
        if not tool:
            continue
        args: Dict[str, Any] = {}
        if args_raw:
            try:
                loaded = json.loads(args_raw)
                if isinstance(loaded, dict):
                    args = loaded
            except Exception:
                try:
                    loaded = ast.literal_eval(args_raw)
                    if isinstance(loaded, dict):
                        args = loaded
                except Exception:
                    args = {}
        out.append({"tool": tool, "args": args})
    return out


def _extract_answer(outputs: List[Dict[str, Any]]) -> str:
    for out in reversed(outputs):
        if not isinstance(out, dict) or not out.get("ok"):
            continue
        payload = out.get("payload")
        if not isinstance(payload, dict):
            continue
        for key in ("composed_text", "text", "summary", "answer", "message"):
            v = payload.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return "Trigger ausgeführt."


class TriggerRuntime:
    def __init__(self, *, settings: Settings, registry: TriggerRegistry, step_executor: StepExecutor, poll_seconds: float = 1.0):
        self.settings = settings
        self.registry = registry
        self.step_executor = step_executor
        self.poll_seconds = max(0.5, float(poll_seconds))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._instances: Dict[Tuple[str, str], TriggerInstance] = {}
        self._instance_specs: Dict[Tuple[str, str], str] = {}
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, name="koveria-trigger-runtime", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)

    def run_trigger_now(self, *, user_id: str, trigger_id: str) -> Dict[str, Any]:
        with self._lock:
            all_data = load_user_triggers(self.settings, user_id)
            triggers = all_data.get("triggers") if isinstance(all_data.get("triggers"), list) else []
            trigger = next((t for t in triggers if str(t.get("id") or "") == trigger_id), None)
            if trigger is None:
                raise ValueError("Trigger not found")
            # Manual execution intentionally ignores the enabled flag.
            result = self._execute_trigger(
                user_id,
                trigger,
                {"manual": True, "fired_at": _now_iso(), "enabled_at_run": bool(trigger.get("enabled", True))},
            )
            trigger["last_fired_at"] = _now_iso()
            if result.get("ok"):
                trigger["last_error"] = ""
            else:
                trigger["last_error"] = str(result.get("error") or "execution_failed")
            save_user_triggers(self.settings, user_id, triggers)
            return result

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as exc:
                print(f"[trigger-runtime] tick error: {exc}")
            self._stop.wait(self.poll_seconds)

    def _tick(self) -> None:
        with self._lock:
            users = list_users_with_triggers(self.settings)
            active_keys: set[Tuple[str, str]] = set()

            for user_id in users:
                payload = load_user_triggers(self.settings, user_id)
                triggers = payload.get("triggers") if isinstance(payload.get("triggers"), list) else []
                changed = False
                now_utc = datetime.now(timezone.utc)

                for t in triggers:
                    trigger_id = str(t.get("id") or "").strip()
                    if not trigger_id:
                        continue
                    key = (user_id, trigger_id)
                    active_keys.add(key)
                    if not bool(t.get("enabled", True)):
                        self._instances.pop(key, None)
                        self._instance_specs.pop(key, None)
                        continue

                    trigger_type = str(t.get("trigger_type") or "").strip()
                    config = t.get("config") if isinstance(t.get("config"), dict) else {}
                    spec = f"{trigger_type}:{json.dumps(config, ensure_ascii=False, sort_keys=True)}"
                    inst = self._instances.get(key)

                    if inst is None or self._instance_specs.get(key) != spec:
                        try:
                            inst = self.registry.create_instance(trigger_type, config)
                        except Exception as exc:
                            t["last_error"] = f"trigger_init_failed: {exc}"
                            changed = True
                            continue
                        self._instances[key] = inst
                        self._instance_specs[key] = spec

                    try:
                        events = inst.poll(now_utc)
                    except Exception as exc:
                        t["last_error"] = f"trigger_poll_failed: {exc}"
                        changed = True
                        continue

                    if not events:
                        continue

                    for event in events:
                        result = self._execute_trigger(user_id, t, event)
                        t["last_fired_at"] = _now_iso()
                        if result.get("ok"):
                            t["last_error"] = ""
                        else:
                            t["last_error"] = str(result.get("error") or "execution_failed")
                        changed = True

                if changed:
                    save_user_triggers(self.settings, user_id, triggers)

            # Drop cached instances that are no longer present
            stale = [k for k in self._instances.keys() if k not in active_keys]
            for k in stale:
                self._instances.pop(k, None)
                self._instance_specs.pop(k, None)

    def _execute_trigger(self, user_id: str, trigger: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        trigger_id = str(trigger.get("id") or "")
        trigger_name = str(trigger.get("name") or trigger_id)
        task_id = int(trigger.get("task_id") or 0)

        task = load_task_by_id(self.settings, user_id, task_id)
        if task is None:
            result = {"ok": False, "error": "task_not_found", "answer": ""}
            append_trigger_run_log(
                self.settings,
                user_id,
                {
                    "at": _now_iso(),
                    "trigger_id": trigger_id,
                    "trigger_name": trigger_name,
                    "task_id": task_id,
                    "ok": False,
                    "error": "task_not_found",
                    "event": event,
                },
            )
            return result

        planned_steps = [str(s).strip() for s in (task.get("planned_steps") or []) if str(s).strip()]
        steps = _parse_planned_steps(planned_steps)
        if not steps:
            result = {"ok": False, "error": "planned_steps_empty", "answer": ""}
            append_trigger_run_log(
                self.settings,
                user_id,
                {
                    "at": _now_iso(),
                    "trigger_id": trigger_id,
                    "trigger_name": trigger_name,
                    "task_id": task_id,
                    "ok": False,
                    "error": "planned_steps_empty",
                    "event": event,
                },
            )
            return result

        print("\n===== TRIGGER RUN =====")
        print(f"user_id={user_id}")
        print(f"trigger_id={trigger_id}")
        print(f"trigger_name={trigger_name}")
        print(f"task_id={task_id}")
        print(f"event={event}")
        print("=======================\n")

        print("\n===== PLANNED STEPS =====")
        for i, step in enumerate(steps, 1):
            print(f"{i}. tool={step.get('tool')} args={step.get('args')}")
        print("=========================\n")

        goal = "TRIGGER EVENT:\n" + json.dumps(event, ensure_ascii=False) + "\n\nPLANNED STEPS:\n" + "\n".join(planned_steps)
        outputs = self.step_executor(user_id, steps, goal)
        ok = all(bool(o.get("ok")) for o in outputs) if outputs else True
        answer = _extract_answer(outputs)

        print("\n===== FINAL ANSWER =====")
        print(answer)
        print("========================\n")

        append_trigger_run_log(
            self.settings,
            user_id,
            {
                "at": _now_iso(),
                "trigger_id": trigger_id,
                "trigger_name": trigger_name,
                "task_id": task_id,
                "ok": ok,
                "answer": answer,
                "event": event,
            },
        )
        return {"ok": ok, "answer": answer}
