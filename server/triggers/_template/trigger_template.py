from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List


class TemplateTrigger:
    def __init__(self, *, interval_seconds: int):
        self.interval_seconds = max(10, int(interval_seconds))
        self._next_fire = datetime.now(timezone.utc) + timedelta(seconds=self.interval_seconds)

    def poll(self, now_utc: datetime) -> List[Dict[str, Any]]:
        if now_utc < self._next_fire:
            return []
        self._next_fire = now_utc + timedelta(seconds=self.interval_seconds)
        return [{"type": "template", "fired_at": now_utc.isoformat()}]

