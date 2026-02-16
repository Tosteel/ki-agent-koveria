from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List


class TimeScheduleTrigger:
    def __init__(self, *, interval_seconds: int, fire_immediately: bool = False):
        self.interval_seconds = max(10, int(interval_seconds))
        now = datetime.now(timezone.utc)
        if fire_immediately:
            self._next_fire = now
        else:
            self._next_fire = now + timedelta(seconds=self.interval_seconds)

    def poll(self, now_utc: datetime) -> List[Dict[str, Any]]:
        if now_utc < self._next_fire:
            return []
        fired_at = now_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        scheduled_for = self._next_fire.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        self._next_fire = now_utc + timedelta(seconds=self.interval_seconds)
        return [{"type": "time_schedule", "fired_at": fired_at, "scheduled_for": scheduled_for}]

