from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List


class ManuallyTrigger:
    """
    Manual-only trigger.
    It never fires from polling; execution happens through /run-now.
    """

    def poll(self, now_utc: datetime) -> List[Dict[str, Any]]:
        _ = now_utc
        return []

