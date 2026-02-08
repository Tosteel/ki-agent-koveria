from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

@dataclass(frozen=True)
class Settings:
    base_dir: Path
    data_dir: Path
    rag_base_url: str = "http://localhost:8005"  # optional, falls du es nutzt

    def user_dir(self, user_id: str) -> Path:
        return self.data_dir / "users" / user_id

    def user_work_dir(self, user_id: str) -> Path:
        return self.user_dir(user_id) / "work"

    def user_rag_dir(self, user_id: str) -> Path:
        return self.user_dir(user_id) / "rag"

    def user_logs_dir(self, user_id: str) -> Path:
        return self.user_dir(user_id) / "logs"


_settings: Settings | None = None

def get_settings() -> Settings:
    global _settings
    if _settings is not None:
        return _settings

    base_dir = Path(os.getenv("KOVERIA_BASE_DIR", Path(__file__).resolve().parents[1]))
    data_dir = Path(os.getenv("KOVERIA_DATA_DIR", base_dir / "data"))
    rag_base_url = os.getenv("RAG_BASE_URL", "http://localhost:8005").rstrip("/")

    _settings = Settings(base_dir=base_dir, data_dir=data_dir, rag_base_url=rag_base_url)
    return _settings
