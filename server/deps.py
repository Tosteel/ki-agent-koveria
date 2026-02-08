from fastapi import Depends
from .auth import get_current_user as _get_current_user
from .core.settings import Settings, get_settings


def get_current_user(user_id: str = Depends(_get_current_user)) -> str:
    return user_id

def settings(s: Settings = Depends(get_settings)) -> Settings:
    return s
