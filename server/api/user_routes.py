from __future__ import annotations

from typing import Dict

from fastapi import APIRouter, Depends

from server.deps import get_current_user

router = APIRouter()


@router.get('/health')
def health(user_id: str = Depends(get_current_user)) -> Dict[str, str]:
    return {'status': 'ok', 'user': user_id}


@router.get('/user')
def user(user_id: str = Depends(get_current_user)) -> Dict[str, str]:
    return {'user_id': user_id}
