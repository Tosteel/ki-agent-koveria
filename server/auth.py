# server/auth.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# API Keys -> User (später DB oder JWT)
API_KEYS = {
    "46deb572-0c06-4515-b6f0-4e7dc2aeff82": "user1", #Standard GUI
    "0cc77b0e-c381-471d-bfbd-4fb057cd6b91": "user2", # Wettbewerbsanalyse
    "a9c48f5a-446c-4ee3-84ec-e4d1b7bee79c": "user3", # MatchUp
    "59e81efc-57a2-4825-8f96-6b8be1f2e101": "user4", # Mail_assistant
    "3015214b-9604-4f59-b7dd-0e349788e2ac": "user5",  # Competitive_Intelligence
    "f202328e-be6c-462a-b842-dcd47eab5978": "user6",
    "6083d3de-a913-4ec8-94e8-2d76bb9ea5fa": "user7",
    "ad6c977d-c53b-4890-ba67-dbd75b53a41c": "user8",
    "171b5eb0-ff01-4a77-acc3-4db6af9b6f80": "user9",
    "1923bf64-4b64-436b-808a-c538a7a0f0da": "user10",
}


def get_token_for_user(user_id: str) -> str:
    for token, uid in API_KEYS.items():
        if uid == user_id:
            return token
    return ""

security = HTTPBearer(auto_error=False)

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    # Kein Header oder falsches Schema
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header. Expected: Bearer <token>",
        )

    token = credentials.credentials  # Der Teil NACH "Bearer"
    user_id = API_KEYS.get(token)

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid token",
        )

    return user_id
