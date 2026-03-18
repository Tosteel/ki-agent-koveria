from __future__ import annotations

import json
import os
from urllib.parse import parse_qs, urlencode, urlparse

import requests


AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _extract_code(user_input: str) -> str:
    raw = (user_input or "").strip()
    if not raw:
        return ""
    if raw.startswith("http://") or raw.startswith("https://"):
        parsed = urlparse(raw)
        values = parse_qs(parsed.query)
        code = values.get("code", [""])[0]
        return str(code or "").strip()
    return raw


def main() -> None:
    client_id = _env("GOOGLE_OAUTH_CLIENT_ID")
    client_secret = _env("GOOGLE_OAUTH_CLIENT_SECRET")
    redirect_uri = _env("GOOGLE_OAUTH_REDIRECT_URI", "http://localhost:8080/oauth2callback")
    scope = _env("GOOGLE_OAUTH_SCOPE", "https://www.googleapis.com/auth/calendar")

    if not client_id or not client_secret:
        raise SystemExit("Missing GOOGLE_OAUTH_CLIENT_ID/GOOGLE_OAUTH_CLIENT_SECRET")

    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": scope,
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
    }
    auth_link = f"{AUTH_URL}?{urlencode(params)}"
    print("\nOpen this URL in your browser and authorize:\n")
    print(auth_link)
    print("\nAfter redirect, paste either the full redirect URL or just the code:\n")
    code_input = input("Code/URL: ").strip()
    code = _extract_code(code_input)
    if not code:
        raise SystemExit("No authorization code provided")

    payload = {
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }
    resp = requests.post(TOKEN_URL, data=payload, timeout=30)
    try:
        data = resp.json()
    except Exception:
        data = {"raw": resp.text}

    if resp.status_code >= 400:
        print("\nToken exchange failed:\n")
        print(json.dumps(data, ensure_ascii=False, indent=2))
        raise SystemExit(1)

    access_token = str(data.get("access_token") or "").strip()
    refresh_token = str(data.get("refresh_token") or "").strip()
    expires_in = data.get("expires_in")

    print("\nSuccess. Tokens:\n")
    print(f"access_token: {access_token[:24]}... (len={len(access_token)})")
    print(f"refresh_token: {refresh_token[:24]}... (len={len(refresh_token)})")
    print(f"expires_in: {expires_in}")

    print("\nAdd these to your .env:\n")
    print(f"GOOGLE_ACCESS_TOKEN={access_token}")
    if refresh_token:
        print(f"GOOGLE_OAUTH_REFRESH_TOKEN={refresh_token}")
    print(f"GOOGLE_OAUTH_CLIENT_ID={client_id}")
    print(f"GOOGLE_OAUTH_CLIENT_SECRET={client_secret}")
    print(f"GOOGLE_OAUTH_REDIRECT_URI={redirect_uri}")


if __name__ == "__main__":
    main()
