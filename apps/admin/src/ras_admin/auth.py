"""Authentication: Open WebUI bridge + JWT session cookies."""

from __future__ import annotations

import datetime
from typing import Any

import httpx
import jwt
from fastapi import Request, Response
from fastapi.responses import RedirectResponse

from .config import AdminConfig

COOKIE_NAME = "admin_session"


def authenticate_with_open_webui(config: AdminConfig, email: str, password: str) -> dict[str, Any] | None:
    """POST credentials to Open WebUI's signin endpoint. Returns user info on success."""
    try:
        resp = httpx.post(
            f"{config.open_webui_url}/api/v1/auths/signin",
            json={"email": email, "password": password},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
        return None
    except httpx.HTTPError:
        return None


def create_session_token(config: AdminConfig, user_info: dict[str, Any]) -> str:
    """Create a JWT session token from Open WebUI user info."""
    payload = {
        "email": user_info.get("email", ""),
        "name": user_info.get("name", ""),
        "role": user_info.get("role", "user"),
        "exp": datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=config.session_expiry_hours),
    }
    return jwt.encode(payload, config.secret_key, algorithm="HS256")


def set_session_cookie(response: Response, token: str) -> None:
    """Set the session cookie on a response."""
    response.set_cookie(
        COOKIE_NAME,
        token,
        httponly=True,
        samesite="lax",
        max_age=86400,
        path="/",
    )


def get_current_user(request: Request) -> dict[str, Any] | None:
    """Extract and validate the session from the request cookie."""
    config: AdminConfig = request.app.state.config
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None
    try:
        payload = jwt.decode(token, config.secret_key, algorithms=["HS256"])
        return payload
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


def require_auth(request: Request) -> dict[str, Any] | RedirectResponse:
    """Dependency: returns user dict or redirects to login."""
    user = get_current_user(request)
    if user is None:
        return None
    return user
