"""What every request needs: the shared services (one set per process, no per-session state) and
the authentication / CSRF / pagination dependencies (docs/adr/0016, 0018)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Annotated

from fastapi import Depends, Header, Query, Request, Response

from api.errors import ApiError
from infra.auth import AuthError, Principal, Tokens, UserStore
from infra.points import PointCache
from infra.settings import Settings

ACCESS_COOKIE, REFRESH_COOKIE, CSRF_COOKIE = "lal_access", "lal_refresh", "lal_csrf"
CSRF_HEADER, CLIENT_HEADER = "X-CSRF-Token", "X-Client-Id"
REFRESH_PATH = "/api/v1/auth"
SAFE_METHODS = ("GET", "HEAD", "OPTIONS")
CLIENT_ID = re.compile(r"^[A-Za-z0-9_-]{8,40}$")


@dataclass
class Services:
    settings: Settings
    engine: object
    users: UserStore
    tokens: Tokens
    sessions: object            # PgSessionStore
    jobs: object                # PgJobStore
    queue: object               # CeleryTaskQueue
    sequences: object           # SequenceStore
    tabs: object                # TabStore
    ping_redis: object          # () -> None, raises when Redis is unreachable
    point_cache: PointCache = field(default_factory=lambda: PointCache(64))


def services(request: Request) -> Services:
    return request.app.state.services


Svc = Annotated[Services, Depends(services)]


# --------------------------------------------------------------------- cookies
def set_login_cookies(response: Response, svc: Services, user: Principal, csrf: str) -> None:
    s, secure = svc.settings, svc.settings.cookie_secure
    response.set_cookie(ACCESS_COOKIE, svc.tokens.access(user, csrf), max_age=s.access_token_minutes * 60,
                        path="/", httponly=True, secure=secure, samesite="strict")
    response.set_cookie(REFRESH_COOKIE, svc.tokens.refresh(user, csrf), max_age=s.refresh_token_days * 86400,
                        path=REFRESH_PATH, httponly=True, secure=secure, samesite="strict")
    # readable by the page's script, which echoes it in X-CSRF-Token (double submit)
    response.set_cookie(CSRF_COOKIE, csrf, max_age=s.refresh_token_days * 86400, path="/", httponly=False,
                        secure=secure, samesite="strict")


def clear_login_cookies(response: Response, svc: Services) -> None:
    secure = svc.settings.cookie_secure
    response.delete_cookie(ACCESS_COOKIE, path="/", secure=secure, httponly=True, samesite="strict")
    response.delete_cookie(REFRESH_COOKIE, path=REFRESH_PATH, secure=secure, httponly=True, samesite="strict")
    response.delete_cookie(CSRF_COOKIE, path="/", secure=secure, samesite="strict")


def check_csrf(request: Request, claims: dict) -> None:
    if request.method in SAFE_METHODS:
        return
    sent = request.headers.get(CSRF_HEADER)
    if not sent or sent != claims.get("csrf"):
        raise ApiError(403, "missing or wrong CSRF token (header X-CSRF-Token)", code="csrf_failed")


# ------------------------------------------------------------------ principals
def current_user(request: Request, svc: Svc) -> Principal:
    """The logged-in user (access-token cookie). State-changing requests must also carry the CSRF
    token of the login in X-CSRF-Token."""
    try:
        user, claims = svc.tokens.verify(request.cookies.get(ACCESS_COOKIE), "access", svc.users)
    except AuthError as exc:
        raise ApiError(401, str(exc)) from None
    check_csrf(request, claims)
    return user


CurrentUser = Annotated[Principal, Depends(current_user)]


def admin_user(user: CurrentUser) -> Principal:
    if not user.is_admin:
        raise ApiError(403, "only administrators may do this")
    return user


AdminUser = Annotated[Principal, Depends(admin_user)]


def owner_filter(user: Principal) -> int | None:
    """The owner id a query is restricted to: the user's own id, or None (all) for admins."""
    return None if user.is_admin else user.id


def client_id(x_client_id: Annotated[str | None, Header(alias=CLIENT_HEADER, description=(
        "random id of the browser tab (kept in sessionStorage)"))] = None) -> str:
    if not x_client_id or not CLIENT_ID.match(x_client_id):
        raise ApiError(400, "header X-Client-Id is missing or invalid (8-40 characters A-Z a-z 0-9 _ -)")
    return x_client_id


ClientId = Annotated[str, Depends(client_id)]


# ------------------------------------------------------------------- pagination
@dataclass
class PageParams:
    limit: int
    offset: int

    def wrap(self, items: list, total: int) -> dict:
        return {"items": items, "total": total, "limit": self.limit, "offset": self.offset}


def page_params(svc: Svc, limit: Annotated[int, Query(ge=1, description="items per page")] = 50,
                offset: Annotated[int, Query(ge=0, description="items to skip")] = 0) -> PageParams:
    if limit > svc.settings.page_max:
        raise ApiError(422, f"limit: at most {svc.settings.page_max}", code="validation_error",
                       details=[{"field": "query.limit", "message": f"at most {svc.settings.page_max}"}])
    return PageParams(limit, offset)


Page = Annotated[PageParams, Depends(page_params)]
