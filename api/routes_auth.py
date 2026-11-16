"""Login, token refresh, logout, the current user, and user management (admins)."""
from __future__ import annotations

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, Field

from api.deps import (
    REFRESH_COOKIE,
    AdminUser,
    CurrentUser,
    Page,
    Svc,
    check_csrf,
    clear_login_cookies,
    set_login_cookies,
)
from api.errors import ApiError, responses
from infra.auth import ROLES, AuthError, UserExists, new_csrf_token

router = APIRouter(prefix="/api/v1")


class Login(BaseModel):
    username: str = Field(max_length=100)
    password: str = Field(max_length=1024)


class UserView(BaseModel):
    id: int
    username: str
    role: str


class NewUser(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=8, max_length=1024)
    role: str = Field("annotator", description=f"one of: {', '.join(ROLES)}")


@router.post("/auth/login", tags=["auth"], response_model=UserView, responses=responses())
def login(body: Login, response: Response, svc: Svc):
    """Check the password and set the session cookies: `lal_access` (HttpOnly, 15 min by default),
    `lal_refresh` (HttpOnly, path /api/v1/auth) and `lal_csrf` (readable by the page, to be sent
    back in X-CSRF-Token with every POST)."""
    try:
        user = svc.users.authenticate(body.username, body.password)
    except AuthError as exc:
        raise ApiError(401, str(exc), code="invalid_credentials") from None
    set_login_cookies(response, svc, user, new_csrf_token())
    return user.view()


@router.post("/auth/refresh", tags=["auth"], response_model=UserView, responses=responses(403))
def refresh(request: Request, response: Response, svc: Svc):
    """A new access token (and refresh token) from the refresh cookie. Needs X-CSRF-Token."""
    try:
        user, claims = svc.tokens.verify(request.cookies.get(REFRESH_COOKIE), "refresh", svc.users)
    except AuthError as exc:
        raise ApiError(401, str(exc)) from None
    check_csrf(request, claims)
    set_login_cookies(response, svc, user, claims["csrf"])
    return user.view()


@router.post("/auth/logout", tags=["auth"], responses=responses(403))
def logout(response: Response, user: CurrentUser, svc: Svc):
    """End every login of this user (all tokens issued so far stop working) and clear the cookies."""
    svc.users.revoke_tokens(user.id)
    clear_login_cookies(response, svc)
    return {"ok": True}


@router.get("/auth/me", tags=["auth"], response_model=UserView, responses=responses())
def me(user: CurrentUser):
    return user.view()


@router.get("/users", tags=["users"], responses=responses(403))
def list_users(page: Page, _: AdminUser,
               svc: Svc):
    """All users (admins only)."""
    return page.wrap(*svc.users.page(page.limit, page.offset))


@router.post("/users", tags=["users"], status_code=201, response_model=UserView, responses=responses(403, 409))
def create_user(body: NewUser, _: AdminUser, svc: Svc):
    """Create a user (admins only)."""
    try:
        return svc.users.create(body.username, body.password, body.role).view()
    except UserExists as exc:
        raise ApiError(409, str(exc)) from None
    except ValueError as exc:
        raise ApiError(400, str(exc)) from None
