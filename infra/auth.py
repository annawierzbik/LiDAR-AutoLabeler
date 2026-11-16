"""Users, password hashes and tokens of the API v1 (docs/adr/0016).

- Passwords: argon2id (argon2-cffi defaults); a hash made with older parameters is re-hashed at the
  next successful login.
- Tokens: HS256 JWTs signed with LAL_JWT_SECRET. An access token (short-lived) authorises requests;
  a refresh token (long-lived, typ "refresh") only buys a new access token. Both carry the user's
  `token_version`; bumping it (logout, password change, role change) revokes every token issued so
  far. Every request reads the user row, so a deleted user or a changed role takes effect at once.
- Roles: "annotator" and "admin".
"""
from __future__ import annotations

import secrets
import time
import uuid
from dataclasses import dataclass

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from sqlalchemy import func, insert, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

from infra.db import User

ROLES = ("annotator", "admin")
ALGORITHM = "HS256"
_hasher = PasswordHasher()


class AuthError(Exception):
    """Wrong credentials or an invalid / expired / revoked token."""


class UserExists(Exception):
    pass


@dataclass(frozen=True)
class Principal:
    id: int
    username: str
    role: str
    token_version: int

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def view(self) -> dict:
        return {"id": self.id, "username": self.username, "role": self.role}


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def check_password_rules(password: str) -> None:
    if len(password) < 8:
        raise ValueError("a password needs at least 8 characters")


# ---------------------------------------------------------------------- users
class UserStore:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    @staticmethod
    def _principal(row) -> Principal:
        return Principal(row.id, row.username, row.role, row.token_version)

    def create(self, username: str, password: str, role: str = "annotator") -> Principal:
        username = username.strip()
        if not username:
            raise ValueError("the user name is empty")
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r} (use one of: {', '.join(ROLES)})")
        check_password_rules(password)
        try:
            with self.engine.begin() as c:
                row = c.execute(insert(User).values(username=username, password_hash=hash_password(password),
                                                    role=role, token_version=0)
                                .returning(User.id, User.username, User.role, User.token_version)).first()
        except IntegrityError:
            raise UserExists(f"user {username!r} already exists") from None
        return self._principal(row)

    def get(self, user_id: int) -> Principal | None:
        with self.engine.connect() as c:
            row = c.execute(select(User.id, User.username, User.role, User.token_version)
                            .where(User.id == user_id)).first()
        return None if row is None else self._principal(row)

    def by_name(self, username: str) -> Principal | None:
        with self.engine.connect() as c:
            row = c.execute(select(User.id, User.username, User.role, User.token_version)
                            .where(User.username == username.strip())).first()
        return None if row is None else self._principal(row)

    def authenticate(self, username: str, password: str) -> Principal:
        with self.engine.connect() as c:
            row = c.execute(select(User.id, User.username, User.role, User.token_version, User.password_hash)
                            .where(User.username == username.strip())).first()
        if row is None or not row.password_hash:
            _hasher_dummy_verify(password)          # same work for unknown users (no user enumeration by timing)
            raise AuthError("wrong user name or password")
        try:
            _hasher.verify(row.password_hash, password)
        except (VerificationError, InvalidHashError):
            raise AuthError("wrong user name or password") from None
        if _hasher.check_needs_rehash(row.password_hash):
            with self.engine.begin() as c:
                c.execute(update(User).where(User.id == row.id).values(password_hash=hash_password(password)))
        return self._principal(row)

    def revoke_tokens(self, user_id: int) -> None:
        with self.engine.begin() as c:
            c.execute(update(User).where(User.id == user_id).values(token_version=User.token_version + 1))

    def set_password(self, user_id: int, password: str) -> None:
        check_password_rules(password)
        with self.engine.begin() as c:
            c.execute(update(User).where(User.id == user_id)
                      .values(password_hash=hash_password(password), token_version=User.token_version + 1))

    def page(self, limit: int, offset: int) -> tuple[list[dict], int]:
        with self.engine.connect() as c:
            total = c.execute(select(func.count()).select_from(User)).scalar()
            rows = c.execute(select(User.id, User.username, User.role, User.created_at)
                             .order_by(User.id).limit(limit).offset(offset)).all()
        return [{"id": r.id, "username": r.username, "role": r.role, "created_at": r.created_at.isoformat()}
                for r in rows], total

    def count_admins(self) -> int:
        with self.engine.connect() as c:
            return c.execute(select(func.count()).select_from(User).where(User.role == "admin")).scalar()


_DUMMY_HASH = None


def _hasher_dummy_verify(password: str) -> None:
    global _DUMMY_HASH
    if _DUMMY_HASH is None:
        _DUMMY_HASH = hash_password("not-a-real-password")
    try:
        _hasher.verify(_DUMMY_HASH, password)
    except VerificationError:
        pass


# --------------------------------------------------------------------- tokens
def new_csrf_token() -> str:
    return secrets.token_urlsafe(32)


class Tokens:
    """Access and refresh JWTs. Both carry `csrf`, the double-submit token of the login (docs/adr/0016):
    a state-changing request must send it in X-CSRF-Token, and it must equal the access token's."""

    def __init__(self, secret: str, access_minutes: float, refresh_days: float, clock=time.time) -> None:
        if not secret or len(secret) < 32:
            raise ValueError("LAL_JWT_SECRET must be set to a random string of at least 32 characters")
        self.secret, self.clock = secret, clock
        self.access_seconds = int(access_minutes * 60)
        self.refresh_seconds = int(refresh_days * 86400)

    def _make(self, user: Principal, typ: str, seconds: int, csrf: str) -> str:
        now = int(self.clock())
        return jwt.encode({"sub": str(user.id), "typ": typ, "ver": user.token_version, "csrf": csrf, "iat": now,
                           "exp": now + seconds, "jti": uuid.uuid4().hex}, self.secret, algorithm=ALGORITHM)

    def access(self, user: Principal, csrf: str) -> str:
        return self._make(user, "access", self.access_seconds, csrf)

    def refresh(self, user: Principal, csrf: str) -> str:
        return self._make(user, "refresh", self.refresh_seconds, csrf)

    def verify(self, token: str | None, typ: str, users: UserStore) -> tuple[Principal, dict]:
        """(user, claims) of a valid, unexpired, unrevoked token of type `typ`; AuthError otherwise."""
        if not token:
            raise AuthError("not logged in")
        try:
            claims = jwt.decode(token, self.secret, algorithms=[ALGORITHM],      # expiry: checked below
                                options={"require": ["sub", "typ", "ver", "csrf", "exp"], "verify_exp": False})
        except jwt.PyJWTError:
            raise AuthError("invalid token") from None
        if int(self.clock()) >= claims["exp"]:
            raise AuthError("the session has expired; log in again")
        if claims["typ"] != typ:
            raise AuthError("invalid token")
        user = users.get(int(claims["sub"]))
        if user is None or user.token_version != claims["ver"]:
            raise AuthError("the session was ended; log in again")
        return user, claims
