"""Login, cookies, CSRF, refresh, logout (token revocation), roles, the create_user CLI."""
import pytest
from fastapi.testclient import TestClient

from tests.test_api.conftest import PASSWORD

pytestmark = pytest.mark.postgres


def set_cookies(response) -> dict:
    out = {}
    for h in response.headers.get_list("set-cookie"):
        name = h.split("=", 1)[0]
        out[name] = h.lower()
    return out


def test_login_sets_httponly_cookies_and_no_token_in_the_body(api):
    api.user("anna")
    c = TestClient(api.app)
    r = c.post("/api/v1/auth/login", json={"username": "anna", "password": PASSWORD})
    assert r.status_code == 200 and r.json() == {"id": 1, "username": "anna", "role": "annotator"}
    cookies = set_cookies(r)
    assert set(cookies) == {"lal_access", "lal_refresh", "lal_csrf"}
    for name in ("lal_access", "lal_refresh"):
        assert "httponly" in cookies[name] and "samesite=strict" in cookies[name]
    assert "httponly" not in cookies["lal_csrf"] and "samesite=strict" in cookies["lal_csrf"]
    assert "path=/api/v1/auth" in cookies["lal_refresh"]
    assert "token" not in r.text.lower()


def test_secure_cookies_by_default(api, monkeypatch):
    from infra.settings import get_settings

    monkeypatch.delenv("LAL_COOKIE_SECURE")
    get_settings.cache_clear()
    a = api.restart()
    a.user("anna")
    r = TestClient(a.app).post("/api/v1/auth/login", json={"username": "anna", "password": PASSWORD})
    assert all("secure" in v for v in set_cookies(r).values())


def test_wrong_password_and_unknown_user_look_the_same(api):
    api.user("anna")
    c = TestClient(api.app)
    a = c.post("/api/v1/auth/login", json={"username": "anna", "password": "wrong password"})
    b = c.post("/api/v1/auth/login", json={"username": "nobody", "password": "wrong password"})
    assert a.status_code == b.status_code == 401
    assert a.json() == b.json() == {"error": {"code": "invalid_credentials", "message": "wrong user name or password",
                                              "details": None}}


def test_requests_need_a_login(api):
    c = TestClient(api.app)
    for method, path in (("get", "/api/v1/auth/me"), ("get", "/api/v1/sequences"), ("get", "/api/v1/jobs"),
                         ("get", "/api/v1/sessions"), ("post", "/api/v1/tabs")):
        r = getattr(c, method)(path)
        assert r.status_code == 401 and r.json()["error"]["code"] == "unauthorized", path
    assert c.get("/").status_code == 200 and c.get("/login").status_code == 200    # pages are public, data is not


def test_post_without_or_with_a_wrong_csrf_header_is_refused(api):
    api.user("anna")
    c = api.login("anna")
    body = {"sequence_id": 1}
    r = c.http.post("/api/v1/tabs", json=body, headers={"X-Client-Id": c.client_id})
    assert r.status_code == 403 and r.json()["error"]["code"] == "csrf_failed"
    r = c.http.post("/api/v1/tabs", json=body, headers={"X-Client-Id": c.client_id, "X-CSRF-Token": "forged"})
    assert r.status_code == 403
    assert c.post("/api/v1/tabs", body).status_code == 404               # the right token: on to the next check
    assert c.http.get("/api/v1/auth/me").status_code == 200              # GET needs no CSRF header


def test_csrf_token_of_another_login_does_not_work(api):
    api.user("anna")
    a, b = api.login("anna"), api.login("anna")
    r = a.http.post("/api/v1/tabs", json={"sequence_id": 1},
                    headers={"X-Client-Id": a.client_id, "X-CSRF-Token": b.http.cookies["lal_csrf"]})
    assert r.status_code == 403


def test_refresh_gives_a_new_access_token(api):
    api.user("anna")
    c = api.login("anna")
    assert c.http.post("/api/v1/auth/refresh").status_code == 403         # CSRF header needed
    r = c.post("/api/v1/auth/refresh")
    assert r.status_code == 200 and "lal_access" in set_cookies(r)
    c.http.cookies.delete("lal_access")
    assert c.get("/api/v1/auth/me").status_code == 401
    assert c.post("/api/v1/auth/refresh").status_code == 200
    assert c.get("/api/v1/auth/me").status_code == 200


def test_logout_revokes_every_token(api):
    api.user("anna")
    c = api.login("anna")
    old_access, old_refresh = c.http.cookies["lal_access"], c.http.cookies["lal_refresh"]
    other_device = api.login("anna")
    assert c.post("/api/v1/auth/logout").json() == {"ok": True}
    stale = TestClient(api.app, cookies={"lal_access": old_access})
    assert stale.get("/api/v1/auth/me").status_code == 401
    assert other_device.get("/api/v1/auth/me").status_code == 401
    r = TestClient(api.app).post("/api/v1/auth/refresh", cookies={"lal_refresh": old_refresh},
                                 headers={"X-CSRF-Token": c.http.cookies.get("lal_csrf") or "x"})
    assert r.status_code == 401


def test_tokens_expire_and_cannot_be_swapped_or_forged():
    import jwt

    from infra.auth import AuthError, Principal, Tokens

    user = Principal(7, "anna", "annotator", 0)

    class Users:
        def get(self, uid):
            return user if uid == 7 else None

    now = [1_000_000.0]
    t = Tokens("s" * 40, access_minutes=15, refresh_days=1, clock=lambda: now[0])
    access, refresh = t.access(user, "c"), t.refresh(user, "c")
    assert t.verify(access, "access", Users())[0] == user
    with pytest.raises(AuthError):
        t.verify(refresh, "access", Users())                    # a refresh token is not an access token
    with pytest.raises(AuthError):
        t.verify(access, "refresh", Users())
    forged = jwt.encode({"sub": "7", "typ": "access", "ver": 0, "csrf": "c", "exp": 2_000_000}, "x" * 40,
                        algorithm="HS256")
    with pytest.raises(AuthError):
        t.verify(forged, "access", Users())
    none_alg = jwt.encode({"sub": "7", "typ": "access", "ver": 0, "csrf": "c", "exp": 2_000_000}, None,
                          algorithm="none")
    with pytest.raises(AuthError):
        t.verify(none_alg, "access", Users())
    now[0] += 15 * 60
    with pytest.raises(AuthError, match="expired"):
        t.verify(access, "access", Users())
    assert t.verify(refresh, "refresh", Users())[0] == user
    with pytest.raises(ValueError):
        Tokens("short", 15, 1)


def test_roles(api):
    api.user("boss", "admin")
    api.user("anna")
    admin, anna = api.login("boss"), api.login("anna")
    assert anna.get("/api/v1/users").status_code == 403
    assert anna.post("/api/v1/users", {"username": "x", "password": PASSWORD}).status_code == 403
    r = admin.post("/api/v1/users", {"username": "bob", "password": PASSWORD})
    assert r.status_code == 201 and r.json()["role"] == "annotator"
    assert admin.post("/api/v1/users", {"username": "bob", "password": PASSWORD}).status_code == 409
    assert admin.post("/api/v1/users", {"username": "eve", "password": PASSWORD, "role": "root"}).status_code == 400
    assert admin.post("/api/v1/users", {"username": "eve", "password": "short"}).status_code == 422
    page = admin.get("/api/v1/users").json()
    assert page["total"] == 3 and [u["username"] for u in page["items"]] == ["boss", "anna", "bob"]
    assert "password_hash" not in str(page)


def test_passwords_are_argon2_hashes(api, pg_engine):
    from sqlalchemy import text

    api.user("anna")
    with pg_engine.connect() as c:
        h = c.execute(text("SELECT password_hash FROM users WHERE username = 'anna'")).scalar()
    assert h.startswith("$argon2id$") and PASSWORD not in h


def test_create_user_cli(api, monkeypatch, capsys):
    import io

    from scripts.create_user import main

    monkeypatch.setattr("sys.stdin", io.StringIO(PASSWORD + "\n"))
    assert main(["root", "--role", "admin", "--password-stdin"]) == 0
    assert "created user root" in capsys.readouterr().out
    assert api.login("root").user["role"] == "admin"
    monkeypatch.setattr("sys.stdin", io.StringIO(PASSWORD + "\n"))
    assert main(["root", "--password-stdin"]) == 1                        # exists already
    monkeypatch.setattr("sys.stdin", io.StringIO("short\n"))
    assert main(["other", "--password-stdin"]) == 1
