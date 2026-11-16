# 0016 Authentication: JWT in HttpOnly cookies, CSRF by double submit

Status: accepted (phase 4)

## Context
The API v1 needs logins with two roles (annotator, admin), must work behind several API replicas
without server-side session state, and the page must not keep a token where a script injected into
it could read it (no `localStorage`, no `sessionStorage`, no JavaScript-readable cookie holding a
token).

## Decision
- **Users** are rows of `users`; passwords are argon2id hashes (`argon2-cffi`, its default
  parameters; a hash with older parameters is re-hashed at the next login). Unknown user names
  cost the same hashing work as wrong passwords, and both answer 401 `invalid_credentials` with the
  same message. The first admin is made with `python -m scripts.create_user <name> --role admin`;
  admins add further users through `POST /api/v1/users`.
- **Tokens** are HS256 JWTs signed with `LAL_JWT_SECRET` (at least 32 characters; the same value on
  every API replica). Claims: `sub` (user id), `typ` (`access` | `refresh`), `ver` (the user's
  `token_version`), `csrf`, `iat`, `exp`, `jti`. Access tokens live `LAL_ACCESS_TOKEN_MINUTES`
  (15), refresh tokens `LAL_REFRESH_TOKEN_DAYS` (7). Every request reads the user row, so a deleted
  user or changed role takes effect at once; bumping `token_version` (logout, password change)
  revokes every token issued before. The `none` algorithm and other keys are refused.
- **Where tokens live**: only in cookies the page's JavaScript cannot read:
  `lal_access` (HttpOnly, `SameSite=Strict`, `Path=/`), `lal_refresh` (HttpOnly,
  `SameSite=Strict`, `Path=/api/v1/auth`, so it is sent only to login / refresh / logout). Both
  are `Secure` unless `LAL_COOKIE_SECURE=false` (plain-http local work). No response body contains
  a token. The page keeps only a random per-tab client id in `sessionStorage` (ADR 0017), which is
  not a credential.
- **CSRF**: `SameSite=Strict` already keeps the cookies off cross-site requests; in addition every
  state-changing request (any method other than GET / HEAD / OPTIONS) must carry the header
  `X-CSRF-Token` equal to the `csrf` claim of the access token (for `/auth/refresh`: of the refresh
  token). The value is random per login and is also set as the cookie `lal_csrf` (not HttpOnly,
  `SameSite=Strict`), from which the page copies it into the header (double submit). A cross-site
  form or script can neither read that cookie nor set custom headers without a CORS preflight,
  which the API does not allow. Because the header must match a claim inside the signed token, a
  `lal_csrf` cookie planted by a sibling domain does not help an attacker. Login itself needs a
  JSON body (no form post) and sets new cookies, so login CSRF can only log a user into the
  attacker's account, which the role checks and ownership make harmless here.
- **Expiry in the page**: `web/static/client.js` answers a 401 by calling `/auth/refresh` once and
  repeating the request; if that fails it goes to `/login?next=<path>` (only same-site paths are
  followed after login).

## Review after phase 4: cookie flags, lifetimes, logout, failed logins

| cookie | content | HttpOnly | Secure | SameSite | Path | Max-Age |
|---|---|---|---|---|---|---|
| `lal_access` | access JWT | yes | yes* | Strict | `/` | `LAL_ACCESS_TOKEN_MINUTES` (15 min) |
| `lal_refresh` | refresh JWT | yes | yes* | Strict | `/api/v1/auth` | `LAL_REFRESH_TOKEN_DAYS` (7 days) |
| `lal_csrf` | CSRF value (not a credential alone) | no | yes* | Strict | `/` | as the refresh token |

\* `Secure` is set by the API itself (`LAL_COOKIE_SECURE`, default true), independent of the scheme
the API process sees. Behind the TLS gateway of phase 5 the browser talks HTTPS, so the flag is
right although nginx speaks plain HTTP to the API; `LAL_COOKIE_SECURE=false` is only for local
work over plain http. The `__Host-` cookie prefix is not used (it would forbid `Path=/api/v1/auth`
for the refresh cookie).

- **Lifetimes.** An access token is valid 15 minutes, a refresh token 7 days (both checked against
  `exp` in the token, not only by the cookie's Max-Age). `/auth/refresh` issues a new access token
  *and* a new refresh token, so the 7 days are a sliding window: a user who uses the page at least
  once a week stays logged in until logout. Refresh tokens are not single-use: an older refresh
  token stays valid until its `exp` or until `token_version` changes; a stolen one works for up to
  7 days unless the user logs out or the password changes. Single-use rotation would need a token
  table (not done).
- **Logout** (`POST /api/v1/auth/logout`, needs the CSRF header) increments the user's
  `token_version`, which ends every access and refresh token of that user on every device, and
  clears the three cookies with the same attributes they were set with. With an expired access
  token the page refreshes first and then logs out (`client.js`); if that fails too the cookies are
  left in the browser but are already useless. The change takes effect on the next request to any
  replica, because every request reads the user row.
- **Repeated failed logins: nothing in the API limits them yet.** There is no per-account lockout,
  no per-IP counter and no delay; the only cost is the argon2id verification, which takes the same
  time for unknown user names. Measured once on the Windows 11 development machine (argon2-cffi
  default parameters m=65536 KiB, t=3, p=4; 20 wrong-password verifications in one process):
  median 35.2 ms, min 30.7 ms, max 41.2 ms. The limit is planned at the gateway in phase 5 (nginx `limit_req` on
  `/api/v1/auth/login` per client IP). A per-account lockout was not chosen: it lets anyone lock a
  known user out; if needed, a per-account delay with a counter in Redis would be the next step.
- These facts are pinned by
  `tests/test_api/test_auth.py::test_cookie_lifetimes_refresh_window_and_no_failed_login_limit`.

## Consequences
- Revocation is per user (all devices), not per token; a per-device logout would need a token
  table.
- `LAL_JWT_SECRET` must be rotated by restarting all replicas with the new value; every user then
  logs in again.
- Clients other than the browser log in the same way and must keep the cookies and send
  `X-CSRF-Token` (see `/api/v1/docs`).
