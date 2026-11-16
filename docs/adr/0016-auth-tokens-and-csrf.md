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

## Consequences
- Revocation is per user (all devices), not per token; a per-device logout would need a token
  table.
- `LAL_JWT_SECRET` must be rotated by restarting all replicas with the new value; every user then
  logs in again.
- Clients other than the browser log in the same way and must keep the cookies and send
  `X-CSRF-Token` (see `/api/v1/docs`).
