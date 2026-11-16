// Shared by the pages: calls to /api/v1 with the login cookies (HttpOnly, never visible here), the
// CSRF header (double submit: the value of the lal_csrf cookie) and this browser tab's client id
// (sessionStorage: one per tab, survives a reload). No token is ever stored by the page (docs/adr/0016).

export class ApiError extends Error {
  constructor(status, code, message, details) { super(message); this.status = status; this.code = code; this.details = details; }
}

function newId() {
  const a = new Uint8Array(12); crypto.getRandomValues(a);
  return 'tab-' + Array.from(a, b => b.toString(16).padStart(2, '0')).join('');
}
export const CLIENT_ID = (() => {
  try {
    let id = sessionStorage.getItem('lal:client');
    if (!id) { id = newId(); sessionStorage.setItem('lal:client', id); }
    return id;
  } catch (e) { return newId(); }
})();

export function csrfToken() {
  const m = document.cookie.match(/(?:^|;\s*)lal_csrf=([^;]*)/);
  return m ? decodeURIComponent(m[1]) : '';
}
export const loggedIn = () => csrfToken() !== '';

export function toLogin() {
  location.href = '/login?next=' + encodeURIComponent(location.pathname + location.search);
}

let refreshing = null;
async function refresh() {
  refreshing = refreshing || fetch('/api/v1/auth/refresh', { method: 'POST', headers: { 'X-CSRF-Token': csrfToken() } })
    .then(r => r.ok).finally(() => { setTimeout(() => { refreshing = null; }, 0); });
  return refreshing;
}

// api(path) -> GET; api(path, { json }) -> POST with a JSON body; api(path, { body, type }) -> POST raw.
// Resolves to the Response; rejects with ApiError (status, code, message) for every non-2xx answer.
// An expired access token is renewed once with the refresh cookie; without a valid login the page
// goes to /login.
export async function api(path, opts = {}) {
  const post = opts.method === 'POST' || opts.json !== undefined || opts.body !== undefined;
  const send = () => fetch(path, {
    method: opts.method || (post ? 'POST' : 'GET'),
    headers: {
      'X-Client-Id': CLIENT_ID,
      ...(post ? { 'X-CSRF-Token': csrfToken() } : {}),
      ...(opts.json !== undefined ? { 'Content-Type': 'application/json' } : {}),
      ...(opts.type ? { 'Content-Type': opts.type } : {}),
    },
    body: opts.json !== undefined ? JSON.stringify(opts.json) : opts.body,
  });
  if (!loggedIn()) { toLogin(); throw new ApiError(401, 'unauthorized', 'Please log in.'); }
  let r = await send();
  if (r.status === 401) {
    if (await refresh()) r = await send();
    else { toLogin(); throw new ApiError(401, 'unauthorized', 'Please log in again.'); }
  }
  if (!r.ok) {
    let code = 'error', message = r.statusText, details = null;
    try { ({ code, message, details } = (await r.json()).error); } catch (e) { /* not our error body */ }
    throw new ApiError(r.status, code, message, details);
  }
  return r;
}
export const getJson = async (path) => (await api(path)).json();
export const postJson = async (path, json = {}) => (await api(path, { json })).json();

export async function logout() {
  try { await api('/api/v1/auth/logout', { json: {} }); } catch (e) { /* cookies are cleared or already gone */ }
  location.href = '/login';
}
