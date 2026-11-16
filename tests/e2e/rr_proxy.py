"""A minimal round-robin HTTP gateway for the statelessness test: every request goes to the next
backend in turn, so consecutive requests of one page hit different API processes.

    python -m tests.e2e.rr_proxy <port> http://127.0.0.1:<a> http://127.0.0.1:<b>

GET /__proxy_stats answers {"hits": [requests per backend]}. The gateway keeps no cookies of its
own: Cookie and Set-Cookie headers pass through unchanged.
"""
from __future__ import annotations

import itertools
import sys
from http.cookiejar import CookieJar, DefaultCookiePolicy

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

HOP = {"connection", "keep-alive", "transfer-encoding", "content-length", "content-encoding", "host", "upgrade"}


def make_app(backends: list[str]) -> Starlette:
    order = itertools.cycle(range(len(backends)))
    hits = [0] * len(backends)
    no_cookies = httpx.Cookies(CookieJar(policy=DefaultCookiePolicy(allowed_domains=[])))
    client = httpx.AsyncClient(timeout=120, cookies=no_cookies)

    async def forward(request: Request) -> Response:
        if request.url.path == "/__proxy_stats":
            return JSONResponse({"hits": hits})
        i = next(order)
        hits[i] += 1
        url = backends[i] + request.url.path + (f"?{request.url.query}" if request.url.query else "")
        headers = [(k, v) for k, v in request.headers.items() if k.lower() not in HOP]
        r = await client.request(request.method, url, headers=headers, content=await request.body())
        out = Response(r.content, status_code=r.status_code)
        for k, v in r.headers.multi_items():
            if k.lower() not in HOP:
                out.headers.append(k, v)
        return out

    methods = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]
    return Starlette(routes=[Route("/{path:path}", forward, methods=methods)])


if __name__ == "__main__":
    uvicorn.run(make_app(sys.argv[2:]), host="127.0.0.1", port=int(sys.argv[1]), log_level="warning")
