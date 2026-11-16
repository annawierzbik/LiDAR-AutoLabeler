"""One error shape for every response that is not 2xx (docs/adr/0018):

    {"error": {"code": "not_found", "message": "unknown tab 1a2b", "details": null}}

`code` is stable and meant for programs; `message` is for people; `details` is optional structured
data (validation errors list the fields). Unexpected exceptions are logged and answered with 500
"internal_error" without internals.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger(__name__)

CODES = {400: "bad_request", 401: "unauthorized", 403: "forbidden", 404: "not_found", 405: "method_not_allowed",
         409: "conflict", 410: "gone", 413: "payload_too_large", 415: "unsupported_media_type",
         422: "validation_error", 500: "internal_error", 503: "unavailable"}


class ApiError(Exception):
    def __init__(self, status: int, message: str, code: str | None = None, details=None) -> None:
        super().__init__(message)
        self.status, self.message, self.code, self.details = status, message, code or CODES.get(status, "error"), details


class ErrorBody(BaseModel):
    code: str
    message: str
    details: object | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


def body(status: int, code: str, message: str, details=None, headers=None) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message, "details": details}}, status_code=status,
                        headers=headers)


def not_found(what: str, ident) -> ApiError:
    return ApiError(404, f"unknown {what} {ident}")


def install(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError):
        return body(exc.status, exc.code, exc.message, exc.details)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        message = exc.detail if isinstance(exc.detail, str) else CODES.get(exc.status_code, "error")
        return body(exc.status_code, CODES.get(exc.status_code, "error"), message, headers=getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        details = [{"field": ".".join(str(p) for p in e.get("loc", ())), "message": e.get("msg", "")}
                   for e in exc.errors()]
        first = details[0] if details else {"field": "", "message": "invalid request"}
        return body(422, "validation_error", f"{first['field']}: {first['message']}", details)

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception):
        log.exception("unhandled error in %s %s", request.method, request.url.path)
        return body(500, "internal_error", "internal error")


# for OpenAPI: the error responses every endpoint may give
COMMON = {401: {"model": ErrorResponse, "description": "not logged in, or the session expired"},
          422: {"model": ErrorResponse, "description": "invalid parameters"}}


def responses(*statuses: int) -> dict:
    text = {400: "invalid request", 403: "needs another role, or the CSRF header is missing",
            404: "no such resource, or it belongs to another user", 409: "conflicting state (see message)",
            410: "the stored result has expired", 413: "the upload is too large"}
    return {**COMMON, **{s: {"model": ErrorResponse, "description": text[s]} for s in statuses}}
