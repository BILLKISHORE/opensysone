"""Jev-compatible HTTP API over one Engine. One GPU, one worker, a bounded queue."""
from __future__ import annotations

import asyncio
import logging
import secrets
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .engine import MODEL_ALIASES, MODEL_VERSION
from .schema import SchemaError
from .wire import BulkRequest, SystemOneRequest, error_response, validation_error

RELEASE_DATE = "2026-09-19"
logger = logging.getLogger(__name__)


def create_app(engine: Any, *, api_key: str | None = None, max_queue: int = 64) -> FastAPI:
    app = FastAPI(title="opensysone", version=MODEL_VERSION)
    state = app.state
    state.engine = engine
    state.lock = asyncio.Lock()
    state.waiting = 0
    state.served = 0

    @app.middleware("http")
    async def request_id_and_auth(request: Request, call_next):
        request.state.request_id = uuid.uuid4().hex
        if api_key and request.url.path.startswith("/v1/"):
            header = request.headers.get("authorization", "")
            token = header[7:] if header.startswith("Bearer ") else ""
            if not token or not secrets.compare_digest(token, api_key):
                return error_response(401, "authentication_error", "invalid or missing API key")
        response = await call_next(request)
        response.headers["x-request-id"] = request.state.request_id
        return response

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        logger.exception("request %s failed", getattr(request.state, "request_id", "?"))
        return error_response(500, "api_error", f"{type(exc).__name__}: {exc}")

    async def run_serial(fn, *args):
        if state.waiting >= max_queue:
            return error_response(529, "overloaded_error", "queue is full; retry later", {"retry-after": "1"})
        state.waiting += 1
        try:
            async with state.lock:
                return await asyncio.to_thread(fn, *args)
        finally:
            state.waiting -= 1
            state.served += 1

    @app.get("/health")
    async def health():
        return {
            "ok": bool(engine.ready),
            "model": engine.model_id,
            "busy": state.lock.locked(),
            "queue_depth": state.waiting,
            "requests_served": state.served,
        }

    @app.get("/v1/models")
    async def models():
        return {"models": [
            {"name": "opensysone-latest",
             "description": f"Alias for the newest opensysone release. Currently {MODEL_VERSION}.",
             "release_date": RELEASE_DATE},
            {"name": MODEL_VERSION, "description": f"{MODEL_VERSION} on {engine.model_id}",
             "release_date": RELEASE_DATE},
        ]}

    @app.post("/v1/systemone")
    async def systemone(req: SystemOneRequest):
        if req.model not in MODEL_ALIASES:
            return error_response(
                404, "not_found_error", f"Model {req.model!r} not found. Available: opensysone-latest.")
        try:
            return await run_serial(engine.decide, req)
        except SchemaError as e:
            return validation_error(e.loc, e.msg)

    @app.post("/v1/bulk")
    async def bulk(req: BulkRequest):
        try:
            out = await run_serial(engine.decide_bulk, req.items)
        except SchemaError as e:
            return validation_error(e.loc, e.msg)
        if isinstance(out, JSONResponse):
            return out
        return {"items": out}

    return app
