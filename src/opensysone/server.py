"""Jev-compatible HTTP API over one Engine. One GPU, one worker, a bounded queue."""
from __future__ import annotations

import asyncio
import logging
import secrets
import uuid
from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .calib.calibrators import Calibrator
from .calib.store import OutcomeStore, UnknownDecision
from .engine import MODEL_ALIASES, MODEL_VERSION, RAW_KEY
from .schema import SchemaError
from .wire import BulkRequest, SystemOneRequest, error_response, validation_error

RELEASE_DATE = "2026-09-19"
logger = logging.getLogger(__name__)


class FeedbackRequest(BaseModel):
    request_id: str
    question: str
    label: bool | int | str


def _public(out: Any) -> Any:
    if isinstance(out, dict) and RAW_KEY in out:
        out = {k: v for k, v in out.items() if k != RAW_KEY}
    return out


def create_app(engine: Any, *, api_key: str | None = None, max_queue: int = 64,
               store: OutcomeStore | None = None, calibration_dir: str | None = None) -> FastAPI:
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

    def _no_store():
        return error_response(404, "not_found_error",
                              "outcome store is off; start the server with --outcomes PATH")

    def _served(rid: str, req: SystemOneRequest, out: Any) -> Any:
        if not isinstance(out, dict):
            return out
        if store is not None:
            store.record(rid, req, out)
        pub = _public(out)
        pub["request_id"] = rid
        return pub

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
    async def systemone(req: SystemOneRequest, request: Request):
        if req.model not in MODEL_ALIASES:
            return error_response(
                404, "not_found_error", f"Model {req.model!r} not found. Available: opensysone-latest.")
        try:
            out = await run_serial(engine.decide, req)
        except SchemaError as e:
            return validation_error(e.loc, e.msg)
        return _served(request.state.request_id, req, out)

    @app.post("/v1/bulk")
    async def bulk(req: BulkRequest, request: Request):
        try:
            out = await run_serial(engine.decide_bulk, req.items)
        except SchemaError as e:
            return validation_error(e.loc, e.msg)
        if isinstance(out, JSONResponse):
            return out
        rid = request.state.request_id
        return {"items": [_served(f"{rid}-{i}", item, o) for i, (item, o) in enumerate(zip(req.items, out))]}

    @app.post("/v1/feedback")
    async def feedback(body: FeedbackRequest):
        if store is None:
            return _no_store()
        try:
            result = store.add_outcome(body.request_id, body.question, body.label)
        except UnknownDecision as e:
            return error_response(404, "not_found_error", str(e.args[0]))
        except ValueError as e:
            return validation_error(["body", "label"], str(e), body.label)
        return {"ok": True, **result}

    @app.get("/v1/audit")
    async def audit(window: int = Query(1000, ge=1, le=100_000)):
        if store is None:
            return _no_store()
        return store.audit(window)

    @app.get("/v1/queue")
    async def queue(threshold: float = Query(0.6, ge=0.0, le=1.0), limit: int = Query(50, ge=1, le=500),
                    offset: int = Query(0, ge=0)):
        if store is None:
            return _no_store()
        return {"items": store.queue(threshold, limit, offset)}

    @app.post("/v1/calibration/reload")
    async def reload_calibration():
        if calibration_dir is None:
            return error_response(404, "not_found_error",
                                  "no calibration directory; start the server with --calibration DIR")
        cal = await asyncio.to_thread(Calibrator.load, calibration_dir)
        async with state.lock:
            engine.set_calibrator(cal)
        return {"calibration": cal.sha}

    return app
