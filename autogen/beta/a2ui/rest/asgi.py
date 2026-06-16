# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Starlette ASGI apps that serve canonical A2UI over HTTP.

Two wire encodings share one core (:func:`stream_turn`):

- **SSE** (``text/event-stream``): the conversational prose arrives as an
  ``event: text`` frame, each A2UI message as a default (unnamed) ``data:``
  frame, and the turn closes with ``event: done``. Suited to browser
  ``EventSource`` clients and reconnection.
- **NDJSON** (``application/x-ndjson``): A2UI JSON Lines — an optional leading
  prose ``{"text": ...}`` line (AG2 framing, not itself an A2UI message),
  followed by canonical A2UI messages, one per line. The stream ends at EOF; a
  failed turn emits a final ``{"error": ...}`` line. Suited to generic A2UI
  clients that consume the native JSONL wire.

Importing this module requires Starlette; ``rest/__init__.py`` turns a missing
install into a clear ``missing_additional_dependency`` hint.
"""

import functools
import json
import logging
from collections.abc import AsyncIterator

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from ..agent import A2UIAgent
from ..serialize import to_jsonl
from .dispatch import A2UIFrame, A2UIProseFrame, stream_turn
from .request import A2UIServerRequest, parse_request

logger = logging.getLogger(__name__)

_SSE_MEDIA_TYPE = "text/event-stream"
_JSONL_MEDIA_TYPE = "application/x-ndjson"


def build_sse_app(agent: A2UIAgent, *, path: str = "/a2ui") -> Starlette:
    """Build a Starlette app that streams the turn as Server-Sent Events."""
    endpoint = functools.partial(_sse_endpoint, agent)
    return Starlette(routes=[Route(path, endpoint, methods=["POST"])])


def build_jsonl_app(agent: A2UIAgent, *, path: str = "/a2ui") -> Starlette:
    """Build a Starlette app that streams the turn as canonical A2UI NDJSON."""
    endpoint = functools.partial(_jsonl_endpoint, agent)
    return Starlette(routes=[Route(path, endpoint, methods=["POST"])])


async def _read_request(agent: A2UIAgent, request: Request) -> "A2UIServerRequest | Response":
    """Read+parse the body, or return a 400 ``Response`` on failure.

    Body reading can fail independently of parsing — e.g. ``ClientDisconnect``
    mid-upload — so it gets its own guard; a parse error (bad shape) is a 400
    with the validation message.
    """
    try:
        body = await request.body()
    except Exception:  # noqa: BLE001 - transport/disconnect errors → 400, not 500
        return JSONResponse({"error": "could not read request body"}, status_code=400)
    try:
        return parse_request(
            body,
            resolve_action=agent.get_action,
            version_key=agent.schema_manager.version_string,
        )
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)


async def _sse_endpoint(agent: A2UIAgent, request: Request) -> Response:
    parsed = await _read_request(agent, request)
    if isinstance(parsed, Response):
        return parsed
    return StreamingResponse(_sse_frames(agent, parsed), media_type=_SSE_MEDIA_TYPE)


async def _jsonl_endpoint(agent: A2UIAgent, request: Request) -> Response:
    parsed = await _read_request(agent, request)
    if isinstance(parsed, Response):
        return parsed
    return StreamingResponse(_jsonl_frames(agent, parsed), media_type=_JSONL_MEDIA_TYPE)


async def _sse_frames(agent: A2UIAgent, parsed: A2UIServerRequest) -> AsyncIterator[str]:
    # The turn runs lazily as the response streams, so a mid-turn failure can't
    # change the already-sent 200 status. Surface it as an ``event: error`` frame
    # and log it, rather than tearing down the connection silently.
    try:
        async for frame in stream_turn(agent, parsed):
            yield _encode_sse(frame)
    except Exception as e:
        logger.exception("A2UI SSE turn failed")
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
    else:
        yield "event: done\ndata: {}\n\n"


async def _jsonl_frames(agent: A2UIAgent, parsed: A2UIServerRequest) -> AsyncIterator[str]:
    try:
        async for frame in stream_turn(agent, parsed):
            yield _encode_jsonl(frame)
    except Exception as e:
        logger.exception("A2UI NDJSON turn failed")
        yield json.dumps({"error": str(e)}) + "\n"


def _encode_sse(frame: A2UIFrame) -> str:
    if isinstance(frame, A2UIProseFrame):
        return f"event: text\ndata: {json.dumps({'text': frame.text})}\n\n"
    # One A2UI message per frame, serialized via the canonical JSONL serializer
    # so the wire bytes stay identical across transports.
    return f"data: {to_jsonl((frame.message,))}\n\n"


def _encode_jsonl(frame: A2UIFrame) -> str:
    if isinstance(frame, A2UIProseFrame):
        return json.dumps({"text": frame.text}) + "\n"
    return to_jsonl((frame.message,)) + "\n"


__all__ = ("build_jsonl_app", "build_sse_app")
