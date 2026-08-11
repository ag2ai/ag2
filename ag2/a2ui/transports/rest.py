# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""REST transport: serve a turn as canonical A2UI over HTTP in one of two wire
encodings — SSE (``text/event-stream``) or NDJSON (``application/x-ndjson``).
Both share the same minimal-JSON request contract (``messages``, ``variables``,
``a2ui``, ``a2uiClientCapabilities``); they differ only in how frames are framed.
Importing this module requires Starlette.
"""

import functools
import json
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Literal

from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from ..dispatch import A2UIFrame, A2UIProseFrame
from ..request import A2UIServerRequest, parse_request
from ..serialize import to_jsonl

if TYPE_CHECKING:
    from ..dispatch import _A2UITurnCore

logger = logging.getLogger(__name__)

_SSE_MEDIA_TYPE = "text/event-stream"
_JSONL_MEDIA_TYPE = "application/x-ndjson"

RestEncoding = Literal["sse", "jsonl"]


class RestTransport:
    """Serve the turn over HTTP as either SSE or NDJSON.

    Args:
        encoding: ``"sse"`` (default, ``text/event-stream``) or ``"jsonl"``
            (``application/x-ndjson``). Same JSON request contract either way.
        path: The POST route path. Defaults to ``"/a2ui"``.
    """

    __slots__ = ("_encoding", "_media_type", "_path")

    def __init__(self, encoding: RestEncoding = "sse", *, path: str = "/a2ui") -> None:
        if encoding not in ("sse", "jsonl"):
            raise ValueError(f"encoding must be 'sse' or 'jsonl', got {encoding!r}")
        self._encoding = encoding
        self._media_type = _SSE_MEDIA_TYPE if encoding == "sse" else _JSONL_MEDIA_TYPE
        self._path = path

    def routes(self, core: "_A2UITurnCore") -> list[Route]:
        endpoint = functools.partial(_endpoint, self, core)
        return [Route(self._path, endpoint, methods=["POST"])]


async def _endpoint(transport: RestTransport, core: "_A2UITurnCore", request: Request) -> Response:
    parsed = await _read_request(core, request)
    if isinstance(parsed, Response):
        return parsed
    return StreamingResponse(_frames(transport, core, parsed), media_type=transport._media_type)


async def _read_request(core: "_A2UITurnCore", request: Request) -> "A2UIServerRequest | Response":
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
            resolve_action=core.runtime.get_action,
            version_key=core.runtime.version_string,
        )
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)


async def _frames(transport: RestTransport, core: "_A2UITurnCore", parsed: A2UIServerRequest) -> AsyncIterator[str]:
    # The turn runs lazily as the response streams, so a mid-turn failure can't
    # change the already-sent 200 status. Surface it as an error frame and log
    # it, rather than tearing down the connection silently.
    sse = transport._encoding == "sse"
    try:
        async for frame in core.run_turn(parsed):
            yield _encode_sse(frame) if sse else _encode_jsonl(frame)
    except Exception as e:
        logger.exception("A2UI %s turn failed", transport._encoding)
        if sse:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
        else:
            yield json.dumps({"error": str(e)}) + "\n"
    else:
        if sse:
            yield "event: done\ndata: {}\n\n"


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


__all__ = ("RestTransport",)
