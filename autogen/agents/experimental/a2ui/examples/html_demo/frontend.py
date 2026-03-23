# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Frontend server that connects to the A2A social previewer agent.

This server:
1. Serves the HTML frontend at /
2. Proxies chat requests to the A2A agent at http://localhost:9000
3. Receives A2UI DataParts from the A2A response
4. Streams them to the browser as SSE events (same format as AG-UI)

The frontend HTML is reused from the ag_ui_demo — it renders the same
ACTIVITY_SNAPSHOT events regardless of whether they came via AG-UI or A2A.

Run a2a_agent.py first, then run this.

Usage:
    python frontend.py
    # Open http://localhost:8456 in your browser
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import StreamingResponse

from autogen.a2a import A2aRemoteAgent, HttpxClientFactory
from autogen.agents.experimental.a2ui.a2a_helpers import create_a2ui_part

load_dotenv()

A2A_AGENT_URL = os.getenv("A2A_AGENT_URL", "http://localhost:9000")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve images (same images as ag_ui_demo)
ag_ui_images = Path(__file__).parent.parent / "ag_ui_demo" / "images"
if ag_ui_images.exists():
    app.mount("/images", StaticFiles(directory=ag_ui_images), name="images")


async def sse_stream(messages: list[dict[str, Any]], a2ui_action: dict[str, Any] | None = None) -> AsyncIterator[str]:
    """Connect to the A2A agent and stream responses as SSE events."""
    run_id = str(uuid4())
    thread_id = "a2a-demo"

    # Emit RUN_STARTED
    yield f"data: {json.dumps({'type': 'RUN_STARTED', 'threadId': thread_id, 'runId': run_id})}\n\n"

    try:
        client_factory = HttpxClientFactory(timeout=httpx.Timeout(60.0))
        remote = A2aRemoteAgent(
            url=A2A_AGENT_URL,
            name="a2a_client",
            client=client_factory,
        )

        msg_list = [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages]

        # Build extra_parts for A2UI action DataPart (spec-compliant)
        extra = None
        if a2ui_action:
            extra = [create_a2ui_part(a2ui_action)]

        _, reply = await remote.a_generate_remote_reply(
            messages=msg_list,
            sender=None,
            extra_parts=extra,
        )

        if reply and isinstance(reply, dict):
            # Check if we got A2UI DataPart (messages key with version objects)
            a2ui_messages = reply.get("messages", [])
            has_a2ui = a2ui_messages and any(isinstance(m, dict) and "version" in m for m in a2ui_messages)

            if has_a2ui:
                yield f"data: {json.dumps({'type': 'ACTIVITY_SNAPSHOT', 'messageId': str(uuid4()), 'activityType': 'a2ui-surface', 'content': {'operations': a2ui_messages}, 'replace': True})}\n\n"

            # Emit text content if present
            text_content = reply.get("content", "")
            if text_content:
                msg_id = str(uuid4())
                yield f"data: {json.dumps({'type': 'TEXT_MESSAGE_START', 'messageId': msg_id})}\n\n"
                yield f"data: {json.dumps({'type': 'TEXT_MESSAGE_CONTENT', 'messageId': msg_id, 'delta': text_content})}\n\n"
                yield f"data: {json.dumps({'type': 'TEXT_MESSAGE_END', 'messageId': msg_id})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'RUN_ERROR', 'message': str(e)})}\n\n"

    # Emit RUN_FINISHED
    yield f"data: {json.dumps({'type': 'RUN_FINISHED', 'threadId': thread_id, 'runId': run_id})}\n\n"


@app.post("/chat/")
async def chat(request: Request) -> StreamingResponse:
    body = await request.json()
    messages = body.get("messages", [])
    a2ui_action = body.get("a2uiAction")
    return StreamingResponse(
        sse_stream(messages, a2ui_action=a2ui_action),
        media_type="text/event-stream",
    )


@app.get("/styles.css")
async def serve_styles() -> FileResponse:
    return FileResponse(Path(__file__).parent / "styles.css", media_type="text/css")


@app.get("/")
async def serve_frontend() -> FileResponse:
    return FileResponse(Path(__file__).parent / "frontend.html")


if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("  A2A Frontend Proxy")
    print(f"  A2A Agent: {A2A_AGENT_URL}")
    print("  Frontend: http://localhost:8456")
    print("=" * 60)
    print("\n  Make sure a2a_agent.py is running first!")
    uvicorn.run(app, host="0.0.0.0", port=8456)
