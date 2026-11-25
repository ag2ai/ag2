# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""An in-memory stand-in for the kosa8 Cloud REST API, served through
``httpx.MockTransport`` so the tests run offline."""

import asyncio
import base64
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

Exec = Callable[[dict[str, Any]], httpx.Response]


def exec_answer(stdout: str = "", stderr: str = "", exit_code: int | None = 0) -> httpx.Response:
    return httpx.Response(200, json={"exit": exit_code, "stdout": stdout, "stderr": stderr})


def stream_answer(*frames: dict[str, Any]) -> httpx.Response:
    body = "".join(json.dumps(frame) + "\n" for frame in frames)
    return httpx.Response(200, content=body.encode(), headers={"Content-Type": "application/x-ndjson"})


def out_frame(text: str, stream: str = "stdout") -> dict[str, Any]:
    return {"stream": stream, "data": base64.b64encode(text.encode()).decode()}


def error_answer(status: int, code: str, message: str = "", **extra: Any) -> httpx.Response:
    return httpx.Response(status, json={"error": code, "message": message or code, **extra})


@dataclass
class FakeCloud:
    """Records every request and answers like kosa8 Cloud.

    ``exec_results`` answer the exec calls in order (the default is an empty
    success); ``create_state`` is the state a create answers with.
    """

    exec_results: list[httpx.Response] = field(default_factory=list)
    create_state: str = "running"
    states_after_create: list[str] = field(default_factory=list)
    create_error: httpx.Response | None = None
    delete_errors: list[httpx.Response] = field(default_factory=list)
    files_error: httpx.Response | None = None
    labels_error: httpx.Response | None = None
    requests: list[httpx.Request] = field(default_factory=list)
    created: int = 0
    deleted: list[str] = field(default_factory=list)
    files: dict[str, bytes] = field(default_factory=dict)
    dirs: list[str] = field(default_factory=list)
    polled: asyncio.Event = field(default_factory=asyncio.Event)

    def client(self, base_url: str = "https://api.kosa8.com") -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": "Bearer k8c_live_test"},  # pragma: allowlist secret
            transport=httpx.MockTransport(self.handle),
        )

    def calls(self, method: str, suffix: str) -> list[httpx.Request]:
        return [r for r in self.requests if r.method == method and r.url.path.endswith(suffix)]

    def exec_bodies(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.calls("POST", "/exec")]

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if request.method == "POST" and path == "/v1/sandboxes":
            if self.create_error is not None:
                return self.create_error
            self.created += 1
            return httpx.Response(
                201,
                json={"sandbox": {"id": f"sbx_{self.created}", "state": self.create_state, "template": "python"}},
            )
        sandbox_id = path.split("/")[3]
        if request.method == "GET" and path == f"/v1/sandboxes/{sandbox_id}":
            self.polled.set()
            state = self.states_after_create.pop(0) if self.states_after_create else "running"
            return httpx.Response(200, json={"sandbox": {"id": sandbox_id, "state": state, "last_error": "boom"}})
        if request.method == "DELETE" and path == f"/v1/sandboxes/{sandbox_id}":
            if self.delete_errors:
                return self.delete_errors.pop(0)
            self.deleted.append(sandbox_id)
            return httpx.Response(200, json={"deleted": True})
        if path.endswith("/exec"):
            return self.exec_results.pop(0) if self.exec_results else exec_answer()
        if path.endswith("/labels"):
            if self.labels_error is not None:
                return self.labels_error
            return httpx.Response(200, json=json.loads(request.content))
        if path.endswith("/egress"):
            return httpx.Response(200, json={"egress": json.loads(request.content)})
        if path.endswith("/files/mkdir"):
            if self.files_error is not None:
                return self.files_error
            self.dirs.append(json.loads(request.content)["path"])
            return httpx.Response(200, json={"file": {}})
        if request.method == "PUT" and path.endswith("/files"):
            if self.files_error is not None:
                return self.files_error
            self.files[request.url.params["path"]] = request.content
            return httpx.Response(200, json={"file": {"path": request.url.params["path"]}})
        return httpx.Response(404, json={"error": "not_found", "message": f"no route {request.method} {path}"})
