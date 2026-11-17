# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Drive one A2UI turn through the public serving seam and read back what a user observes.

``serve_turn`` scripts the model with ``TestConfig``, posts the request to an
``A2UIServer`` over HTTP (JSONL encoding), and returns the frames on the wire,
the calls the framework made to the model (with any retry feedback), and the
``A2UIValidationFailedEvent``\\s, ``A2UIClientEvent``\\s and final ``ModelResponse``\\s an observer saw.
"""

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from ag2 import Agent
from ag2.a2ui import A2UIClientEvent, A2UIServer, A2UIValidationFailedEvent
from ag2.a2ui.transports import RestTransport
from ag2.events import ModelRequest, ModelResponse, TextInput
from ag2.observers import observer
from ag2.testing import ModelCall, TestConfig, TrackingConfig

WIRE_VERSION = {"v0.9": "v0.9", "v0.9.1": "v0.9.1", "v1.0": "1.0"}
BASIC_CATALOG = {
    "v0.9": "https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json",
    "v0.9.1": "https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json",
    "v1.0": "https://a2ui.org/specification/v1_0/catalogs/basic/catalog.json",
}


def renderer_capabilities(*catalog_ids: str) -> dict[str, Any]:
    """The request fields of a v1.0 renderer that supports ``catalog_ids`` (the basic catalog when none are given).

    A v1.0 request without them offers no A2UI ("omitting them withdraws A2UI for that turn").
    """
    supported = list(catalog_ids) or [BASIC_CATALOG["v1.0"]]
    return {"a2uiRendererCapabilities": {"1.0": {"supportedCatalogIds": supported}}}


def from_renderer(body: Mapping[str, Any], *catalog_ids: str) -> dict[str, Any]:
    """``body`` as a v1.0 renderer supporting ``catalog_ids`` sends it; capabilities already in ``body`` win."""
    return {**renderer_capabilities(*catalog_ids), **body}


def catalog_id_of(catalog: "Mapping[str, Any] | str | os.PathLike[str]") -> str:
    """The id a custom catalog (a dict or a JSON file) is served under."""
    document = catalog if isinstance(catalog, Mapping) else json.loads(Path(catalog).read_text())
    return str(document.get("catalogId", document.get("$id")))


def served_catalogs(server_kwargs: Mapping[str, Any]) -> list[str]:
    """Every catalog a v1.0 ``A2UIServer`` built with ``server_kwargs`` serves: the basic one and any custom one."""
    custom = server_kwargs.get("custom_catalog")
    return [BASIC_CATALOG["v1.0"], *([catalog_id_of(custom)] if custom is not None else [])]


@dataclass
class Turn:
    frames: list[dict[str, Any]]
    model_calls: list[ModelCall]
    failures: list[A2UIValidationFailedEvent] = field(default_factory=list)
    client_events: list[A2UIClientEvent] = field(default_factory=list)
    responses: list[ModelResponse] = field(default_factory=list)
    """The model responses the agent emitted, after the A2UI middleware rewrote them."""

    @property
    def ui(self) -> list[dict[str, Any]]:
        """The A2UI messages streamed to the client."""
        return [frame for frame in self.frames if "text" not in frame and "error" not in frame]

    @property
    def prose(self) -> list[str]:
        return [frame["text"] for frame in self.frames if "text" in frame]

    @property
    def feedback(self) -> list[str]:
        """The retry prompts the framework sent the model, one per retry (a tool round-trip is not one)."""
        prompts: list[str] = []
        for call in self.model_calls[1:]:
            last = call.messages[-1]
            if isinstance(last, ModelRequest):
                prompts.extend(part.content for part in last.parts if isinstance(part, TextInput))
        return prompts

    @property
    def errors(self) -> list[list[str]]:
        """The validation errors each retry prompt listed (its ``- `` lines), one list per retry."""
        return [[line for line in prompt.splitlines() if line.startswith("- ")] for prompt in self.feedback]


def a2ui_reply(messages: list[dict[str, Any]], prose: str = "Here.") -> str:
    """A model reply carrying ``messages`` in an ``<a2ui-json>`` block."""
    return f"{prose}\n<a2ui-json>\n{json.dumps(messages)}\n</a2ui-json>"


async def serve_turn(
    *replies: "str | ModelResponse",
    body: dict[str, Any] | None = None,
    tools: Sequence[Any] = (),
    with_capabilities: bool = True,
    **server_kwargs: Any,
) -> Turn:
    """Run one turn whose model answers with ``replies`` in order; ``server_kwargs`` configure ``A2UIServer``.

    A v1.0 request comes from a renderer that supports every catalog the server serves, unless ``body`` carries its
    own capabilities or ``with_capabilities`` is false.
    """
    tracking = TrackingConfig(TestConfig(*replies))
    failures: list[A2UIValidationFailedEvent] = []
    client_events: list[A2UIClientEvent] = []
    responses: list[ModelResponse] = []

    async def record_failure(event: A2UIValidationFailedEvent) -> None:
        failures.append(event)

    async def record_client_event(event: A2UIClientEvent) -> None:
        client_events.append(event)

    async def record_response(event: ModelResponse) -> None:
        responses.append(event)

    agent = Agent(
        "ui",
        config=tracking,
        tools=tools,
        observers=[
            observer(A2UIValidationFailedEvent, record_failure),
            observer(A2UIClientEvent, record_client_event),
            observer(ModelResponse, record_response),
        ],
    )
    server = A2UIServer(agent, transport=RestTransport(encoding="jsonl"), **server_kwargs)
    request = body if body is not None else {"messages": [{"role": "user", "content": "show ui"}]}
    if with_capabilities and server_kwargs.get("protocol_version") == "v1.0":
        request = from_renderer(request, *served_catalogs(server_kwargs))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server), base_url="http://a2ui.test") as client:
        response = await client.post("/a2ui", json=request)
    assert response.status_code == 200, response.text
    frames = [json.loads(line) for line in response.text.splitlines() if line]
    return Turn(
        frames=frames,
        model_calls=list(tracking.calls),
        failures=failures,
        client_events=client_events,
        responses=responses,
    )
