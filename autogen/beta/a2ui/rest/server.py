# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import TYPE_CHECKING

from autogen.beta.agent import Agent

from .._runtime import _A2UIRuntime
from .._types import A2UIVersion, JsonSchema
from .asgi import build_jsonl_app, build_sse_app

if TYPE_CHECKING:
    from starlette.applications import Starlette


class A2UIServer:
    """Serve a plain :class:`~autogen.beta.Agent` over HTTP as canonical A2UI.

    A transport-neutral REST/SSE adapter that depends only on Starlette (no
    ``ag-ui`` / ``a2a-sdk``). It mirrors ``autogen.beta.a2a.A2AServer`` and
    ``autogen.beta.ag_ui.AGUIStream``: hold a normal ``Agent`` and configure
    A2UI with flat kwargs, then call a ``build_*`` method for a ready-to-serve
    Starlette ASGI app. Bring your own ``uvicorn`` (or any ASGI server).

    Clickable buttons (``@a2ui_action``) live on the agent's tool list and are
    discovered automatically — there is nothing extra to register here.

    The server is **stateless** — clients send the full conversation on every
    request (see :func:`autogen.beta.a2ui.rest.parse_request` for the JSON body
    contract). Each turn runs on a fresh stream; the A2UI messages the validation
    middleware emits are streamed out as the canonical wire format.

    A2UI's wire is transport-agnostic, so two encodings are offered:

    - :meth:`build_sse_app` — Server-Sent Events (``text/event-stream``).
    - :meth:`build_jsonl_app` — canonical A2UI NDJSON (``application/x-ndjson``).

    Example::

        from autogen.beta import Agent
        from autogen.beta.a2ui import a2ui_action
        from autogen.beta.a2ui.rest import A2UIServer


        @a2ui_action(description="Schedule all posts for the given time")
        def schedule_posts(time: str) -> str: ...


        agent = Agent(name="ui", config=..., tools=[schedule_posts])
        app = A2UIServer(agent, protocol_version="v0.9").build_sse_app()
    """

    __slots__ = ("_agent", "_runtime")

    def __init__(
        self,
        agent: Agent,
        *,
        protocol_version: A2UIVersion = "v0.9",
        custom_catalog: "str | os.PathLike[str] | JsonSchema | None" = None,
        custom_catalog_rules: str | None = None,
        include_schema_in_prompt: bool = True,
        include_rules_in_prompt: bool = True,
        validate_responses: bool = True,
        validation_retries: int = 1,
        system_message: str | None = None,
    ) -> None:
        """Wrap ``agent`` and configure A2UI.

        Args:
            agent: A plain ``autogen.beta.Agent``. ``@a2ui_action`` tools passed
                to ``Agent(tools=[...])`` are discovered as clickable buttons.
            protocol_version: A2UI protocol version: "v0.9" (default), "v0.9.1", or "v1.0".
            custom_catalog: A custom catalog extending the basic catalog (path or
                dict). Must include a ``$id`` used as the catalogId.
            custom_catalog_rules: Plain-text rules for the custom catalog components.
            include_schema_in_prompt: Include the full JSON schema in the prompt
                (better validation, more tokens).
            include_rules_in_prompt: Include catalog rules in the prompt.
            validate_responses: Validate A2UI output against the schema and retry
                on failure. Requires ``ag2[a2ui]`` (jsonschema).
            validation_retries: Additional retries when validation fails (total
                attempts = ``validation_retries + 1``). 0 disables retry.
            system_message: Custom prefix system message. If None, uses the
                default A2UI system message.
        """
        self._agent = agent
        self._runtime = _A2UIRuntime(
            agent,
            protocol_version=protocol_version,
            custom_catalog=custom_catalog,
            custom_catalog_rules=custom_catalog_rules,
            include_schema_in_prompt=include_schema_in_prompt,
            include_rules_in_prompt=include_rules_in_prompt,
            validate_responses=validate_responses,
            validation_retries=validation_retries,
            system_message=system_message,
        )

    @property
    def agent(self) -> Agent:
        return self._agent

    def build_sse_app(self, *, path: str = "/a2ui") -> "Starlette":
        """Starlette ASGI app serving the turn as SSE at ``path`` (POST)."""
        return build_sse_app(self._agent, self._runtime, path=path)

    def build_jsonl_app(self, *, path: str = "/a2ui") -> "Starlette":
        """Starlette ASGI app serving the turn as A2UI NDJSON at ``path`` (POST)."""
        return build_jsonl_app(self._agent, self._runtime, path=path)


__all__ = ("A2UIServer",)
