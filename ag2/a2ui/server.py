# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

from starlette.applications import Starlette

from ag2.agent import Agent

from ._runtime import _A2UIRuntime
from ._types import A2UIVersion, JsonSchema
from .actions import A2UIAction, collect_action_declarations, collect_server_actions
from .dispatch import _A2UITurnCore
from .transports.base import A2UITransport

if TYPE_CHECKING:
    from starlette.types import Receive, Scope, Send


class A2UIServer:
    """Serve a plain :class:`~ag2.Agent` over A2UI as an ASGI app.

    Hold a normal ``Agent``, declare any clickable ``actions``, pick exactly one
    ``transport`` (the deployment's wire encoding), and configure A2UI with flat
    kwargs. The instance **is** the ASGI app — run it directly with
    ``uvicorn mymodule:server``, mount it (``Mount("/x", app=server)``), or drive
    it with ``TestClient(server)``. The server is stateless — clients send the
    full conversation each turn.

    Example::

        from ag2 import Agent
        from ag2.a2ui import A2UIServer, a2ui_action
        from ag2.a2ui.transports import AgUiTransport


        @a2ui_action(description="Schedule all posts for the given time")
        def schedule_posts(time: str) -> str: ...


        agent = Agent(name="ui", config=...)  # plain agent, no A2UI tools
        server = A2UIServer(agent, actions=[schedule_posts], transport=AgUiTransport())
        # uvicorn mymodule:server
    """

    __slots__ = ("_agent", "_core", "_runtime", "_starlette", "_transport")

    def __init__(
        self,
        agent: Agent,
        *,
        transport: A2UITransport,
        actions: Sequence[A2UIAction] = (),
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
            agent: A plain ``ag2.Agent`` (no A2UI tools needed).
            transport: The single wire transport for this deployment (e.g.
                ``AgUiTransport()`` or ``RestTransport(encoding="sse")``).
                Required — one deployment serves one frontend over one transport.
            actions: Clickable buttons (``@a2ui_action``). Each is declared to the
                LLM so it can render the button; a click runs the action's handler
                on the server **without** invoking the agent. A button the LLM
                draws with no registered action still works — its click is
                rewritten into a generic prompt so the agent can react.
            protocol_version: A2UI protocol version: "v0.9" (default), "v0.9.1", or "v1.0".
            custom_catalog: A custom catalog extending the basic catalog (path or
                dict). Must include a ``$id`` used as the catalogId.
            custom_catalog_rules: Plain-text rules for the custom catalog components.
            include_schema_in_prompt: Include the full JSON schema in the prompt
                (better validation, more tokens).
            include_rules_in_prompt: Include catalog rules in the prompt.
            validate_responses: Validate A2UI output against the schema and retry on failure.
                When False, A2UI is still extracted and published (the block is stripped
                from the prose), but the model's UI is trusted as-is with no schema check
                or retry — the client validates and degrades gracefully.
            validation_retries: Additional retries when validation fails (total
                attempts = ``validation_retries + 1``). 0 disables retry.
            system_message: Custom prefix system message. If None, uses the
                default A2UI system message.
        """
        self._agent = agent
        self._transport = transport
        action_objs = tuple(actions)
        self._runtime = _A2UIRuntime(
            # Buttons are declared to the LLM so it can render them.
            actions=collect_action_declarations(action_objs),
            protocol_version=protocol_version,
            custom_catalog=custom_catalog,
            custom_catalog_rules=custom_catalog_rules,
            include_schema_in_prompt=include_schema_in_prompt,
            include_rules_in_prompt=include_rules_in_prompt,
            validate_responses=validate_responses,
            validation_retries=validation_retries,
            system_message=system_message,
        )
        # A click on a registered action runs its handler on the server, never
        # invoking the agent.
        self._core = _A2UITurnCore(
            agent,
            self._runtime,
            collect_server_actions(action_objs),
        )
        # The instance IS the app (no ``.app``/``build_app()``): build the
        # Starlette app once from the transport's routes and delegate to it.
        self._starlette = Starlette(routes=transport.routes(self._core))

    @property
    def agent(self) -> Agent:
        return self._agent

    async def __call__(self, scope: "Scope", receive: "Receive", send: "Send") -> None:
        """ASGI entrypoint — delegate to the transport-built Starlette app."""
        await self._starlette(scope, receive, send)


__all__ = ("A2UIServer",)
