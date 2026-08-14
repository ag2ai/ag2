# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Config classes for ACP-backed agents.

:class:`ACPConfig` is everything about driving an agent — workspace, model,
policies, timeouts, tool exposure — plus what it takes to *launch* one as a local
subprocess. :class:`~ag2.acp.remote.ACPRemoteConfig` extends it with what it
takes to reach one that is already running elsewhere. Both implement the
:class:`~ag2.config.config.ModelConfig` protocol; ``create()`` returns an
``ACPClient`` that drives the agent over the Agent Client Protocol.

``ClaudeCodeConfig``, ``CodexConfig``, ``OpenCodeConfig`` and ``KiloCodeConfig``
are thin subclasses of :class:`ACPConfig` carrying the launch defaults for the
Claude Code, Codex, OpenCode and Kilo Code ACP adapters respectively.
"""

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from typing_extensions import Self

from .tool_gateway import GatewayAddress

if TYPE_CHECKING:
    from asyncio.subprocess import Process
    from collections.abc import Callable
    from contextlib import AbstractAsyncContextManager

    import acp
    from acp.core import ClientSideConnection

    from ag2.config.client import LLMClient
    from ag2.context import StreamId

    from .session import ACPSession

    # Opens the ACP connection for a session. ``ACPConfig`` spawns a subprocess,
    # ``ACPRemoteConfig`` dials a URL, and tests inject an in-process double
    # (see ``acp.testing``) — the process handle is optional precisely so a
    # connection that has no process can satisfy this.
    ConnectHook = Callable[["acp.Client"], "AbstractAsyncContextManager[tuple[ClientSideConnection, Process | None]]"]


def _dispatch_kwargs(client: "acp.Client") -> dict[str, Any]:
    """Connection arguments that make this client's notifications waitable.

    Read off the bridge the connection hook already receives, so no hook signature
    has to carry them (see :mod:`~ag2.acp.dispatch`). A client that is not an AG2
    bridge — a double standing in for one — gets the SDK's own defaults.

    ``ACPBridge`` is imported inside the function, as ``acp`` itself is below: the
    bridge pulls in the SDK, and importing ``ag2.acp.config`` must not.
    """
    from .bridge import ACPBridge

    return client.state.updates.connection_kwargs() if isinstance(client, ACPBridge) else {}


PermissionPolicy = Literal["ask", "auto", "deny"]

# Deliberately two-valued, unlike ``PermissionPolicy``: a permission request
# carries an allow option a client can pick blind, whereas an arbitrary
# elicitation form has no answer AG2 could invent without fabricating data on
# the user's behalf. So there is no ``"auto"`` — only ask a human, or decline.
ElicitationPolicy = Literal["ask", "decline"]


@dataclass(slots=True)
class ACPConfig:
    """Drive a CLI coding agent over ACP, launching it as a local subprocess.

    Also the base every other ACP config extends:
    :class:`~ag2.acp.remote.ACPRemoteConfig` swaps the subprocess for a URL, and
    the four presets below carry a known launch command. Nothing above the
    connection hook knows which of them it is talking to.

    Field order is published API — ``ACPConfig(["claude-agent-acp"])`` is how the
    docs have always spelled it — which is also why this class, not a shared
    base, is where the fields live: an inherited field necessarily comes first in
    a generated ``__init__``, and ``command`` has to stay there. A new field goes
    after ``expose_tools``, never between two a caller may have passed
    positionally.

    Attributes:
        command: Executable + base args launching the agent in ACP mode,
            e.g. ``["claude-agent-acp"]``. The first element is the executable.
        env: Extra environment variables for the subprocess. The subprocess does
            NOT inherit the full parent environment: only a small whitelist
            (``HOME``, ``LOGNAME``, ``PATH``, ``SHELL``, ``TERM``, ``USER``) is
            inherited, merged with this mapping. So API-key auth must be passed
            here explicitly (a shell ``export`` of the key is not inherited); a
            disk login under ``$HOME`` (e.g. ``~/.claude``) works without it.
        cwd: Workspace root passed to ``session/new``. Local to the AG2 process
            even when the agent is remote: ACP's ``fs/*`` and terminal methods
            are requests *from* the agent *to* the client, and AG2 is the
            client, so a remote agent works on the files in front of you.
        model: Agent model selection. Applied at session start via ACP
            ``session/set_config_option`` when the agent advertises a model
            picker in ``session/new`` (Claude Code, OpenCode and Kilo Code all
            do); the value must be one of the agent's advertised model ids.
            ``None`` keeps the agent's default. When the agent has no model
            option the value is response metadata only.
        permission_policy: How to answer ``session/request_permission``:
            ``"ask"`` routes to the agent's ``hitl_hook``/``context.input``,
            ``"auto"`` allows, ``"deny"`` rejects.
        elicitation_policy: How to answer ``elicitation/create`` — the agent
            asking the *user* a question (a form, or a URL to complete an
            out-of-band flow such as OAuth). ``"ask"`` (default) advertises the
            elicitation capability and routes the question to the agent's
            ``hitl_hook``/``context.input``, the same channel permissions use.
            ``"decline"`` does not advertise the capability at all, so a
            conforming agent never asks; one that asks anyway is declined.
            There is no ``"auto"``: see :data:`ElicitationPolicy`.
        fs_root: Root for mediated ``fs/*`` access (defaults to ``cwd``).
        allow_terminal: Whether to advertise the ACP terminal capability.
        additional_directories: Extra ACP workspace roots.
        startup_timeout: Seconds to allow for the tool gateway's HTTP server
            to start when tools are exposed.
        turn_timeout: Per-prompt-turn timeout in seconds (``None`` = no limit).
        cancel_timeout: Grace period (seconds) after a timed-out turn signals
            ``session/cancel`` for the agent to return the in-flight prompt. If
            the agent does not respond within it, the connection is torn down —
            hard-stopping the subprocess where there is one.
        expose_tools: When ``True`` (default), the agent's locally-executable
            tools are served to the CLI agent over an in-process HTTP MCP
            server, and ``MCPServerTool`` entries are handed to it directly
            via ACP ``mcp_servers``. ``False`` disables both. The set of
            servers is fixed when the ACP session is created (first turn):
            function tools added or removed on later turns hot-update the
            gateway, but changing the ``MCPServerTool`` set — or introducing
            function tools when the first turn had none — raises.
    """

    command: list[str] = field(default_factory=list)
    cwd: str = "."
    env: dict[str, str] | None = None
    model: str | None = None
    permission_policy: PermissionPolicy = "ask"
    fs_root: str | None = None
    allow_terminal: bool = True
    additional_directories: list[str] = field(default_factory=list)
    startup_timeout: float = 30.0
    turn_timeout: float | None = None
    cancel_timeout: float = 5.0
    expose_tools: bool = True
    elicitation_policy: ElicitationPolicy = "ask"

    # Run-scoped live sessions, keyed by stream id. Not part of identity and not
    # carried by ``copy()`` (a copy is a distinct config with its own sessions).
    _sessions: "dict[StreamId, ACPSession]" = field(init=False, compare=False, repr=False, default_factory=dict)

    # Optional connection opener. ``None`` means open the subclass's own
    # connection; tests set this to inject an in-process agent. Behavior, not
    # identity — carried by copy().
    _connect: "ConnectHook | None" = field(init=False, compare=False, repr=False, default=None)

    # Exception types that mean "the connection to the agent broke", so a turn
    # that hits one fails with ``ACPTransportError`` instead of coming back
    # empty. Subclasses widen this with whatever their transport raises.
    _transport_errors: ClassVar[tuple[type[BaseException], ...]] = (ConnectionError,)

    @property
    def _transport_label(self) -> str:
        """How this config's transport is named in errors and logs."""
        return "stdio"

    @property
    def _agent_label(self) -> str:
        """How the agent is named in errors, before it introduces itself."""
        return self.command[0] if self.command else "acp-agent"

    def _open_connection(
        self, client: "acp.Client"
    ) -> "AbstractAsyncContextManager[tuple[ClientSideConnection, Process | None]]":
        """Open the ACP connection for a session, honouring an injected opener."""
        if self._connect is not None:
            return self._connect(client)
        return self._connect_transport(client)

    def _connect_transport(
        self, client: "acp.Client"
    ) -> "AbstractAsyncContextManager[tuple[ClientSideConnection, Process | None]]":
        """Reach the agent by launching it, which is what this config is for."""
        import acp

        executable, *args = self.command
        # `use_unstable_protocol` is what registers the `elicitation/*` routes
        # on the client side; without it the SDK answers the agent's question
        # with method-not-found before the bridge ever sees it. Elicitation is
        # the only unstable route a Client serves, so this enables nothing else.
        return acp.spawn_agent_process(
            client,
            executable,
            *args,
            env=self.env,
            cwd=self.cwd,
            use_unstable_protocol=True,
            **_dispatch_kwargs(client),
        )

    def _gateway_address(self) -> GatewayAddress:
        """Where the tool gateway should bind, and what address the agent gets.

        Loopback-only by default, which is everything a locally-launched agent
        needs and nothing a remote one can reach.
        """
        return GatewayAddress()

    def copy(self, /, **overrides: object) -> Self:
        # dataclasses.replace can't statically check dynamic **overrides against
        # each field's type; the values are validated at construction instead.
        new = replace(self, **overrides)  # type: ignore[arg-type]
        new._connect = self._connect  # init=False, so replace() would reset it
        return new

    def create(self) -> "LLMClient":
        from .client import ACPClient

        return ACPClient(self)

    async def aclose(self) -> None:
        """Tear down every live ACP session started from this config."""
        sessions = list(self._sessions.values())
        self._sessions.clear()
        for session in sessions:
            await session.close()

    async def __aenter__(self) -> Self:
        """Enter a scope whose exit tears down every session this config started.

        A session outlives the ``agent.run()`` that created it — ``reply.ask()``
        reuses it — so the config's scope, not the run's, is the conversation's
        lifetime. Nothing reclaims a session implicitly.
        """
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()


@dataclass(slots=True)
class ClaudeCodeConfig(ACPConfig):
    """``ACPConfig`` preset for the Claude Code ACP adapter.

    Launches the ``@agentclientprotocol/claude-agent-acp`` bin, which must be on
    ``PATH`` (install globally, or override ``command`` to run it via
    ``npx -y @agentclientprotocol/claude-agent-acp``). The adapter wraps the
    Claude Agent SDK. Authenticate either by passing ``ANTHROPIC_API_KEY`` in
    ``env`` (billed per-token by the Anthropic API), or via an existing Claude
    Code login under ``$HOME`` -- ``~/.claude``, or a custom dir via
    ``CLAUDE_CONFIG_DIR`` passed in ``env`` -- which uses that login's plan.
    Only a small env whitelist is inherited, so a shell ``export`` of the key
    does not reach the subprocess; put it in ``env`` (see ``ACPConfig.env``).
    Select the model via the ``model`` field (one of the adapter's advertised
    ids — see ``ACPConfig.model``) or the adapter's ``ANTHROPIC_MODEL`` env var.
    """

    command: list[str] = field(default_factory=lambda: ["claude-agent-acp"])


@dataclass(slots=True)
class CodexConfig(ACPConfig):
    """``ACPConfig`` preset for the Codex ACP adapter.

    Launches the ``@agentclientprotocol/codex-acp`` bin, which must be on
    ``PATH`` (install globally, or override ``command`` to run it via
    ``npx -y @agentclientprotocol/codex-acp``). Authenticate either by passing
    ``CODEX_API_KEY`` (takes precedence) or ``OPENAI_API_KEY`` in ``env`` --
    billed per-token by the provider's API -- or with an existing ``codex
    login`` on the host, whose credentials live under ``$HOME`` (``~/.codex``,
    inherited automatically) and whose billing follows that login, which may be
    a ChatGPT subscription. Only a small env whitelist is inherited, so a shell
    ``export`` of a key does not reach the subprocess; put it in ``env`` (see
    ``ACPConfig.env``).
    Select the model via the ``model`` field (one of the adapter's advertised
    ids — see ``ACPConfig.model``) or the adapter's ``MODEL_PROVIDER`` env var.
    """

    command: list[str] = field(default_factory=lambda: ["codex-acp"])


@dataclass(slots=True)
class OpenCodeConfig(ACPConfig):
    """``ACPConfig`` preset for the OpenCode ACP adapter.

    Launches ``opencode acp``, which must be on ``PATH``. Authenticate with
    ``opencode auth login``; its credentials are stored on disk under ``$HOME``
    (inherited automatically), so no ``env`` is needed. Billing follows the
    provider that login is on (an API key or a subscription). Select the model
    via the ``model`` field (``"provider/model"`` as listed by
    ``opencode models``) or in OpenCode's config (``opencode.json``:
    ``"model": "provider/model"``).
    """

    command: list[str] = field(default_factory=lambda: ["opencode", "acp"])


@dataclass(slots=True)
class KiloCodeConfig(ACPConfig):
    """``ACPConfig`` preset for the Kilo Code ACP adapter.

    Launches ``kilo acp``; the ``kilo`` CLI must be on ``PATH`` (install with
    ``npm install -g @kilocode/cli``, or override ``command`` to run it via
    ``npx -y @kilocode/cli acp``). Authenticate with ``kilo auth login``; its
    credentials are stored on disk under ``$HOME`` (inherited automatically),
    so no ``env`` is needed. Billing follows the provider that login is on.
    Always set ``model`` explicitly (``"provider/model"`` as listed by
    ``kilo models``, e.g. ``"kilo/anthropic/claude-haiku-4.5"``): a fresh Kilo
    ACP session may default to an unsuitable model (an image model, at the
    time of writing), which ends every turn with an empty reply.
    """

    command: list[str] = field(default_factory=lambda: ["kilo", "acp"])
