# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Config classes for ACP-backed agents.

``ACPConfig`` implements the :class:`~ag2.config.config.ModelConfig`
protocol; ``create()`` returns an ``ACPClient`` that drives the CLI agent over
the Agent Client Protocol. ``ClaudeCodeConfig``, ``CodexConfig``,
``OpenCodeConfig`` and ``KiloCodeConfig`` are thin subclasses carrying the
launch defaults for the Claude Code, Codex, OpenCode and Kilo Code ACP
adapters respectively.
"""

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

from typing_extensions import Self

if TYPE_CHECKING:
    from asyncio.subprocess import Process
    from collections.abc import Callable
    from contextlib import AbstractAsyncContextManager

    import acp
    from acp.core import ClientSideConnection

    from ag2.config.client import LLMClient
    from ag2.context import StreamId

    from .session import ACPSession

    # Opens the ACP connection for a session. Production uses ``spawn_agent_process``
    # (a subprocess); tests inject an in-process double (see ``acp.testing``).
    ConnectHook = Callable[["acp.Client"], "AbstractAsyncContextManager[tuple[ClientSideConnection, Process | None]]"]

PermissionPolicy = Literal["ask", "auto", "deny"]


@dataclass(slots=True)
class ACPConfig:
    """Configuration for driving a CLI coding agent over ACP.

    Attributes:
        command: Executable + base args launching the agent in ACP mode,
            e.g. ``["claude-agent-acp"]``. The first element is the executable.
        cwd: Workspace root passed to ``session/new``.
        env: Extra environment variables for the subprocess. The subprocess does
            NOT inherit the full parent environment: only a small whitelist
            (``HOME``, ``LOGNAME``, ``PATH``, ``SHELL``, ``TERM``, ``USER``) is
            inherited, merged with this mapping. So API-key auth must be passed
            here explicitly (a shell ``export`` of the key is not inherited); a
            disk login under ``$HOME`` (e.g. ``~/.claude``) works without it.
        model: Agent model selection. Applied at session start via ACP
            ``session/set_config_option`` when the agent advertises a model
            picker in ``session/new`` (Claude Code, OpenCode and Kilo Code all
            do); the value must be one of the agent's advertised model ids.
            ``None`` keeps the agent's default. When the agent has no model
            option the value is response metadata only.
        permission_policy: How to answer ``session/request_permission``:
            ``"ask"`` routes to the agent's ``hitl_hook``/``context.input``,
            ``"auto"`` allows, ``"deny"`` rejects.
        fs_root: Root for mediated ``fs/*`` access (defaults to ``cwd``).
        allow_terminal: Whether to advertise the ACP terminal capability.
        additional_directories: Extra ACP workspace roots.
        startup_timeout: Seconds to allow for the tool gateway's HTTP server
            to start when tools are exposed.
        turn_timeout: Per-prompt-turn timeout in seconds (``None`` = no limit).
        cancel_timeout: Grace period (seconds) after a timed-out turn signals
            ``session/cancel`` for the agent to return the in-flight prompt. If
            the agent does not respond within it, the subprocess is hard-stopped.
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

    # Run-scoped live sessions, keyed by stream id. Not part of identity and not
    # carried by ``copy()`` (a copy is a distinct config with its own sessions).
    _sessions: "dict[StreamId, ACPSession]" = field(init=False, compare=False, repr=False, default_factory=dict)

    # Optional connection opener. ``None`` means spawn the real subprocess; tests
    # set this to inject an in-process agent. Behavior, not identity — carried by copy().
    _connect: "ConnectHook | None" = field(init=False, compare=False, repr=False, default=None)

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
        """Tear down every live ACP subprocess started from this config."""
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
