# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Config classes for ACP-backed agents.

``ACPConfig`` implements the :class:`~ag2.config.config.ModelConfig`
protocol; ``create()`` returns an ``ACPClient`` that drives the CLI agent over
the Agent Client Protocol. ``ClaudeCodeConfig``, ``CodexConfig`` and
``OpenCodeConfig`` are thin subclasses carrying the launch defaults for the
Claude Code, Codex and OpenCode ACP adapters respectively.
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
        env: Extra environment variables for the subprocess (auth typically
            lives here or is inherited from the parent process).
        model: Agent model selection, when the CLI supports it.
        permission_policy: How to answer ``session/request_permission``:
            ``"ask"`` routes to the agent's ``hitl_hook``/``context.input``,
            ``"auto"`` allows, ``"deny"`` rejects.
        fs_root: Root for mediated ``fs/*`` access (defaults to ``cwd``).
        allow_terminal: Whether to advertise the ACP terminal capability.
        additional_directories: Extra ACP workspace roots.
        startup_timeout: Seconds to allow for subprocess spawn + handshake.
        turn_timeout: Per-prompt-turn timeout in seconds (``None`` = no limit).
        cancel_timeout: Grace period (seconds) after a timed-out turn signals
            ``session/cancel`` for the agent to return the in-flight prompt. If
            the agent does not respond within it, the subprocess is hard-stopped.
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


@dataclass(slots=True)
class ClaudeCodeConfig(ACPConfig):
    """``ACPConfig`` preset for the Claude Code ACP adapter.

    Launches the ``@agentclientprotocol/claude-agent-acp`` bin, which must be on
    ``PATH`` (install globally, or override ``command`` to run it via
    ``npx -y @agentclientprotocol/claude-agent-acp``). The adapter wraps the
    Claude Agent SDK; authenticate by setting ``ANTHROPIC_API_KEY`` in ``env``
    or by pointing ``CLAUDE_CONFIG_DIR`` at an existing Claude Code login.
    Select the model via the adapter's ``ANTHROPIC_MODEL`` env var (the
    ``model`` field is currently response metadata only, not sent to the agent).
    """

    command: list[str] = field(default_factory=lambda: ["claude-agent-acp"])


@dataclass(slots=True)
class CodexConfig(ACPConfig):
    """``ACPConfig`` preset for the Codex ACP adapter.

    Launches the ``@agentclientprotocol/codex-acp`` bin, which must be on
    ``PATH`` (install globally, or override ``command`` to run it via
    ``npx -y @agentclientprotocol/codex-acp``). Authenticate by setting
    ``CODEX_API_KEY`` (takes precedence) or ``OPENAI_API_KEY`` in ``env``.
    Select the model via the adapter's ``MODEL_PROVIDER`` env var (the
    ``model`` field is currently response metadata only, not sent to the agent).
    """

    command: list[str] = field(default_factory=lambda: ["codex-acp"])


@dataclass(slots=True)
class OpenCodeConfig(ACPConfig):
    """``ACPConfig`` preset for the OpenCode ACP adapter.

    Launches ``opencode acp``, which must be on ``PATH``. Authenticate with
    ``opencode auth login`` (or env / ``.env``). Select the model in OpenCode's
    config (``opencode.json``: ``"model": "provider/model"``); the ``acp``
    subcommand takes no ``--model`` flag, and the ``model`` field here is
    response metadata only, not sent to the agent.
    """

    command: list[str] = field(default_factory=lambda: ["opencode", "acp"])
