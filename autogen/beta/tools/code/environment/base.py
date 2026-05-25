# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from autogen.beta.context import ConversationContext

CodeLanguage = Literal["python", "bash", "javascript", "typescript"]


@dataclass(slots=True)
class CodeRunResult:
    """Outcome of a single code-execution call.

    ``output`` carries combined stdout/stderr (already trimmed by the
    environment). ``exit_code`` follows POSIX conventions (0 = success).
    """

    output: str
    exit_code: int


@runtime_checkable
class CodeEnvironment(Protocol):
    """Backend that runs source code on behalf of :class:`SandboxCodeTool`.

    Implementations may target a local subprocess, a remote sandbox
    (Daytona, e2b, …), a container, or anything else — :class:`SandboxCodeTool`
    only depends on this protocol.
    """

    @property
    def supported_languages(self) -> tuple[CodeLanguage, ...]:
        """Languages this environment is willing to run.

        Surfaced in the tool description so the LLM knows what to ask for.
        """
        ...

    async def run(
        self,
        code: str,
        language: CodeLanguage,
        *,
        context: "ConversationContext | None" = None,
    ) -> CodeRunResult:
        """Execute *code* in *language*.

        ``context`` is the active conversation context, forwarded by
        :class:`SandboxCodeTool` so backends can resolve
        :class:`~autogen.beta.annotations.Variable` markers from
        ``context.variables`` (e.g. per-tenant credentials). Backends with
        no runtime-configurable parameters can ignore it.
        """
        ...
