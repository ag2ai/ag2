# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

# Filters live with the adapter that owns them; re-exported here so v1
# call-sites (`from autogen.beta.tools.shell.environment.base import …`)
# keep working.
from autogen.beta.tools.sandbox.filter import READONLY_COMMANDS, check_ignore, matches

if TYPE_CHECKING:
    from autogen.beta.context import ConversationContext

__all__ = (
    "READONLY_COMMANDS",
    "ShellEnvironment",
    "check_ignore",
    "matches",
)


@runtime_checkable
class ShellEnvironment(Protocol):
    @property
    def workdir(self) -> Path: ...

    def run(self, command: str, *, context: "ConversationContext | None" = None) -> str:
        """Execute *command* and return its output.

        ``context`` is the active conversation context, forwarded by
        :class:`~autogen.beta.tools.shell.LocalShellTool` so backends can
        resolve :class:`~autogen.beta.annotations.Variable` markers from
        ``context.variables`` (e.g. per-tenant credentials).  Backends with
        no runtime-configurable parameters can ignore it.
        """
        ...
