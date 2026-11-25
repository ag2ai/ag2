# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""kosa8 sandbox extension for AG2.

Runs agent shell commands and code in kosa8 Cloud sandboxes (microVMs)
through :class:`Kosa8Environment`, a backend for
:class:`~ag2.tools.SandboxShellTool` and :class:`~ag2.tools.SandboxCodeTool`.
It speaks the kosa8 REST API with ``httpx``, so it needs no extra package.

Maintained by kosa8.
"""

from .environment import Kosa8Environment
from .sandbox import Kosa8APIError, Kosa8Sandbox

__all__ = (
    "Kosa8APIError",
    "Kosa8Environment",
    "Kosa8Sandbox",
)
