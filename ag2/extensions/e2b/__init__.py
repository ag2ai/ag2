# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""E2B Sandbox extension for AG2.

Runs agent shell commands and code in E2B cloud sandboxes through
:class:`E2BEnvironment`, a backend for :class:`~ag2.tools.SandboxShellTool`
and :class:`~ag2.tools.SandboxCodeTool`.

Maintained by E2B.
"""

from ag2.exceptions import missing_additional_dependency

try:
    from .environment import E2BEnvironment
except ImportError as e:
    E2BEnvironment = missing_additional_dependency("E2BEnvironment", "e2b>=2.0.0,<3", e)  # type: ignore[misc]

__all__ = ("E2BEnvironment",)
