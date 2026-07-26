# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Drive external CLI coding agents (Claude Code, Codex, …) via the Agent Client Protocol.

AG2 plays the ACP *Client* role; each CLI agent runs as an ACP *Agent* subprocess.
The integration is a :class:`ModelConfig` + :class:`LLMClient` pair — no changes
to the :class:`~ag2.Agent` class.
"""

from ag2.exceptions import missing_optional_dependency

try:
    from .config import ACPConfig, ClaudeCodeConfig, CodexConfig, OpenCodeConfig
    from .tool_gateway import MCPCapabilityError
except ImportError as e:  # pragma: no cover - exercised only when ag2[acp] is absent
    ACPConfig = missing_optional_dependency("ACPConfig", "acp", e)  # type: ignore[misc]
    ClaudeCodeConfig = missing_optional_dependency("ClaudeCodeConfig", "acp", e)  # type: ignore[misc]
    CodexConfig = missing_optional_dependency("CodexConfig", "acp", e)  # type: ignore[misc]
    OpenCodeConfig = missing_optional_dependency("OpenCodeConfig", "acp", e)  # type: ignore[misc]
    MCPCapabilityError = missing_optional_dependency("MCPCapabilityError", "acp", e)  # type: ignore[misc]

__all__ = ["ACPConfig", "ClaudeCodeConfig", "CodexConfig", "MCPCapabilityError", "OpenCodeConfig"]
