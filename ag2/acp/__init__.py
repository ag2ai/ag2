# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Drive external CLI coding agents (Claude Code, Codex, …) via the Agent Client Protocol.

AG2 plays the ACP *Client* role; each CLI agent runs as an ACP *Agent* subprocess.
The integration is a :class:`ModelConfig` + :class:`LLMClient` pair — no changes
to the :class:`~ag2.Agent` class.
"""

from .config import ACPConfig, ClaudeCodeConfig, CodexConfig, OpenCodeConfig

__all__ = ["ACPConfig", "ClaudeCodeConfig", "CodexConfig", "OpenCodeConfig"]
