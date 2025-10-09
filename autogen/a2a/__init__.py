# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

try:
    from a2a.types import AgentCard
except ImportError as e:
    raise ImportError("a2a-sdk is not installed. Please install it with:\npip install ag2[a2a]") from e

from .agent_executor import AutogenAgentExecutor
from .client import A2aRemoteAgent
from .server import A2aAgentServer, CardSettings

__all__ = (  # noqa: RUF022
    "A2aAgentServer",
    "A2aRemoteAgent",
    "AutogenAgentExecutor",
    "AgentCard",
    "CardSettings",
)
