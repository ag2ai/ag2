# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
try:
    from a2a.types import AgentCard
except ImportError as e:
    raise ImportError("a2a-sdk is not installed. Please install it with:\npip install ag2[a2a]") from e

import warnings

warnings.warn(
    (
        "AG2 Implementation for A2A support is in experimental mode "
        "and is subjected to breaking changes. Once it's stable enough the "
        "experimental mode will be removed. Your feedback is welcome."
    ),
    ImportWarning,
    stacklevel=2,
)

from autogen.remote.httpx_client_factory import HttpxClientFactory

from .agent_executor import AutogenAgentExecutor
from .client import A2aRemoteAgent
from .httpx_client_factory import MockClient
from .server import A2aAgentServer, CardSettings

__all__ = (
    "A2aAgentServer",
    "A2aRemoteAgent",
    "AgentCard",
    "AutogenAgentExecutor",
    "CardSettings",
    "HttpxClientFactory",
    "MockClient",
)
