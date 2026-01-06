# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

try:
    import httpx  # noqa: F401
except ImportError as e:
    raise ImportError("httpx is not installed. Please install it with:\npip install httpx") from e

from .agent_service import AgentService
from .errors import RemoteAgentError, RemoteAgentNotFoundError
from .httpx_client_factory import ClientFactory, EmptyClientFactory, HttpxClientFactory
from .protocol import RequestMessage, ResponseMessage

__all__ = (
    "AgentService",
    "ClientFactory",
    "EmptyClientFactory",
    "HttpxClientFactory",
    "RemoteAgentError",
    "RemoteAgentNotFoundError",
    "RequestMessage",
    "ResponseMessage",
)
