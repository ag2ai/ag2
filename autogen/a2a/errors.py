# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


from autogen.remote.errors import RemoteAgentError, RemoteAgentNotFoundError


class A2aClientError(RemoteAgentError):
    """Base class for A2A agent errors"""

    pass


class A2aAgentNotFoundError(A2aClientError, RemoteAgentNotFoundError):
    """Raised when a A2A agent is not found"""

    pass
