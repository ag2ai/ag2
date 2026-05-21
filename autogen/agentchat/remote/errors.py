# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


class RemoteAgentError(Exception):
    """Base class for remote agent errors"""

    pass


class RemoteAgentNotFoundError(RemoteAgentError):
    """Raised when a remote agent is not found"""

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name
        super().__init__(f"Remote agent `{agent_name}` not found")
