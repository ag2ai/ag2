# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import Dict, Optional

from autogen.agentchat import ConversableAgent


@dataclass
class BroadcastMessage:
    """Message to be broadcast to all agents."""

    content: Dict
    request_halt: bool = False


@dataclass
class ResetMessage:
    """Message to reset agent state."""

    pass


@dataclass
class OrchestrationEvent:
    """Event for logging orchestration."""

    source: str
    message: str
