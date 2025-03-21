# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, Protocol, Union

if TYPE_CHECKING:
    from ..agentchat.agent import Agent, LLMMessageType
    from ..agentchat.chat import ChatResult


class ChatManagerProtocol(Protocol):
    def run(
        self,
        *agents: "Agent",
        message: str,
        messages: Iterable["LLMMessageType"],
        max_turns: int,
        summary_method: Optional[Union[str, Callable[..., Any]]],
    ) -> "ChatResult": ...

    async def a_run(
        self,
        *agents: "Agent",
        message: str,
        messages: Iterable["LLMMessageType"],
        max_turns: int,
        summary_method: Optional[Union[str, Callable[..., Any]]],
    ) -> "ChatResult": ...
