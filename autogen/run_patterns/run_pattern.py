# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, Protocol, Union

from ..doc_utils import export_module

if TYPE_CHECKING:
    from ..agentchat.agent import LLMMessageType
    from ..agentchat.chat import ChatResult


@export_module("autogen.run_patterns")
class RunPatternProtocol(Protocol):
    def run(
        self,
        message: str,
        messages: Iterable["LLMMessageType"],
        summary_method: Optional[Union[str, Callable[..., Any]]],
    ) -> "ChatResult": ...

    async def a_run(
        self,
        message: str,
        messages: Iterable["LLMMessageType"],
        summary_method: Optional[Union[str, Callable[..., Any]]],
    ) -> "ChatResult": ...
