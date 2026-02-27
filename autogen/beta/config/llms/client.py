# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from autogen.beta.events import BaseEvent
from autogen.beta.stream import Context
from autogen.beta.tools import Tool


@runtime_checkable
class LLMClient(Protocol):
    async def __call__(
        self,
        *messages: BaseEvent,
        ctx: Context,
        tools: Iterable[Tool],
    ) -> None: ...
