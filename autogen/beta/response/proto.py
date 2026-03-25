# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Any, Generic

from fast_depends import Provider
from typing_extensions import TypeVar as TypeVar313

from autogen.beta.annotations import Context

T = TypeVar313("T", default=str)


class ResponseProto(ABC, Generic[T]):
    name: str
    description: str | None
    json_schema: dict[str, Any] | None
    system_prompt: str | None

    @abstractmethod
    async def validate(
        self,
        response: str,
        context: "Context",
        provider: "Provider | None" = None,
    ) -> T: ...
