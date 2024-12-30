# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from typing import Any

from fast_depends import Depends as FastDepends

__all__ = ["BaseContext", "ChatContext", "Depends"]


class BaseContext(ABC):
    pass


class ChatContext(BaseContext):
    messages: list[str] = []


def Depends(x: Any) -> Any:
    if isinstance(x, BaseContext):
        return FastDepends(lambda: x)

    return FastDepends(x)
