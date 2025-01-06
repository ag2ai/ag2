# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0


from typing import Any, Callable, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel

__all__ = ["BaseMessage"]


class BaseMessage(BaseModel):
    uuid: UUID

    def __init__(self, uuid: Optional[UUID] = None, **kwargs: Any) -> None:
        uuid = uuid or uuid4()
        super().__init__(uuid=uuid, **kwargs)

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        """Print message

        Args:
            f (Optional[Callable[..., Any]], optional): Print function. If none, python's default print will be used.
        """
        ...
