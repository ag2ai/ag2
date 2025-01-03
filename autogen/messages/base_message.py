# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0


from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel

__all__ = ["BaseMessage"]


class BaseMessage(BaseModel):
    uuid: UUID

    def __init__(self, uuid: Optional[UUID] = None, **kwargs: Any) -> None:
        uuid = uuid or uuid4()
        super().__init__(uuid=uuid, **kwargs)
