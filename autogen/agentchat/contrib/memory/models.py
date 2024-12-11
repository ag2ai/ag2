# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from pydantic import BaseModel


class StaticMemoryOperation(BaseModel):
    operation: Literal["store", "append", "read", "delete"]
    doc_id: str
