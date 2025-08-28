# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any

from pydantic import BaseModel


class AgentBusMessage(BaseModel):
    messages: list[dict[str, Any]]
    context: dict[str, Any] | None = None
