# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any

from pydantic import BaseModel, Field


class AgentBusMessage(BaseModel):
    messages: list[dict[str, Any]] = Field(default_factory=list)
    context: dict[str, Any] | None = None


class RequestMessage(AgentBusMessage):
    client_tools: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def client_tool_names(self) -> set[str]:
        return get_tool_names(self.client_tools)


class ResponseMessage(AgentBusMessage):
    input_required: str | None = None
    cost: dict[str, Any] | None = None


class ServiceResponse(BaseModel):
    message: dict[str, Any] | None = None
    context: dict[str, Any] | None = None
    input_required: str | None = None


def get_tool_names(tools: list[dict[str, Any]]) -> set[str]:
    return set(filter(bool, (tool.get("function", {}).get("name", "") for tool in tools)))
