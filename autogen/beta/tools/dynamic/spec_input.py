# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, ConfigDict, Field


class DynamicAgentSpec(BaseModel):
    """Specification the LLM passes to ``create_and_run_agent``.

    Subset of :class:`autogen.beta.AgentSpec`: no ``response_schema`` on
    this first iteration. ``extra='forbid'`` causes the LLM to fail
    validation immediately if it tries to pass an unknown field (for
    example a stray ``response_schema``), instead of silently dropping it.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        description="Short identifier for the spawned agent (used in logs and task events).",
    )
    prompt: list[str] = Field(
        default_factory=list,
        description="System prompt lines for the dynamic agent.",
    )
    tool_names: list[str] = Field(
        default_factory=list,
        description="Subset of available tool names the dynamic agent may use.",
    )
