# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Iterable
from typing import Any

from pydantic import BaseModel, Field

from ag2.agent import Agent, Plugin
from ag2.config.config import ModelConfig
from ag2.exceptions import ToolResolutionError
from ag2.hitl import HumanHook
from ag2.middleware.base import MiddlewareFactory
from ag2.observers import Observer
from ag2.response.proto import ResponseProto
from ag2.response.schema import RawSchema
from ag2.tools.final import FunctionTool, Toolkit
from ag2.tools.tool import Tool


class ResponseSchemaSpec(BaseModel):
    """JSON-serializable description of a response schema."""

    name: str
    description: str | None = None
    json_schema: dict[str, Any]

    def to_response_schema(self) -> ResponseProto[str]:
        """Reconstruct a ``RawSchema`` from this spec."""

        return RawSchema(
            self.json_schema,
            name=self.name,
            description=self.description,
        )


class AgentSpec(BaseModel):
    """JSON-serializable specification of an Agent.

    Captures the declarative, data-only parts of an ``Agent``: name, prompt,
    tool names, and response schema.

    Non-serializable parts (middleware, callbacks, dependencies, dynamic prompts)
    are intentionally excluded and must be supplied at reconstruction time via
    :meth:`to_agent`.
    """

    name: str
    prompt: list[str] = Field(default_factory=list)
    tool_names: list[str] = Field(default_factory=list)
    response_schema: ResponseSchemaSpec | None = None

    @classmethod
    def from_agent(cls, agent: Agent) -> "AgentSpec":
        """Create an ``AgentSpec`` from a live ``Agent`` instance.

        Only serializable state is captured. Dynamic prompts, middleware,
        callbacks, and dependencies are dropped.
        """

        tool_names: list[str] = [t.name for t in agent.tools]

        # Response schema
        rs_spec: ResponseSchemaSpec | None = None
        rs = agent._response_schema
        if rs is not None and rs.json_schema is not None:
            rs_spec = ResponseSchemaSpec(
                name=rs.name,
                description=rs.description,
                json_schema=rs.json_schema,
            )

        return cls(
            name=agent.name,
            prompt=list(agent._system_prompt),
            tool_names=tool_names,
            response_schema=rs_spec,
        )

    def to_agent(
        self,
        *,
        available_tools: Iterable[Tool | Callable[..., Any]] = (),
        config: ModelConfig | None = None,
        hitl_hook: HumanHook | None = None,
        middleware: Iterable[MiddlewareFactory] = (),
        observers: Iterable[Observer] = (),
        dependencies: dict[Any, Any] | None = None,
        variables: dict[str, Any] | None = None,
        response_schema: ResponseProto[Any] | type | None = None,
        plugins: Iterable[Plugin] = (),
    ) -> Agent:
        """Reconstruct an ``Agent`` from this spec."""

        # Build name -> tool index from available_tools
        tool_index: dict[str, Tool] = {}

        tools_to_visit = [FunctionTool.ensure_tool(t) for t in available_tools]
        while tools_to_visit:
            tool = tools_to_visit.pop()
            tool_index[tool.name] = tool
            if isinstance(tool, Toolkit):
                tools_to_visit.extend(tool.tools)

        # Resolve tools by name
        resolved_tools: list[Tool] = []
        missing: list[str] = []
        for name in self.tool_names:
            if name in tool_index:
                resolved_tools.append(tool_index[name])
            else:
                missing.append(name)

        if missing:
            raise ToolResolutionError(missing, sorted(tool_index.keys()))

        # Response schema: explicit param > spec > None
        final_rs = response_schema
        if final_rs is None and self.response_schema is not None:
            final_rs = self.response_schema.to_response_schema()

        return Agent(
            name=self.name,
            prompt=list(self.prompt),
            config=config,
            hitl_hook=hitl_hook,
            tools=resolved_tools,
            middleware=middleware,
            observers=observers,
            dependencies=dependencies,
            variables=variables,
            response_schema=final_rs,
            plugins=plugins,
        )
