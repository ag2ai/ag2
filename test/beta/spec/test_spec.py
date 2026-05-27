# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from dirty_equals import IsPartialDict
from pydantic import BaseModel

from autogen.beta import Agent, AgentSpec
from autogen.beta.exceptions import ToolResolutionError
from autogen.beta.response import ResponseSchema
from autogen.beta.spec import ResponseSchemaSpec
from autogen.beta.tools import FilesystemToolkit, WebSearchTool

from .helpers import add, greet, make_agent, multiply


class Answer(BaseModel):
    value: int
    reasoning: str


class TestFromAgent:
    def test_extracts_name_and_prompt(self) -> None:
        agent = make_agent(prompt="Be helpful.")
        spec = AgentSpec.from_agent(agent)

        assert spec.model_dump() == IsPartialDict({"name": "test_agent", "prompt": ["Be helpful."]})

    def test_extracts_tools(self) -> None:
        agent = make_agent(tools=[add, multiply, greet])
        spec = AgentSpec.from_agent(agent)

        assert spec.tool_names == ["add", "multiply", "greet"]

    def test_no_tools(self) -> None:
        agent = make_agent(tools=[])
        spec = AgentSpec.from_agent(agent)

        assert spec.tool_names == []

    def test_with_response_schema(self) -> None:
        agent = make_agent(response_schema=Answer)
        spec = AgentSpec.from_agent(agent)

        rs = spec.response_schema
        assert rs is not None
        assert rs.name == "Answer"
        assert "value" in str(rs.json_schema)


class TestToAgent:
    def test_resolves_tools(self) -> None:
        spec = AgentSpec(
            name="resolver",
            prompt=["Resolve tools."],
            tool_names=["add", "greet"],
        )

        agent = spec.to_agent(available_tools=[add, multiply, greet])

        assert agent.name == "resolver"
        assert [t.name for t in agent.tools] == ["add", "greet"]

    def test_missing_tool_raises(self) -> None:
        spec = AgentSpec(
            name="broken",
            tool_names=["add", "nonexistent"],
        )

        with pytest.raises(ToolResolutionError, match="nonexistent"):
            spec.to_agent(available_tools=[add, multiply])

    def test_manual_spec(self) -> None:
        spec = AgentSpec(
            name="research_bot",
            prompt=["You are a researcher.", "Be thorough."],
            tool_names=["add"],
        )

        agent = spec.to_agent(available_tools=[add, multiply])

        assert agent.name == "research_bot"
        assert agent._system_prompt == ["You are a researcher.", "Be thorough."]
        assert len(agent.tools) == 1

    def test_passes_hitl_hook(self) -> None:
        spec = AgentSpec(name="bot")

        def my_hook(msg: str) -> str:
            return msg

        agent = spec.to_agent(hitl_hook=my_hook)
        assert agent._hitl_hook is not None

    def test_passes_variables(self) -> None:
        spec = AgentSpec(name="bot")

        agent = spec.to_agent(variables={"key": "value"})
        assert agent._agent_variables == {"key": "value"}

    def test_passes_dependencies(self) -> None:
        spec = AgentSpec(name="bot")

        agent = spec.to_agent(dependencies={"db": "mock"})
        assert agent._agent_dependencies == {"db": "mock"}


class TestRoundTrip:
    def test_json(self) -> None:
        agent = make_agent(prompt="Round trip test.")

        spec = AgentSpec.from_agent(agent)

        json_str = spec.model_dump_json()

        restored_spec = AgentSpec.model_validate_json(json_str)
        assert restored_spec == spec

        restored = restored_spec.to_agent(available_tools=[add, multiply])
        # restored == agent
        assert restored.name == "test_agent"
        assert restored._system_prompt == ["Round trip test."]
        assert len(restored.tools) == 2

    def test_response_schema_spec(self) -> None:
        rs = ResponseSchema(Answer)
        assert rs.json_schema is not None
        rs_spec = ResponseSchemaSpec(
            name=rs.name,
            description=rs.description,
            json_schema=rs.json_schema,
        )

        json_data = rs_spec.model_dump_json()
        restored = ResponseSchemaSpec.model_validate_json(json_data)

        assert restored.model_dump() == IsPartialDict({"name": rs_spec.name})
        assert restored.json_schema == rs_spec.json_schema

        raw = restored.to_response_schema()
        assert raw.name == rs.name
        assert raw.json_schema == rs.json_schema


class TestBuiltinTools:
    def test_serialization(self) -> None:
        ws = WebSearchTool()
        agent = make_agent(tools=[add, ws])
        spec = AgentSpec.from_agent(agent)

        assert spec.tool_names == ["add", "web_search"]

    def test_resolution(self) -> None:
        spec = AgentSpec(
            name="bot",
            tool_names=["add", "web_search"],
        )

        agent = spec.to_agent(available_tools=[add, WebSearchTool()])

        assert len(agent.tools) == 2

    def test_missing_raises(self) -> None:
        spec = AgentSpec(
            name="bot",
            tool_names=["web_search"],
        )

        with pytest.raises(ToolResolutionError, match="web_search"):
            spec.to_agent(available_tools=[add])


class TestToolkit:
    def test_round_trip(self, tmp_path: Path) -> None:
        fs = FilesystemToolkit(base_path=tmp_path, read_only=True)
        agent = Agent(name="bot", tools=[add, fs])
        spec = AgentSpec.from_agent(agent)

        assert spec.tool_names == ["add", "filesystem_toolkit"]

        restored = spec.to_agent(available_tools=[add, fs])
        assert [t.name for t in restored.tools] == [t.name for t in agent.tools]

    def test_unpack_inner_tools(self, tmp_path: Path) -> None:
        ws = WebSearchTool()
        fs = FilesystemToolkit(base_path=tmp_path, read_only=True)

        spec = AgentSpec(
            name="bot",
            tool_names=["add", "web_search", "read_file"],
        )

        agent = spec.to_agent(available_tools=[add, multiply, ws, fs])

        assert [t.name for t in agent.tools] == ["add", "web_search", "read_file"]
