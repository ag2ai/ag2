# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

import httpx
import pytest
import respx
from dirty_equals import IsPartialDict

from ag2 import Agent, Context, DataInput
from ag2.events import ModelResponse, ToolCallEvent, ToolCallsEvent, ToolResultsEvent
from ag2.extensions.tools.cohesivity import (
    CohesivityProvisionResult,
    CohesivityStatusResult,
    CohesivityTenant,
    CohesivityToolkit,
    _parse_dotfile,
    _user_agent,
)
from ag2.testing import TestConfig, TrackingConfig
from ag2.tools.final.function_tool import FunctionToolSchema

COHESIVITY_BASE_URL = "https://cohesivity.ai"

SAMPLE_GENESIS_RESPONSE = (
    "tenant_id=bold-elk-dining\n"
    "coh_management_key=coh_man_test1234567890ab\n"
    "coh_application_key=coh_app_test1234567890ab\n"
    "expires_at=2026-10-18T12:00:00Z\n"
    "tenant_lifecycle=ephemeral\n"
    "runtime_profile=v1-42\n"
)

SAMPLE_PROVISION_RESPONSE: dict[str, Any] = {
    "success": True,
    "resource": "postgres",
    "status": "active",
    "edge_url": "https://cohesivity.ai/edge/postgres",
    "region": "aws-ap-southeast-1",
    "quota_profile": "ephemeral",
}

SAMPLE_STATUS_RESPONSE: dict[str, Any] = {
    "tenant_id": "bold-elk-dining",
    "status": "active",
    "resources": {
        "postgres": {"status": "active", "edge_url": "https://cohesivity.ai/edge/postgres"},
    },
    "expires_at": "2026-10-18T12:00:00Z",
}


def _tool_call_config(
    arguments: dict[str, object],
    *,
    tool_name: str,
    final_reply: str = "done",
) -> TestConfig:
    return TestConfig(
        ModelResponse(
            tool_calls=ToolCallsEvent([
                ToolCallEvent(arguments=json.dumps(arguments), name=tool_name),
            ]),
        ),
        final_reply,
    )


class TestParseDotfile:
    def test_parses_key_value_lines(self) -> None:
        text = "tenant_id=abc\ncoh_management_key=coh_man_xyz\n"
        result = _parse_dotfile(text)
        assert result == {"tenant_id": "abc", "coh_management_key": "coh_man_xyz"}

    def test_skips_comments_and_blanks(self) -> None:
        text = "# comment\n\ntenant_id=abc\n"
        result = _parse_dotfile(text)
        assert result == {"tenant_id": "abc"}

    def test_handles_values_with_equals(self) -> None:
        text = "key=value=with=equals\n"
        result = _parse_dotfile(text)
        assert result == {"key": "value=with=equals"}


class TestUserAgent:
    def test_default_attribution(self) -> None:
        assert _user_agent("gh-ag2") == "{npx:ag2.gh-ag2}"

    def test_empty_attribution(self) -> None:
        assert _user_agent("") == "{npx:ag2}"

    def test_custom_attribution(self) -> None:
        assert _user_agent("gh-custom") == "{npx:ag2.gh-custom}"


@pytest.mark.asyncio
class TestSchema:
    async def test_default_schemas(self, context: Context) -> None:
        toolkit = CohesivityToolkit(management_key="test")

        schemas = list(await toolkit.schemas(context))
        names = [s.function.name for s in schemas]

        assert names == ["cohesivity_create_tenant", "cohesivity_provision", "cohesivity_status"]

    async def test_all_schemas_are_function_tools(self, context: Context) -> None:
        toolkit = CohesivityToolkit(management_key="test")

        schemas = list(await toolkit.schemas(context))

        for schema in schemas:
            assert isinstance(schema, FunctionToolSchema)

    async def test_provision_schema_has_resource_param(self, context: Context) -> None:
        toolkit = CohesivityToolkit(management_key="test")

        schemas = list(await toolkit.schemas(context))
        provision_schema = schemas[1]

        assert provision_schema.function.name == "cohesivity_provision"
        assert provision_schema.function.parameters == IsPartialDict({
            "required": ["resource"],
            "properties": IsPartialDict({"resource": IsPartialDict({"type": "string"})}),
        })


@pytest.mark.asyncio
class TestCreateTenant:
    @respx.mock
    async def test_creates_tenant_and_returns_parsed_keys(self) -> None:
        respx.post(f"{COHESIVITY_BASE_URL}/api/genesis").mock(
            return_value=httpx.Response(200, text=SAMPLE_GENESIS_RESPONSE)
        )
        toolkit = CohesivityToolkit()
        config = TrackingConfig(_tool_call_config({}, tool_name="cohesivity_create_tenant"))
        agent = Agent("a", config=config, tools=[toolkit])

        await agent.ask("create a backend")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            CohesivityTenant(
                tenant_id="bold-elk-dining",
                management_key="coh_man_test1234567890ab",
                application_key="coh_app_test1234567890ab",
                base_url="https://cohesivity.ai",
                expires_at="2026-10-18T12:00:00Z",
                runtime_profile="v1-42",
            )
        )

    @respx.mock
    async def test_stores_management_key_for_subsequent_calls(self) -> None:
        respx.post(f"{COHESIVITY_BASE_URL}/api/genesis").mock(
            return_value=httpx.Response(200, text=SAMPLE_GENESIS_RESPONSE)
        )
        toolkit = CohesivityToolkit()
        agent = Agent(
            "a",
            config=_tool_call_config({}, tool_name="cohesivity_create_tenant"),
            tools=[toolkit],
        )

        await agent.ask("create")

        assert toolkit._management_key == "coh_man_test1234567890ab"

    @respx.mock
    async def test_sends_correct_user_agent(self) -> None:
        route = respx.post(f"{COHESIVITY_BASE_URL}/api/genesis").mock(
            return_value=httpx.Response(200, text=SAMPLE_GENESIS_RESPONSE)
        )
        toolkit = CohesivityToolkit()
        agent = Agent(
            "a",
            config=_tool_call_config({}, tool_name="cohesivity_create_tenant"),
            tools=[toolkit],
        )

        await agent.ask("create")

        assert route.calls.last.request.headers["user-agent"] == "{npx:ag2.gh-ag2}"

    @respx.mock
    async def test_no_content_type_on_genesis(self) -> None:
        route = respx.post(f"{COHESIVITY_BASE_URL}/api/genesis").mock(
            return_value=httpx.Response(200, text=SAMPLE_GENESIS_RESPONSE)
        )
        toolkit = CohesivityToolkit()
        agent = Agent(
            "a",
            config=_tool_call_config({}, tool_name="cohesivity_create_tenant"),
            tools=[toolkit],
        )

        await agent.ask("create")

        # Genesis sends no body, so Content-Type should not be application/json
        assert "application/json" not in route.calls.last.request.headers.get("content-type", "")


@pytest.mark.asyncio
class TestProvision:
    @respx.mock
    async def test_provisions_resource(self) -> None:
        respx.post(f"{COHESIVITY_BASE_URL}/api/resources/postgres").mock(
            return_value=httpx.Response(200, json=SAMPLE_PROVISION_RESPONSE)
        )
        toolkit = CohesivityToolkit(management_key="coh_man_test1234567890ab")
        config = TrackingConfig(_tool_call_config({"resource": "postgres"}, tool_name="cohesivity_provision"))
        agent = Agent("a", config=config, tools=[toolkit])

        await agent.ask("provision postgres")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            CohesivityProvisionResult(
                resource="postgres",
                success=True,
                edge_url="https://cohesivity.ai/edge/postgres",
                details={
                    "resource": "postgres",
                    "status": "active",
                    "region": "aws-ap-southeast-1",
                    "quota_profile": "ephemeral",
                },
            )
        )

    @respx.mock
    async def test_sends_auth_header(self) -> None:
        route = respx.post(f"{COHESIVITY_BASE_URL}/api/resources/redis").mock(
            return_value=httpx.Response(200, json={"success": True})
        )
        toolkit = CohesivityToolkit(management_key="coh_man_secret")
        agent = Agent(
            "a",
            config=_tool_call_config({"resource": "redis"}, tool_name="cohesivity_provision"),
            tools=[toolkit],
        )

        await agent.ask("provision redis")

        assert route.calls.last.request.headers["authorization"] == "Bearer coh_man_secret"

    async def test_raises_without_management_key(self) -> None:
        toolkit = CohesivityToolkit()
        with pytest.raises(RuntimeError, match="No management key available"):
            toolkit._require_key()


@pytest.mark.asyncio
class TestStatus:
    @respx.mock
    async def test_returns_tenant_status(self) -> None:
        respx.get(f"{COHESIVITY_BASE_URL}/api/status").mock(
            return_value=httpx.Response(200, json=SAMPLE_STATUS_RESPONSE)
        )
        toolkit = CohesivityToolkit(management_key="coh_man_test1234567890ab")
        config = TrackingConfig(_tool_call_config({}, tool_name="cohesivity_status"))
        agent = Agent("a", config=config, tools=[toolkit])

        await agent.ask("check status")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            CohesivityStatusResult(
                tenant_id="bold-elk-dining",
                status="active",
                resources={
                    "postgres": {"status": "active", "edge_url": "https://cohesivity.ai/edge/postgres"},
                },
                details={"expires_at": "2026-10-18T12:00:00Z"},
            )
        )

    @respx.mock
    async def test_sends_auth_and_user_agent(self) -> None:
        route = respx.get(f"{COHESIVITY_BASE_URL}/api/status").mock(
            return_value=httpx.Response(200, json=SAMPLE_STATUS_RESPONSE)
        )
        toolkit = CohesivityToolkit(management_key="coh_man_test1234567890ab")
        agent = Agent(
            "a",
            config=_tool_call_config({}, tool_name="cohesivity_status"),
            tools=[toolkit],
        )

        await agent.ask("status")

        assert route.calls.last.request.headers["authorization"] == "Bearer coh_man_test1234567890ab"
        assert route.calls.last.request.headers["user-agent"] == "{npx:ag2.gh-ag2}"


@pytest.mark.asyncio
class TestCustomConfig:
    @respx.mock
    async def test_custom_base_url(self) -> None:
        route = respx.post("https://custom.example.com/api/genesis").mock(
            return_value=httpx.Response(200, text=SAMPLE_GENESIS_RESPONSE)
        )
        toolkit = CohesivityToolkit(base_url="https://custom.example.com")
        agent = Agent(
            "a",
            config=_tool_call_config({}, tool_name="cohesivity_create_tenant"),
            tools=[toolkit],
        )

        await agent.ask("create")

        assert route.calls.last is not None

    @respx.mock
    async def test_custom_attribution(self) -> None:
        route = respx.post(f"{COHESIVITY_BASE_URL}/api/genesis").mock(
            return_value=httpx.Response(200, text=SAMPLE_GENESIS_RESPONSE)
        )
        toolkit = CohesivityToolkit(attribution="gh-custom-repo")
        agent = Agent(
            "a",
            config=_tool_call_config({}, tool_name="cohesivity_create_tenant"),
            tools=[toolkit],
        )

        await agent.ask("create")

        assert route.calls.last.request.headers["user-agent"] == "{npx:ag2.gh-custom-repo}"
