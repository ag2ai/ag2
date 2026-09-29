# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""What each AG-UI server declares it can do, read the way a client reads it.

The document is checked against the protocol's own schema: the SDK's models
accept unknown and mistyped members, so they cannot say whether it conforms.
`test/ag_ui/fixtures/schema.json` and `fixtures/agent_capabilities/` are copied
verbatim from upstream `ag-ui-protocol/ag-ui` at `024332cb`
(`spec/1.0/schema.json`, `spec/1.0/fixtures/AgentCapabilities/`).
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from dirty_equals import IsPartialDict
from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from starlette.applications import Starlette
from starlette.routing import Route

from ag2 import Agent, TaskConfig
from ag2.a2ui import A2UIServer
from ag2.a2ui.transports import AgUiTransport
from ag2.ag_ui import AGUIStream
from ag2.events import HumanInputRequest, HumanMessage
from ag2.testing import TestConfig
from ag2.tools import Toolkit

_FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture(scope="module")
def capabilities_schema() -> Draft202012Validator:
    """The protocol's `AgentCapabilities` definition, as a validator."""
    schema = json.loads((_FIXTURES / "schema.json").read_text())
    return Draft202012Validator(
        {"$ref": f"{schema['$id']}#/$defs/AgentCapabilities"},
        registry=Registry().with_resource(schema["$id"], Resource.from_contents(schema)),
    )


def _answer_in_process(event: HumanInputRequest) -> HumanMessage:
    return HumanMessage("blue")


def _agent(**kwargs: Any) -> Agent:
    return Agent("test_agent", config=TestConfig("hello"), **kwargs)


def _ag_ui_app(agent: Agent) -> Starlette:
    # One endpoint answers on both paths: the route a client posts runs to,
    # and the sub-path other AG-UI integrations serve capabilities on.
    endpoint = AGUIStream(agent).build_asgi()
    return Starlette(routes=[Route("/agent", endpoint), Route("/agent/capabilities", endpoint)])


def _a2ui_app(agent: Agent) -> A2UIServer:
    return A2UIServer(agent, transport=AgUiTransport(path="/agent"), validate_responses=False)


_APPS = pytest.mark.parametrize("make_app", [_ag_ui_app, _a2ui_app], ids=["AGUIStream", "A2UI"])
_ROUTES = pytest.mark.parametrize("route", ["/agent", "/agent/capabilities"])


async def _capabilities(app: Any, route: str = "/agent/capabilities") -> dict[str, Any]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://ag-ui.test") as client:
        response = await client.get(route)
    assert response.status_code == 200
    document: dict[str, Any] = response.json()
    return document


@pytest.mark.parametrize("fixture", sorted((_FIXTURES / "agent_capabilities" / "valid").glob("*.json")), ids=str)
def test_the_schema_accepts_upstreams_valid_declarations(
    fixture: Path, capabilities_schema: Draft202012Validator
) -> None:
    capabilities_schema.validate(json.loads(fixture.read_text()))


@pytest.mark.parametrize(
    "fixture",
    sorted(p for p in (_FIXTURES / "agent_capabilities" / "invalid").glob("*.json") if ".expect" not in p.suffixes),
    ids=str,
)
def test_the_schema_refuses_upstreams_invalid_declarations(
    fixture: Path, capabilities_schema: Draft202012Validator
) -> None:
    assert not capabilities_schema.is_valid(json.loads(fixture.read_text()))


@_APPS
@_ROUTES
@pytest.mark.asyncio
async def test_the_declaration_conforms_to_the_protocol_schema(
    make_app: Callable[[Agent], Any], route: str, capabilities_schema: Draft202012Validator
) -> None:
    document = await _capabilities(make_app(_agent(tasks=TaskConfig())), route)

    capabilities_schema.validate(document)


@_APPS
@pytest.mark.asyncio
async def test_both_servers_declare_what_every_run_does(make_app: Callable[[Agent], Any]) -> None:
    document = await _capabilities(make_app(_agent()))

    assert document == IsPartialDict({
        "identity": {"name": "test_agent", "type": "ag2"},
        "transport": {"streaming": True},
        "reasoning": {"encrypted": False},
        "multimodal": {"input": {"image": True, "audio": True, "video": True, "pdf": True}},
        "humanInTheLoop": {"supported": True, "interrupts": True},
    })


@pytest.mark.asyncio
async def test_the_ag_ui_stream_takes_client_tools_and_sends_snapshots() -> None:
    document = await _capabilities(_ag_ui_app(_agent()))

    assert document == IsPartialDict({
        "tools": {"supported": True, "clientProvided": True},
        "state": {"snapshots": True},
    })


@pytest.mark.asyncio
async def test_the_a2ui_transport_declares_neither_client_tools_nor_snapshots() -> None:
    """It ignores the run's `tools` and never sends `STATE_SNAPSHOT`."""
    document = await _capabilities(_a2ui_app(_agent()))

    assert document["tools"] == {"supported": True}
    assert "state" not in document


@_APPS
@pytest.mark.asyncio
async def test_an_agent_answering_its_own_questions_puts_none_to_the_client(make_app: Callable[[Agent], Any]) -> None:
    document = await _capabilities(make_app(_agent(hitl_hook=_answer_in_process)))

    assert document["humanInTheLoop"] == {"supported": True, "interrupts": False}


@_APPS
@pytest.mark.asyncio
async def test_an_agent_that_runs_subtasks_declares_delegation(make_app: Callable[[Agent], Any]) -> None:
    document = await _capabilities(make_app(_agent(tasks=TaskConfig())))

    assert document["multiAgent"] == {"supported": True, "delegation": True}


@_APPS
@pytest.mark.parametrize("in_toolkit", [False, True])
@pytest.mark.asyncio
async def test_an_agent_delegated_as_a_tool_is_declared(
    make_app: Callable[[Agent], Any], in_toolkit: bool, capabilities_schema: Draft202012Validator
) -> None:
    delegate = _agent().as_tool(description="Gathers sources.")
    tools = [Toolkit(delegate)] if in_toolkit else [delegate]
    document = await _capabilities(make_app(_agent(tools=tools)))

    assert document["multiAgent"] == {
        "supported": True,
        "delegation": True,
        "subagents": [{"name": "test_agent", "description": "Gathers sources."}],
    }
    capabilities_schema.validate(document)


@_APPS
@pytest.mark.asyncio
async def test_an_agent_that_cannot_delegate_says_nothing_about_it(make_app: Callable[[Agent], Any]) -> None:
    """Omitted, which the protocol reads as undeclared rather than unsupported."""
    document = await _capabilities(make_app(_agent()))

    assert "multiAgent" not in document
