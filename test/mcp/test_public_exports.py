# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from inspect import signature
from typing import Annotated

import pytest
from mcp.client._input_required import run_input_required_driver
from mcp.client._probe import negotiate_auto
from mcp.server.mcpserver import Elicit as SDKElicit
from mcp.server.mcpserver import ListRoots as SDKListRoots
from mcp.server.mcpserver import Resolve as SDKResolve
from mcp.server.mcpserver import Sample as SDKSample
from mcp.server.request_state import RequestStateSecurity as SDKRequestStateSecurity
from pydantic import BaseModel

import ag2.mcp
from ag2.mcp import Elicit, ListRoots, MCPFunctionTool, RequestStateSecurity, Resolve, Sample, mcp_tool
from ag2.tools import MCPAnswerPolicy, MCPServerConfig, MCPStdioServerConfig

# The names the MCP namespace advertises. Pinned rather than derived, so a rename
# is a decision someone made here rather than a diff nobody read.
_PUBLIC_NAMES = {
    "AppContent",
    "AppSandbox",
    "AppText",
    "AskContext",
    "ContextProvider",
    "Elicit",
    "ExtensionMap",
    "ListRoots",
    "MCPApp",
    "MCPFunctionTool",
    "MCPRequestContext",
    "MCPServer",
    "Prompt",
    "PromptArgument",
    "PromptMessage",
    "RequestStateSecurity",
    "Resolve",
    "Resource",
    "ResourceCsp",
    "ResourcePermissions",
    "ResourceTemplate",
    "Sample",
    "SessionConfig",
    "Visibility",
    "build_ask_tool",
    "client_extension",
    "client_supports_apps",
    "mcp_tool",
}


def test_every_public_name_resolves() -> None:
    """``__all__`` must not advertise a name the module does not expose."""
    assert [name for name in ag2.mcp.__all__ if not hasattr(ag2.mcp, name)] == []


def test_the_advertised_surface_is_the_one_that_was_agreed() -> None:
    assert set(ag2.mcp.__all__) == _PUBLIC_NAMES


def test_the_calling_client_model_has_no_options_class() -> None:
    """``client_model`` is a boolean; nothing in that position needs naming."""
    assert "ClientModel" not in ag2.mcp.__all__
    assert not hasattr(ag2.mcp, "ClientModel")


class TestTheCuratedSDKReExports:
    """AG2's own examples type five MCP SDK names; the rest of the SDK is not mirrored."""

    def test_they_are_the_sdk_s_own_objects(self) -> None:
        assert (Resolve, Elicit, Sample, ListRoots, RequestStateSecurity) == (
            SDKResolve,
            SDKElicit,
            SDKSample,
            SDKListRoots,
            SDKRequestStateSecurity,
        )

    def test_nothing_else_is_mirrored(self) -> None:
        # A second protocol taxonomy is the thing being avoided: wire models and
        # less common types come from ``mcp`` directly.
        assert {"CreateMessageRequestParams", "ElicitResult", "Root", "Tool"} & set(ag2.mcp.__all__) == set()


class TestTheDeterministicToolConstructor:
    """A tool author is offered public fields, and no resolver internals."""

    def test_it_accepts_only_the_public_fields(self) -> None:
        assert [f.name for f in dataclasses.fields(MCPFunctionTool) if f.init] == [
            "name",
            "description",
            "handler",
            "input_schema",
            "title",
            "annotations",
            "output_schema",
            "meta",
        ]

    @pytest.mark.parametrize(
        "field_name",
        ["resolved_params", "resolver_plans", "_resolved_params", "_resolver_plans"],
    )
    def test_resolver_metadata_cannot_be_passed_to_it(self, field_name: str) -> None:
        # Both the names the branch published and the private names replacing
        # them: re-adding either as a constructor argument must fail here.
        with pytest.raises(TypeError):
            MCPFunctionTool("paint", "Paint a room.", _handler, **{field_name: {}})  # type: ignore[arg-type]

    def test_the_decorator_still_fills_them(self) -> None:
        """They left the constructor, not the tool: ``mcp_tool`` is their only producer."""

        @mcp_tool
        def paint(room: str, colour: Annotated[_Colour, Resolve(_pick_colour)]) -> str:
            """Paint a room."""
            return f"painted {room} {colour.answer}"

        assert list(paint._resolved_params) == ["colour"]
        # A resolved parameter is filled by its resolver, never advertised.
        assert list(paint.input_schema.get("properties") or ()) == ["room"]

    def test_direct_construction_stays_positional_across_its_leading_fields(self) -> None:
        tool = MCPFunctionTool("paint", "Paint a room.", _handler, {"type": "object"})

        assert (tool.name, tool.description, tool.input_schema) == ("paint", "Paint a room.", {"type": "object"})

    def test_no_constructor_parameter_names_a_type_private_to_the_sdk(self) -> None:
        # ``_ResolverPlan`` is private to ``mcp``; the branch stood ``Mapping[Hashable,
        # Any]`` in for it rather than name it. Neither may reach the signature.
        annotations = [str(p.annotation) for p in signature(MCPFunctionTool.__init__).parameters.values()]

        assert [a for a in annotations if "ResolverPlan" in a or "Hashable" in a] == []


def test_the_answer_policy_grants_nothing_by_default() -> None:
    """Connecting to a server the operator does not control hands it nothing implicitly."""
    policy = MCPAnswerPolicy()

    assert (policy.elicitation, policy.sampling, tuple(policy.roots)) == ("decline", False, ())


class TestTheServerConfigsAreKeywordOnly:
    """Appending a field must never silently reassign someone's positional argument."""

    def test_the_remote_config_rejects_positional_construction(self) -> None:
        with pytest.raises(TypeError):
            MCPServerConfig("https://example.com/mcp")  # type: ignore[misc]

    def test_the_stdio_config_rejects_positional_construction(self) -> None:
        with pytest.raises(TypeError):
            MCPStdioServerConfig("some-mcp-binary")  # type: ignore[misc]

    def test_by_keyword_they_work_as_before(self) -> None:
        remote = MCPServerConfig(server_url="https://example.com/mcp")
        stdio = MCPStdioServerConfig(command="some-mcp-binary", args=["--flag"])

        assert (remote.server_url, remote.protocol_mode) == ("https://example.com/mcp", "legacy")
        assert (stdio.command, stdio.args, stdio.protocol_mode) == ("some-mcp-binary", ["--flag"], "legacy")


def test_the_private_mcp_modules_the_toolkit_depends_on_still_exist() -> None:
    """Both are imported at module scope on the eager ``ag2.tools`` path, so a rename must fail here."""
    assert callable(run_input_required_driver)
    assert callable(negotiate_auto)


def _handler(arguments: dict[str, object], request_context: object) -> str:
    return "painted"


class _Colour(BaseModel):
    answer: str


def _pick_colour() -> Elicit[_Colour]:
    return Elicit("What colour?", _Colour)
