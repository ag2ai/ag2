# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""What an ACP agent can ask of AG2, as the agent sees it: files, terminals and permission."""

import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import acp
import pytest
from acp import RequestError, schema

from ag2 import Agent
from ag2.acp import PermissionPolicy
from ag2.acp.events import ACPPermissionOption, ACPPermissionRequest, ACPToolLocation
from ag2.acp.testing import duplex_acp_config
from ag2.events import ApprovalChoice, ApprovalChoiceMessage, HumanInputRequest, HumanMessage
from ag2.hitl import HumanHook

SESSION_ID = "calling-session"

Act = Callable[[acp.Client, str], Awaitable[Any]]


class CallingAgent:
    """An ACP agent whose one prompt turn is a script of requests to the client."""

    def __init__(self, conn: acp.Client, act: Act, outcome: list[Any]) -> None:
        self._conn = conn
        self._act = act
        self._outcome = outcome

    async def initialize(self, **kwargs: Any) -> schema.InitializeResponse:
        return schema.InitializeResponse(protocol_version=acp.PROTOCOL_VERSION)

    async def new_session(self, **kwargs: Any) -> schema.NewSessionResponse:
        return schema.NewSessionResponse(session_id=SESSION_ID)

    async def prompt(self, **kwargs: Any) -> schema.PromptResponse:
        try:
            self._outcome.append(await self._act(self._conn, SESSION_ID))
        except RequestError as error:
            self._outcome.append(error)
        return schema.PromptResponse(stop_reason="end_turn")


async def agent_asks(
    act: Act,
    workspace: Path,
    permission_policy: PermissionPolicy = "ask",
    hitl_hook: HumanHook | None = None,
) -> Any:
    """Run one turn in which the agent does ``act``, and return what ``act`` returned.

    A request the client refuses comes back as the ``RequestError`` the agent got.
    """
    outcome: list[Any] = []
    cfg = duplex_acp_config(
        lambda conn: CallingAgent(conn, act, outcome),
        cwd=str(workspace),
        permission_policy=permission_policy,
    )
    try:
        await Agent("acp", config=cfg, hitl_hook=hitl_hook).ask("go")
    finally:
        await cfg.aclose()
    [result] = outcome
    return result


@pytest.mark.asyncio
class TestFiles:
    async def test_a_file_the_agent_writes_can_be_read_back(self, tmp_path: Path) -> None:
        async def act(conn: acp.Client, sid: str) -> str:
            await conn.write_text_file(session_id=sid, path="notes.txt", content="hi\nthere\n")
            return (await conn.read_text_file(session_id=sid, path="notes.txt")).content

        assert await agent_asks(act, tmp_path) == "hi\nthere\n"
        assert (tmp_path / "notes.txt").read_text() == "hi\nthere\n"

    async def test_an_absolute_path_inside_the_workspace_is_fine(self, tmp_path: Path) -> None:
        (tmp_path / "notes.txt").write_text("hi\n")

        async def act(conn: acp.Client, sid: str) -> str:
            return (await conn.read_text_file(session_id=sid, path=str(tmp_path / "notes.txt"))).content

        assert await agent_asks(act, tmp_path) == "hi\n"

    async def test_a_read_can_ask_for_a_window_of_lines(self, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("l1\nl2\nl3\nl4\n")

        async def act(conn: acp.Client, sid: str) -> str:
            return (await conn.read_text_file(session_id=sid, path="f.txt", line=2, limit=2)).content

        assert await agent_asks(act, tmp_path) == "l2\nl3\n"

    @pytest.mark.parametrize("path", ["../../etc/passwd", "/etc/passwd"])
    async def test_a_path_outside_the_workspace_is_refused(self, tmp_path: Path, path: str) -> None:
        async def act(conn: acp.Client, sid: str) -> str:
            return (await conn.read_text_file(session_id=sid, path=path)).content

        assert isinstance(await agent_asks(act, tmp_path), RequestError)

    async def test_a_write_outside_the_workspace_is_refused(self, tmp_path: Path) -> None:
        workspace = tmp_path / "work"
        workspace.mkdir()

        async def act(conn: acp.Client, sid: str) -> None:
            await conn.write_text_file(session_id=sid, path="../escaped.txt", content="x")

        assert isinstance(await agent_asks(act, workspace), RequestError)
        assert not (tmp_path / "escaped.txt").exists()


# Opaque ids: nothing may be inferred from an id that spells out its kind.
ALLOW = schema.PermissionOption(option_id="opt-a", kind="allow_once", name="Allow")
ALLOW_ALWAYS = schema.PermissionOption(option_id="opt-b", kind="allow_always", name="Allow for this session")
REJECT = schema.PermissionOption(option_id="opt-c", kind="reject_once", name="Reject")
REJECT_ALWAYS = schema.PermissionOption(option_id="opt-d", kind="reject_always", name="Never")

# Gemini CLI's order (acpUtils.ts::toPermissionOptions): the remembered option first.
GEMINI_ORDER = [ALLOW_ALWAYS, ALLOW, REJECT]

CANCELLED = schema.DeniedOutcome(outcome="cancelled")


def selected(option_id: str) -> schema.AllowedOutcome:
    return schema.AllowedOutcome(option_id=option_id, outcome="selected")


@pytest.mark.asyncio
class TestPermission:
    TOOL_CALL = schema.ToolCallUpdate(
        tool_call_id="tc1",
        title="git status",
        kind="execute",
        locations=[schema.ToolCallLocation(path="/repo/a.py", line=3)],
        raw_input={"command": "git status"},
    )

    async def ask_permission(
        self,
        workspace: Path,
        policy: PermissionPolicy,
        options: list[schema.PermissionOption],
        hitl_hook: HumanHook | None = None,
        tool_call: schema.ToolCallUpdate = TOOL_CALL,
    ) -> Any:
        async def act(conn: acp.Client, sid: str) -> schema.RequestPermissionResponse:
            return await conn.request_permission(session_id=sid, options=options, tool_call=tool_call)

        response = await agent_asks(act, workspace, permission_policy=policy, hitl_hook=hitl_hook)
        return response.outcome

    async def test_the_human_is_asked_with_the_agents_request(self, tmp_path: Path) -> None:
        asked: list[HumanInputRequest] = []

        def hook(event: HumanInputRequest) -> HumanMessage:
            asked.append(event)
            return ApprovalChoiceMessage("opt-c")

        await self.ask_permission(tmp_path, "ask", GEMINI_ORDER, hook)

        [request] = asked
        assert request == ACPPermissionRequest(
            "git status",
            tool_call_id="tc1",
            options=[
                ACPPermissionOption("opt-b", "Allow for this session", "allow_always"),
                ACPPermissionOption("opt-a", "Allow", "allow_once"),
                ACPPermissionOption("opt-c", "Reject", "reject_once"),
            ],
            kind="execute",
            locations=[ACPToolLocation("/repo/a.py", 3)],
            raw_input={"command": "git status"},
            choices=[
                ApprovalChoice("opt-b", "Allow for this session"),
                ApprovalChoice("opt-a", "Allow"),
                ApprovalChoice("opt-c", "Reject"),
            ],
        )
        # Not part of event equality (compare=False), so checked on its own.
        assert isinstance(request, ACPPermissionRequest)
        assert request.tool_call_id == "tc1"

    async def test_an_untitled_call_is_still_asked_about(self, tmp_path: Path) -> None:
        asked: list[HumanInputRequest] = []

        def hook(event: HumanInputRequest) -> str:
            asked.append(event)
            return "no"

        await self.ask_permission(
            tmp_path, "ask", GEMINI_ORDER, hook, tool_call=schema.ToolCallUpdate(tool_call_id="tc1")
        )

        assert [event.content for event in asked] == ["Allow this action?"]

    @pytest.mark.parametrize(
        ("reply", "options", "outcome"),
        [
            pytest.param(ApprovalChoiceMessage("opt-a"), GEMINI_ORDER, selected("opt-a"), id="picked: allow_once"),
            pytest.param(ApprovalChoiceMessage("opt-b"), GEMINI_ORDER, selected("opt-b"), id="picked: allow_always"),
            pytest.param(ApprovalChoiceMessage("opt-c"), GEMINI_ORDER, selected("opt-c"), id="picked: reject_once"),
            pytest.param(
                ApprovalChoiceMessage("opt-d"), [ALLOW, REJECT_ALWAYS], selected("opt-d"), id="picked: reject_always"
            ),
            # A picked id is that option, even when it reads as an approval word.
            pytest.param(
                ApprovalChoiceMessage("y"),
                [ALLOW, REJECT.model_copy(update={"option_id": "y"})],
                selected("y"),
                id="picked rejection whose id is 'y'",
            ),
            pytest.param(
                ApprovalChoiceMessage("opt-unknown"), [ALLOW, REJECT], selected("opt-c"), id="picked unknown id"
            ),
            # GHSA-3r6w-7pcg-c4x2: a plain yes was sent as the agent's first allow
            # option, which Gemini CLI makes allow_always.
            *(
                pytest.param(word, GEMINI_ORDER, selected("opt-a"), id=f"yes word {word!r}")
                for word in ["yes", "Y", "1", " YES ", "ok", "allow", "approve", "allow_once"]
            ),
            *(
                pytest.param(word, GEMINI_ORDER, selected("opt-b"), id=f"always word {word!r}")
                for word in ["always", "Always", "allow_always"]
            ),
            pytest.param("always", [ALLOW, REJECT], selected("opt-a"), id="always, nothing remembered on offer"),
            pytest.param("yes", [ALLOW_ALWAYS, REJECT], selected("opt-c"), id="yes, nothing one-time on offer"),
            pytest.param(
                "y",
                [ALLOW_ALWAYS.model_copy(update={"option_id": "y"}), ALLOW, REJECT],
                selected("opt-a"),
                id="yes word is not an option id",
            ),
            pytest.param(
                "n",
                [ALLOW_ALWAYS.model_copy(update={"option_id": "n"}), ALLOW, REJECT],
                selected("opt-c"),
                id="refusal word is not an option id",
            ),
            *(
                pytest.param(reply, [ALLOW, REJECT_ALWAYS, REJECT], selected("opt-c"), id=f"rejects {reply!r}")
                # A plain answer is a word, never an id: "opt-a" names an option but is not a pick.
                for reply in ["", "no", "n", "maybe", "opt-unknown", "opt-a"]
            ),
        ],
    )
    async def test_the_answer_selects_an_option(
        self, tmp_path: Path, reply: str | HumanMessage, options: list[schema.PermissionOption], outcome: Any
    ) -> None:
        hook: HumanHook = (lambda event: reply) if isinstance(reply, str) else (lambda event: reply)
        assert await self.ask_permission(tmp_path, "ask", options, hook) == outcome

    @pytest.mark.parametrize(
        ("policy", "options", "outcome"),
        [
            pytest.param("auto", GEMINI_ORDER, selected("opt-a"), id="auto prefers allow_once"),
            pytest.param("auto", [ALLOW_ALWAYS, REJECT], selected("opt-b"), id="auto falls back to allow_always"),
            pytest.param("auto", [REJECT], CANCELLED, id="auto with nothing to allow"),
            pytest.param("deny", [ALLOW, REJECT_ALWAYS, REJECT], selected("opt-c"), id="deny prefers reject_once"),
            pytest.param("deny", [ALLOW, REJECT_ALWAYS], selected("opt-d"), id="deny falls back to reject_always"),
            # Nothing to pick, but the answer must still be well-formed.
            pytest.param("deny", [ALLOW_ALWAYS, ALLOW], CANCELLED, id="deny with nothing to reject"),
        ],
    )
    async def test_a_policy_answers_without_asking(
        self, tmp_path: Path, policy: PermissionPolicy, options: list[schema.PermissionOption], outcome: Any
    ) -> None:
        def hook(event: HumanInputRequest) -> str:
            raise AssertionError("a policy must not ask the human")

        assert await self.ask_permission(tmp_path, policy, options, hook) == outcome


@pytest.mark.asyncio
class TestTerminal:
    async def test_a_command_runs_and_its_output_is_captured(self, tmp_path: Path) -> None:
        async def act(conn: acp.Client, sid: str) -> tuple[Any, ...]:
            terminal = await conn.create_terminal(session_id=sid, command="printf", args=["hello"])
            exited = await conn.wait_for_terminal_exit(session_id=sid, terminal_id=terminal.terminal_id)
            output = await conn.terminal_output(session_id=sid, terminal_id=terminal.terminal_id)
            await conn.release_terminal(session_id=sid, terminal_id=terminal.terminal_id)
            return exited, output

        exited, output = await agent_asks(act, tmp_path)

        assert (exited.exit_code, exited.signal) == (0, None)
        assert (output.output, output.truncated) == ("hello", False)

    async def test_output_past_the_byte_limit_is_truncated(self, tmp_path: Path) -> None:
        async def act(conn: acp.Client, sid: str) -> Any:
            terminal = await conn.create_terminal(
                session_id=sid, command="printf", args=["abcdefghij"], output_byte_limit=4
            )
            await conn.wait_for_terminal_exit(session_id=sid, terminal_id=terminal.terminal_id)
            return await conn.terminal_output(session_id=sid, terminal_id=terminal.terminal_id)

        output = await agent_asks(act, tmp_path)

        assert (output.output, output.truncated) == ("abcd", True)

    async def test_a_working_directory_outside_the_workspace_is_refused(self, tmp_path: Path) -> None:
        async def act(conn: acp.Client, sid: str) -> Any:
            return await conn.create_terminal(session_id=sid, command="printf", args=["x"], cwd="../../etc")

        assert isinstance(await agent_asks(act, tmp_path), RequestError)

    async def test_a_running_command_can_be_killed(self, tmp_path: Path) -> None:
        async def act(conn: acp.Client, sid: str) -> Any:
            terminal = await conn.create_terminal(session_id=sid, command="sleep", args=["30"])
            await conn.kill_terminal(session_id=sid, terminal_id=terminal.terminal_id)
            exited = await conn.wait_for_terminal_exit(session_id=sid, terminal_id=terminal.terminal_id)
            await conn.release_terminal(session_id=sid, terminal_id=terminal.terminal_id)
            return exited

        exited = await agent_asks(act, tmp_path)

        if sys.platform == "win32":
            # No signals on Windows: TerminateProcess surfaces as a non-zero exit code.
            assert exited.exit_code is not None
        else:
            assert exited.signal is not None


@pytest.mark.asyncio
class TestUnsupportedClientMethods:
    async def test_an_mcp_message_is_refused_as_an_unknown_method(self, tmp_path: Path) -> None:
        async def act(conn: acp.Client, sid: str) -> Any:
            return await conn.mcp_message(server_id="srv", request_id="1", method="tools/list", session_id=sid)

        error = await agent_asks(act, tmp_path)

        assert isinstance(error, RequestError)
        assert (error.code, error.data) == (-32601, {"method": "mcp/message"})

    async def test_an_auth_status_update_does_not_disturb_the_turn(self, tmp_path: Path) -> None:
        async def act(conn: acp.Client, sid: str) -> Any:
            return await conn.auth_status_update(state="authenticated")

        assert await agent_asks(act, tmp_path) is None
