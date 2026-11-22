# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""An ACP agent's permission request, answered from an AG-UI client.

End to end across both boundaries: a served parent delegates to a worker
driving an ACP agent, the ACP agent asks over a duplex connection, the served
turn puts the question to the client as an approval interrupt, the client
resumes it, and the ACP agent receives the outcome. The cases are the ones
where a yes or no and a picked option could be taken for each other.
"""

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("acp")

import acp  # noqa: E402
from acp import schema  # noqa: E402

from ag2 import Agent  # noqa: E402
from ag2.acp.testing import duplex_acp_config  # noqa: E402
from ag2.ag_ui import AGUIStream, ChoicePayload  # noqa: E402
from ag2.events import ToolCallEvent  # noqa: E402
from ag2.testing import TestConfig  # noqa: E402
from test.acp.test_bridge import ALLOW, ALLOW_ALWAYS, REJECT, CallingAgent, selected  # noqa: E402
from test.ag_ui.serving import answer, app_for, ask_once, post_run, run_body  # noqa: E402

pytestmark = pytest.mark.asyncio


async def outcome_for(
    workspace: Path, options: list[schema.PermissionOption], payload: str | bool | ChoicePayload
) -> tuple[dict[str, Any], schema.AllowedOutcome | schema.DeniedOutcome]:
    """Serve an ACP agent asking permission with `options`, resume with `payload`.

    Returns the interrupt the client was shown and the outcome the agent got.
    """
    outcome: list[schema.RequestPermissionResponse] = []

    async def act(conn: acp.Client, sid: str) -> schema.RequestPermissionResponse:
        tool_call = schema.ToolCallUpdate(tool_call_id="tc1", title="git status", kind="execute")
        return await conn.request_permission(session_id=sid, options=options, tool_call=tool_call)

    cfg = duplex_acp_config(lambda conn: CallingAgent(conn, act, outcome), cwd=str(workspace))
    try:
        parent = Agent(
            "parent",
            config=TestConfig(ToolCallEvent(name="task_coder", arguments='{"objective": "check"}'), "done"),
            tools=[Agent("coder", config=cfg).as_tool(description="Delegate to the coder.")],
        )
        app = app_for(AGUIStream(parent))
        interrupt = await ask_once(app)
        await post_run(app, run_body(thread_id="t1", run_id="r2", text=None, resume=answer(interrupt, payload)))
    finally:
        await cfg.aclose()
    [response] = outcome
    return interrupt, response.outcome


async def test_the_client_is_offered_the_agents_options(tmp_path: Path) -> None:
    interrupt, _ = await outcome_for(tmp_path, [ALLOW_ALWAYS, ALLOW, REJECT], False)

    assert interrupt["responseSchema"]["anyOf"][1]["properties"]["choice"]["oneOf"] == [
        {"const": "opt-b", "title": "Allow for this session"},
        {"const": "opt-a", "title": "Allow"},
        {"const": "opt-c", "title": "Reject"},
    ]


@pytest.mark.parametrize(
    ("options", "payload", "outcome"),
    [
        # An explicit pick: `{"choice": id}` selects exactly that option.
        pytest.param([ALLOW_ALWAYS, ALLOW, REJECT], {"choice": "opt-b"}, selected("opt-b"), id="picked option"),
        pytest.param([ALLOW_ALWAYS, ALLOW, REJECT], True, selected("opt-a"), id="approve button: one-time allow"),
        pytest.param([ALLOW_ALWAYS, ALLOW, REJECT], False, selected("opt-c"), id="refuse button"),
        # The refuse button reaches the asker as "n": it must not select an allow option with that id.
        pytest.param(
            [ALLOW_ALWAYS.model_copy(update={"option_id": "n"}), ALLOW, REJECT],
            False,
            selected("opt-c"),
            id="refuse button, an allow option with id 'n'",
        ),
        # The approve button reaches the asker as "y": it must not select an allow_always with that id.
        pytest.param(
            [ALLOW_ALWAYS.model_copy(update={"option_id": "y"}), ALLOW, REJECT],
            True,
            selected("opt-a"),
            id="approve button, an allow_always option with id 'y'",
        ),
        # Picking a rejecting option whose id reads as a yes must select that option, not allow.
        pytest.param(
            [ALLOW, REJECT.model_copy(update={"option_id": "y"})],
            {"choice": "y"},
            selected("y"),
            id="picked rejection with id 'y'",
        ),
        # A legacy string answer keeps its meaning in words, whatever ids the agent offers.
        pytest.param(
            [ALLOW_ALWAYS.model_copy(update={"option_id": "no"}), ALLOW, REJECT],
            "no",
            selected("opt-c"),
            id="refusal in words, an allow option with id 'no'",
        ),
        pytest.param(
            [ALLOW, REJECT.model_copy(update={"option_id": "y"})],
            "y",
            selected("opt-a"),
            id="yes in words, a reject option with id 'y'",
        ),
        pytest.param([ALLOW_ALWAYS, ALLOW, REJECT], "opt-b", selected("opt-c"), id="an id as a string is not a pick"),
    ],
)
async def test_the_agent_gets_the_option_the_client_chose(
    tmp_path: Path,
    options: list[schema.PermissionOption],
    payload: str | bool | ChoicePayload,
    outcome: schema.AllowedOutcome | schema.DeniedOutcome,
) -> None:
    _, received = await outcome_for(tmp_path, options, payload)

    assert received == outcome
