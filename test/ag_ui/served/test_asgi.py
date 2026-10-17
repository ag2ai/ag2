# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""The endpoint `build_asgi` hands a server, mounted and driven over real HTTP.

What the rest of `served/` takes for granted: that the thing returned is an
endpoint a Starlette app will route to, and that posting a `RunAgentInput` at it
comes back as the same run the generator seam would have produced.
"""

import pytest
from dirty_equals import IsInt, IsPartialDict
from starlette.endpoints import HTTPEndpoint

from ag2 import Agent
from ag2.ag_ui import AGUIStream
from ag2.testing import TestConfig
from test.ag_ui.harness import only, types_of
from test.ag_ui.serving import app_for, post_run, run_body

pytestmark = pytest.mark.asyncio


async def test_build_asgi_returns_an_endpoint_a_server_can_route_to() -> None:
    assert issubclass(AGUIStream(Agent("test_agent")).build_asgi(), HTTPEndpoint)


async def test_a_posted_run_comes_back_as_a_stream_of_its_own_events() -> None:
    agent = Agent("test_agent", config=TestConfig("Hello from ASGI!"))

    frames = await post_run(app_for(AGUIStream(agent)), run_body(thread_id="t1", run_id="r1", text="Hello!"))

    assert types_of(frames) == ["RUN_STARTED", "TEXT_MESSAGE_CHUNK", "RUN_FINISHED"]
    identified = IsPartialDict({"threadId": "t1", "runId": "r1", "timestamp": IsInt()})
    assert only(frames, "RUN_STARTED") == identified
    assert only(frames, "RUN_FINISHED") == identified
