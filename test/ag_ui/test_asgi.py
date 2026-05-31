# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import json

import pytest
from ag_ui.core import UserMessage
from dirty_equals import IsInt, IsPartialDict

from autogen import ConversableAgent
from autogen.ag_ui import AGUIStream
from autogen.testing import TestAgent
from test.ag_ui.utils import assert_event_type, create_run_input

try:
    from starlette.applications import Starlette
    from starlette.endpoints import HTTPEndpoint
    from starlette.routing import Route
    from starlette.testclient import TestClient

    starlette = True
except ImportError:
    starlette = False

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not starlette, reason="starlette not installed"),
]


class TestASGIEndpoint:
    async def test_build_asgi_creates_endpoint(self) -> None:
        agent = ConversableAgent("test_agent")

        stream = AGUIStream(agent)
        endpoint_class = stream.build_asgi()

        assert issubclass(endpoint_class, HTTPEndpoint)

    async def test_asgi_endpoint_handles_request(self) -> None:
        agent = ConversableAgent("test_agent")

        with TestAgent(agent, ["Hello from ASGI!"]):
            stream = AGUIStream(agent)
            endpoint_class = stream.build_asgi()

            run_input = create_run_input(UserMessage(id="msg_1", content="Hello!"))

            app = Starlette(routes=[Route("/", endpoint_class)])
            client = TestClient(app)

            response = client.post(
                "/",
                content=run_input.model_dump_json(),
                headers={
                    "Content-Type": "application/json",
                },
            )

            assert response.status_code == 200

            events = []
            for line in response.text.split("\n"):
                line = line.strip()
                if line.startswith("data: "):
                    events.append(json.loads(line.removeprefix("data: ")))
                elif line and not line.startswith(":"):
                    with contextlib.suppress(json.JSONDecodeError):
                        events.append(json.loads(line))

            run_started = assert_event_type(events, "RUN_STARTED")
            assert run_started == IsPartialDict({
                "type": "RUN_STARTED",
                "threadId": run_input.thread_id,
                "runId": run_input.run_id,
                "timestamp": IsInt(),
            })
            run_finished = assert_event_type(events, "RUN_FINISHED")
            assert run_finished == IsPartialDict({
                "type": "RUN_FINISHED",
                "threadId": run_input.thread_id,
                "runId": run_input.run_id,
                "timestamp": IsInt(),
            })
