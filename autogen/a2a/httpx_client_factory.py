# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from uuid import uuid4

from a2a.types import AgentCapabilities, AgentCard, DataPart, Message, Part, Role, SendMessageSuccessResponse, TextPart
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH, EXTENDED_AGENT_CARD_PATH, PREV_AGENT_CARD_WELL_KNOWN_PATH
from httpx import MockTransport, Request, Response

from autogen.doc_utils import export_module
from autogen.remote.httpx_client_factory import HttpxClientFactory


@export_module("autogen.a2a")
def MockClient(  # noqa: N802
    response_message: str | dict[str, Any] | TextPart | DataPart | Part,
) -> HttpxClientFactory:
    """Create a mock HTTP client for testing A2A agent interactions.

    This function creates a mock HTTP client that simulates responses from an A2A agent server.
    It handles both agent card requests and message sending requests with configurable responses.

    Args:
        response_message: The message to return in response to SendMessage requests.

    Returns:
        An HttpxClientFactory configured with a mock transport that handles requests
        to agent card endpoints and message sending endpoints.

    Example:
        >>> client = MockClient("Hello, world!")
        >>> agent = A2aRemoteAgent(name="remote", url="http://fake", client=client)
    """
    if isinstance(response_message, Part):
        parts = [response_message]
    elif isinstance(response_message, (DataPart, TextPart)):
        parts = [Part(root=response_message)]
    elif isinstance(response_message, str):
        parts = [Part(root=DataPart(data={"role": "assistant", "content": response_message}))]
    elif isinstance(response_message, dict):
        parts = [Part(root=DataPart(data={"role": "assistant", **response_message}))]
    else:
        raise ValueError(f"Invalid message type: {type(response_message)}")

    async def mock_handler(request: Request) -> Response:
        if (
            request.url.path == AGENT_CARD_WELL_KNOWN_PATH
            or request.url.path == EXTENDED_AGENT_CARD_PATH
            or request.url.path == PREV_AGENT_CARD_WELL_KNOWN_PATH
        ):
            return Response(
                status_code=200,
                content=AgentCard(
                    capabilities=AgentCapabilities(streaming=False),
                    default_input_modes=["text"],
                    default_output_modes=["text"],
                    name="mock_agent",
                    description="mock_agent",
                    url="http://localhost:8000",
                    supports_authenticated_extended_card=False,
                    version="0.1.0",
                    skills=[],
                ).model_dump_json(),
            )

        return Response(
            status_code=200,
            content=SendMessageSuccessResponse(
                result=Message(
                    message_id=str(uuid4()),
                    role=Role.agent,
                    parts=parts,
                ),
            ).model_dump_json(),
        )

    return HttpxClientFactory(transport=MockTransport(handler=mock_handler))
