# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("anthropic")

from a2a.utils.errors import ContentTypeNotSupportedError

from ag2 import Agent, AudioInput
from ag2.a2a import A2AConfig, A2AServer, build_card
from ag2.a2a.extension import MIME_HISTORY
from ag2.a2a.testing import make_test_client_factory
from ag2.config import AnthropicConfig


def test_card_declares_only_what_the_models_mapper_accepts() -> None:
    agent = Agent("srv", config=AnthropicConfig("claude-sonnet-4-5", api_key="k"))

    modes = set(build_card(agent, url="http://test").default_input_modes)

    assert {"image/png", "application/pdf", MIME_HISTORY} <= modes
    assert not modes & {"audio/wav", "video/mp4", "image/heic"}


@pytest.mark.asyncio
async def test_media_the_model_cannot_take_is_refused_at_the_edge() -> None:
    server = A2AServer(
        Agent("srv", config=AnthropicConfig("claude-sonnet-4-5", api_key="k")), validate_input_modes=True
    )
    client = Agent(
        "cli",
        config=A2AConfig(
            card_url="http://test",
            httpx_client_factory=make_test_client_factory(server, url="http://test"),
            streaming=False,
        ),
    )

    with pytest.raises(ContentTypeNotSupportedError, match="audio/wav"):
        await client.ask("listen", AudioInput(data=b"RIFF", media_type="audio/wav"))
