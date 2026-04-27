# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta import Agent
from autogen.beta.files import FileProvider, UploadedFile
from autogen.beta.testing import TestConfig


@pytest.mark.asyncio
async def test_uploaded_file_passed_to_agent_ask() -> None:
    agent = Agent("test", config=TestConfig("done"))
    uploaded = UploadedFile(file_id="file-123", filename="doc.pdf", provider=FileProvider.OPENAI)

    reply = await agent.ask("Summarize this", uploaded)

    assert reply.body == "done"
