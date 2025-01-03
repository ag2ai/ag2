# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
#!/usr/bin/env python3 -m pytest

import asyncio

import pytest

from autogen.agentchat import ConversableAgent
from autogen.agentchat.contrib.magentic_one.coder_agent import create_coder_agent
from autogen.agentchat.contrib.magentic_one.filesurfer_agent import (
    create_file_surfer,
)
from autogen.agentchat.contrib.magentic_one.orchestrator_agent import (
    OrchestratorAgent,
)
from autogen.agentchat.contrib.magentic_one.websurfer import (
    MultimodalWebSurfer,
)


@pytest.mark.asyncio
async def test_orchestrator_with_websurfer():
    """Test the OrchestratorAgent with a MultimodalWebSurfer."""
    coder = create_coder_agent("Coder")
    filesurfer = create_file_surfer("FileSurfer")
    websurfer = MultimodalWebSurfer(name="WebSurfer", description="A web surfing agent.")
    await websurfer.init()
    orchestrator = OrchestratorAgent(name="Orchestrator", agents=[websurfer, coder, filesurfer])

    task = "Visit https://example.com and summarize the page."
    response = await orchestrator.a_generate_reply(messages=[{"role": "user", "content": task}])
    assert response == "placeholder"
    assert isinstance(orchestrator._agents[0], ConversableAgent)
    assert isinstance(orchestrator._agents[0], MultimodalWebSurfer)
    await websurfer._reset()
