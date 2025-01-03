# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
#!/usr/bin/env python3 -m pytest

import asyncio

import pytest

from autogen.agentchat.contrib.magentic_one.websurfer import (
    MultimodalWebSurfer,
)


@pytest.mark.asyncio
async def test_websurfer_interaction():
    """Test the MultimodalWebSurfer."""
    websurfer = MultimodalWebSurfer(name="WebSurfer", description="A web surfing agent.")
    await websurfer.init()

    task = "Visit https://example.com and summarize the page."
    response = await websurfer.a_generate_reply(messages=[{"role": "user", "content": task}])
    assert response == "placeholder"
    await websurfer._reset()
    if hasattr(websurfer, "_web_controller") and websurfer._web_controller:
        await websurfer._web_controller.close()
