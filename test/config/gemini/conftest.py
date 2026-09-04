# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest


@pytest.fixture
def gemini_config() -> MagicMock:
    config = MagicMock()
    config.api_key = "test-key"
    return config


@pytest.fixture
def captured_request() -> dict[str, Any]:
    """The body of the last request ``capturing_http_client`` carried, under ``"body"``."""
    return {}


@pytest.fixture
def capturing_http_client(captured_request: dict[str, Any]) -> httpx.AsyncClient:
    """A client that records the outgoing request body and answers ``"ok"``.

    The seam for asserting what ag2 puts on the wire: drive the real Gemini
    client, then read ``captured_request["body"]``.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
            },
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))
