# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from prometheus_client import CollectorRegistry

from autogen.beta import Agent
from autogen.beta.events import ModelMessage, ModelResponse
from autogen.beta.middleware import MetricsMiddleware
from autogen.beta.testing import TestConfig


@pytest.fixture
def registry() -> CollectorRegistry:
    return CollectorRegistry()


@pytest.mark.asyncio()
async def test_metrics(registry: CollectorRegistry) -> None:
    agent = Agent(
        "assistant",
        config=TestConfig(ModelResponse(ModelMessage("Hello!"))),
        middleware=[MetricsMiddleware(registry=registry)],
    )

    await agent.ask("Hi")

    assert registry.get_sample_value("ag2_llm_calls_total") == 1.0
