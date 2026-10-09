# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end smoke tests against the real OpenAI Decisions API.

The Decisions API is decision-only, so the generic ``test/providers/agent`` matrix
(tools, streaming, free text) does not apply; each question type is covered here
instead. Skipped without ``OPENAI_API_KEY``; excluded from ``just test`` by the
``openai`` mark.
"""

import os
from enum import Enum, IntEnum

import pytest

from ag2 import Agent, ImageInput, ResponseSchema
from ag2.config import OpenAIDecisionsConfig

pytestmark = [pytest.mark.openai, pytest.mark.asyncio]

# A 1x1 red PNG.
RED_PIXEL = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de"
    "0000000c4944415408d763f8cfc000000301010018dd8db00000000049454e44ae426082"
)


class Department(Enum):
    """Which department should handle this complaint?"""

    BILLING = "billing"
    """Payments, invoices, and refunds."""
    TECHNICAL = "technical"
    """Problems using the product."""
    SHIPPING = "shipping"
    """Delivery and tracking."""


class Severity(IntEnum):
    COSMETIC = 0
    """Appearance only; no lost functionality."""
    WORKAROUND_AVAILABLE = 1
    """A task fails, but another way works."""
    FULLY_BLOCKED = 2
    """A task fails with no workaround."""


@pytest.fixture()
def decisions_config() -> OpenAIDecisionsConfig:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    return OpenAIDecisionsConfig(api_key=api_key)


async def test_enum_is_a_choice(decisions_config: OpenAIDecisionsConfig) -> None:
    router = Agent("router", config=decisions_config, response_schema=Department)

    reply = await router.ask("I was charged twice for my order.")

    assert await reply.content() is Department.BILLING
    assert {p["value"] for p in reply.response.metadata["probabilities"]} == {"billing", "technical", "shipping"}
    assert reply.response.usage.prompt_tokens


async def test_bool_is_a_predicate(decisions_config: OpenAIDecisionsConfig) -> None:
    guard = Agent(
        "guard",
        config=decisions_config,
        response_schema=ResponseSchema(bool, description="Is the customer asking for a refund?"),
    )

    reply = await guard.ask("You billed me twice for the same month. I want that money back.")

    assert await reply.content() is True
    assert 0.5 <= reply.response.metadata["probability"] <= 1


async def test_int_enum_is_a_score(decisions_config: OpenAIDecisionsConfig) -> None:
    grader = Agent(
        "grader",
        config=decisions_config,
        response_schema=ResponseSchema(Severity, description="How severe is this issue?"),
    )

    reply = await grader.ask("The whole app crashes on launch for every user; nothing works.")

    assert await reply.content() is Severity.FULLY_BLOCKED
    assert len(reply.response.metadata["probabilities"]) == 3


async def test_image_input(decisions_config: OpenAIDecisionsConfig) -> None:
    inspector = Agent(
        "inspector",
        config=decisions_config,
        response_schema=ResponseSchema(bool, description="Is this image mostly red?"),
    )

    reply = await inspector.ask("Inspect this image.", ImageInput(data=RED_PIXEL, media_type="image/png"))

    assert await reply.content() is True
