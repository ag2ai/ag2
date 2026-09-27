# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end smoke tests against the real TypeSafe API.

Jev is decision-only, so the generic ``test/providers/agent`` matrix (tools,
streaming, free text) does not apply; each of its three primitives is covered here
instead. Skipped without ``TYPESAFE_API_KEY``; excluded from ``just test`` by the
``typesafe`` mark.
"""

import os
from enum import Enum, IntEnum

import pytest

from ag2 import Agent, ResponseSchema
from ag2.config import TypeSafeConfig

pytestmark = [pytest.mark.typesafe, pytest.mark.asyncio]


class Department(Enum):
    """Which team should handle this ticket?"""

    BILLING = "billing"
    """Payments, invoicing, refunds."""
    TECHNICAL = "technical"
    """Bugs, outages, integrations."""


class Severity(IntEnum):
    LOW = 0
    """Cosmetic; nothing is blocked."""
    MEDIUM = 1
    """Degraded but usable."""
    HIGH = 2
    """An outage, data loss, or a security exposure."""


@pytest.fixture()
def typesafe_config() -> TypeSafeConfig:
    api_key = os.getenv("TYPESAFE_API_KEY")
    if not api_key:
        pytest.skip("TYPESAFE_API_KEY not set")
    return TypeSafeConfig(api_key=api_key)


async def test_enum_is_a_choice(typesafe_config: TypeSafeConfig) -> None:
    router = Agent(
        "router",
        prompt="You triage customer support tickets.",
        config=typesafe_config,
        response_schema=Department,
    )

    reply = await router.ask("Our webhook integration has been returning 502s since Monday.")

    assert await reply.content() is Department.TECHNICAL
    assert set(reply.response.metadata["probabilities"]) == {"billing", "technical"}
    assert reply.response.usage.prompt_tokens


async def test_bool_is_a_yes_no_question(typesafe_config: TypeSafeConfig) -> None:
    guard = Agent(
        "guard",
        config=typesafe_config,
        response_schema=ResponseSchema(bool, description="Is the customer asking for a refund?"),
    )

    reply = await guard.ask("You billed me twice for the same month. I want that money back.")

    assert await reply.content() is True
    assert 0.5 <= reply.response.metadata["noul"] <= 1


async def test_int_enum_is_a_score(typesafe_config: TypeSafeConfig) -> None:
    grader = Agent(
        "grader",
        prompt="How severe is this incident report?",
        config=typesafe_config,
        response_schema=Severity,
    )

    reply = await grader.ask("Customers can see each other's invoices after the last deploy.")

    assert await reply.content() is Severity.HIGH
    assert set(reply.response.metadata["probabilities"]) == {0, 1, 2}
