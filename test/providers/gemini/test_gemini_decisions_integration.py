# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end smoke tests against the real Gemini Decisions API.

The Decisions API is decision-only, so the generic ``test/providers/agent`` matrix
(tools, streaming, free text) does not apply; each question type is covered here
instead. Skipped without ``GEMINI_API_KEY``; excluded from ``just test`` by the
``gemini`` mark.
"""

import io
import math
import os
import struct
import wave
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Annotated

import pytest

from ag2 import Agent, ImageInput, ResponseSchema
from ag2.config import GeminiDecisionsConfig
from ag2.events import BinaryInput, BinaryType
from ag2.response import Question

pytestmark = [pytest.mark.gemini, pytest.mark.asyncio]

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


@dataclass
class Triage:
    is_refund: Annotated[bool, Question("Is the customer asking for a refund?")]
    department: Department


@dataclass
class TicketTriage:
    is_refund: Annotated[bool, Question("Is the customer asking for a refund?")]
    is_angry: Annotated[bool, Question("Is the customer angry or frustrated?")]
    severity: Annotated[Severity, Question("How severe is the customer's problem?")]


def _tone_wav(seconds: float = 1.0, hz: float = 440.0, rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(
            b"".join(
                struct.pack("<h", int(12000 * math.sin(2 * math.pi * hz * i / rate)))
                for i in range(int(seconds * rate))
            )
        )
    return buffer.getvalue()


def _invoice_pdf() -> bytes:
    image_module = pytest.importorskip("PIL.Image")
    draw_module = pytest.importorskip("PIL.ImageDraw")
    image = image_module.new("RGB", (600, 200), "white")
    draw = draw_module.Draw(image)
    draw.text((20, 40), "INVOICE #1042", fill="black")
    draw.text((20, 90), "Total due: 42.00 USD", fill="black")
    draw.text((20, 140), "Status: PAID", fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PDF")
    return buffer.getvalue()


@pytest.fixture()
def decisions_config() -> GeminiDecisionsConfig:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set")
    return GeminiDecisionsConfig(api_key=api_key)


async def test_enum_is_a_choice(decisions_config: GeminiDecisionsConfig) -> None:
    router = Agent("router", config=decisions_config, response_schema=Department)

    reply = await router.ask("I was charged twice for my order.")

    assert await reply.content() is Department.BILLING
    assert set(reply.response.metadata["probabilities"]) == {"billing", "technical", "shipping"}
    assert reply.response.usage.prompt_tokens


async def test_bool_is_a_noul(decisions_config: GeminiDecisionsConfig) -> None:
    guard = Agent(
        "guard",
        config=decisions_config,
        response_schema=ResponseSchema(bool, description="Is the customer asking for a refund?"),
    )

    reply = await guard.ask("You billed me twice for the same month. I want that money back.")

    assert await reply.content() is True
    assert 0.5 <= reply.response.metadata["noul"] <= 1


async def test_int_enum_is_a_score(decisions_config: GeminiDecisionsConfig) -> None:
    grader = Agent(
        "grader",
        config=decisions_config,
        response_schema=ResponseSchema(Severity, description="How severe is this issue?"),
    )

    reply = await grader.ask("The whole app crashes on launch for every user; nothing works.")

    assert await reply.content() is Severity.FULLY_BLOCKED
    assert len(reply.response.metadata["probabilities"]) == 3


async def test_class_is_several_questions(decisions_config: GeminiDecisionsConfig) -> None:
    triage = Agent("triage", config=decisions_config, response_schema=Triage)

    reply = await triage.ask("You billed me twice for the same month. I want that money back.")

    assert await reply.content() == Triage(is_refund=True, department=Department.BILLING)


async def test_image_input(decisions_config: GeminiDecisionsConfig) -> None:
    inspector = Agent(
        "inspector",
        config=decisions_config,
        response_schema=ResponseSchema(bool, description="Is this image mostly red?"),
    )

    reply = await inspector.ask("Inspect this image.", ImageInput(data=RED_PIXEL, media_type="image/png"))

    assert await reply.content() is True


async def test_https_image_uri(decisions_config: GeminiDecisionsConfig) -> None:
    inspector = Agent(
        "inspector",
        config=decisions_config,
        response_schema=ResponseSchema(bool, description="Is this image a logo containing the word 'python'?"),
    )

    reply = await inspector.ask("Look at this.", ImageInput("https://www.python.org/static/img/python-logo.png"))

    assert await reply.content() is True


async def test_pdf_document(decisions_config: GeminiDecisionsConfig) -> None:
    checker = Agent(
        "checker",
        config=decisions_config,
        response_schema=ResponseSchema(bool, description="Does this invoice say it has been paid?"),
    )

    reply = await checker.ask(BinaryInput(_invoice_pdf(), media_type="application/pdf", kind=BinaryType.DOCUMENT))

    assert await reply.content() is True


async def test_audio(decisions_config: GeminiDecisionsConfig) -> None:
    listener = Agent(
        "listener",
        config=decisions_config,
        response_schema=ResponseSchema(bool, description="Does this recording contain human speech?"),
    )

    reply = await listener.ask(BinaryInput(_tone_wav(), media_type="audio/wav", kind=BinaryType.AUDIO))

    assert await reply.content() is False


async def test_history_reaches_the_next_turn(decisions_config: GeminiDecisionsConfig) -> None:
    agent = Agent(
        "watcher",
        config=decisions_config,
        response_schema=ResponseSchema(
            bool, description="Has the customer, anywhere in the conversation, said they were charged twice?"
        ),
    )

    first = await agent.ask("Hello, I was charged twice last month.")
    reply = await first.ask("Also, the app logo looks blurry.")

    assert await reply.content() is True


async def test_question_instructions_follow_the_prompt(decisions_config: GeminiDecisionsConfig) -> None:
    screener = Agent(
        "screener",
        prompt="You screen support tickets.",
        config=decisions_config.copy(
            question_instructions="Only answer yes if the message mentions a specific order number."
        ),
        response_schema=ResponseSchema(bool, description="Should this ticket be escalated?"),
    )

    reply = await screener.ask("My order is late!!!")

    assert await reply.content() is False


async def test_class_mixes_question_types(decisions_config: GeminiDecisionsConfig) -> None:
    triage = Agent("triage", config=decisions_config, response_schema=TicketTriage)

    reply = await triage.ask(
        "This is the THIRD time I write!! The export crashes every single time, no way around it, and I want a refund."
    )

    assert await reply.content() == TicketTriage(is_refund=True, is_angry=True, severity=Severity.FULLY_BLOCKED)
    assert reply.response.usage.completion_tokens == 0
