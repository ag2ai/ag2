# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass, make_dataclass
from enum import Enum, IntEnum
from typing import Annotated, Any

import httpx2
import pytest
from dirty_equals import IsPartialDict

from ag2 import Agent, ImageInput, ResponseSchema
from ag2.config import GeminiDecisionsConfig
from ag2.config.gemini import DecisionBlockedError, GeminiDecisionsClient, UnsupportedResponseSchemaError
from ag2.events import AudioInput, BinaryInput, FileIdInput, Usage
from ag2.exceptions import FilesAPIUnsupportedError, UnsupportedInputError, UnsupportedToolError
from ag2.files.types import FileProvider, UploadedFile
from ag2.response import Question
from ag2.tools import tool

USAGE = {"total_input_tokens": 30, "total_cached_tokens": 4, "total_tokens": 30}


class Department(Enum):
    """Which department should handle this complaint?"""

    BILLING = "billing"
    """Payments, invoices, and refunds."""
    TECHNICAL = "technical"
    """Problems using the product."""


class Severity(IntEnum):
    COSMETIC = 0
    """Appearance only."""
    FULLY_BLOCKED = 1


@dataclass
class Triage:
    is_spam: Annotated[bool, Question("Is this message spam?")]
    department: Department
    severity: Annotated[Severity, Question("How severe is the problem?")]


@tool
def lookup(order_id: str) -> str:
    return order_id


@dataclass
class _FakeAPI:
    requests: list[httpx2.Request]
    body: dict[str, Any]
    status: int

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return httpx2.Response(self.status, json=self.body)


def _config(
    requests: list[httpx2.Request], answers: dict[str, Any] | None = None, *, status: int = 200, **body: Any
) -> GeminiDecisionsConfig:
    payload = {"answers": answers, "usage": USAGE, **body} if answers is not None else {"usage": USAGE, **body}
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(_FakeAPI(requests, payload, status)))
    return GeminiDecisionsConfig(api_key="test-key", http_client=client)


def _body(request: httpx2.Request) -> dict[str, Any]:
    body: dict[str, Any] = json.loads(request.content)
    return body


def test_defaults() -> None:
    config = GeminiDecisionsConfig()

    assert (config.model, config.boolean_threshold) == ("gemini-decision-1", 0.5)
    assert isinstance(config.copy(api_key="k").create(), GeminiDecisionsClient)


def test_copy_overrides_without_mutating_original() -> None:
    config = GeminiDecisionsConfig(boolean_threshold=0.8)

    copy = config.copy(model="gemini-decision-1-preview")

    assert (copy.model, copy.boolean_threshold) == ("gemini-decision-1-preview", 0.8)
    assert config.model == "gemini-decision-1"


def test_no_files_api() -> None:
    with pytest.raises(FilesAPIUnsupportedError):
        GeminiDecisionsConfig().create_files_client()


@pytest.mark.asyncio
async def test_enum_schema_is_a_choice() -> None:
    requests: list[httpx2.Request] = []
    answer = {"type": "choice", "choice": "billing", "confidence": 0.93, "probabilities": {"billing": 0.93}}
    router = Agent(
        "router",
        prompt="You route customer complaints.",
        config=_config(requests, {"answer": answer}),
        response_schema=Department,
    )

    reply = await router.ask("I was charged twice for my order.")

    assert await reply.content() is Department.BILLING
    assert reply.response.metadata == {"choice": "billing", "confidence": 0.93, "probabilities": {"billing": 0.93}}
    assert reply.response.usage == Usage(prompt_tokens=30, total_tokens=30, cache_read_input_tokens=4)

    [request] = requests
    assert request.url.path == "/v1beta/decisions"
    assert request.headers["x-goog-api-key"] == "test-key"
    assert _body(request) == {
        "model": "gemini-decision-1",
        "input": [{"type": "text", "text": "I was charged twice for my order."}],
        "questions": {
            "answer": {
                "type": "choice",
                "instructions": "You route customer complaints.\n\nWhich department should handle this complaint?",
                "criteria": {"billing": "Payments, invoices, and refunds.", "technical": "Problems using the product."},
            }
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(("threshold", "expected"), [(0.5, True), (0.95, False)])
async def test_bool_schema_is_a_noul_thresholded(threshold: float, expected: bool) -> None:
    requests: list[httpx2.Request] = []
    config = _config(requests, {"answer": {"type": "noul", "noul": 0.92, "confidence": 0.92}})
    guard = Agent(
        "guard",
        config=config.copy(boolean_threshold=threshold),
        response_schema=ResponseSchema(bool, description="Is the customer asking for a refund?"),
    )

    reply = await guard.ask("You billed me twice. I want that money back.")

    assert await reply.content() is expected
    assert reply.response.metadata == {"noul": 0.92, "confidence": 0.92}
    assert _body(requests[0])["questions"] == {
        "answer": {"type": "noul", "instructions": "Is the customer asking for a refund?"}
    }


@pytest.mark.asyncio
async def test_class_schema_is_one_question_per_field() -> None:
    requests: list[httpx2.Request] = []
    answers = {
        "is_spam": {"type": "noul", "noul": 0.03, "confidence": 0.97},
        "department": {"type": "choice", "choice": "technical", "confidence": 0.8, "probabilities": {}},
        "severity": {"type": "score", "score": 0.7, "confidence": 0.7, "probabilities": {}, "legend": {}},
    }
    triage = Agent(
        "triage",
        prompt="Triage the ticket.",
        config=_config(requests, answers).copy(question_instructions="Be strict."),
        response_schema=Triage,
    )

    reply = await triage.ask("The export button crashes the app.")

    assert await reply.content() == Triage(
        is_spam=False, department=Department.TECHNICAL, severity=Severity.FULLY_BLOCKED
    )
    assert set(reply.response.metadata) == {"is_spam", "department", "severity"}
    questions = _body(requests[0])["questions"]
    assert questions["is_spam"] == {
        "type": "noul",
        "instructions": "Triage the ticket.\n\nBe strict.\n\nIs this message spam?",
    }
    assert questions["severity"] == {
        "type": "score",
        "instructions": "Triage the ticket.\n\nBe strict.\n\nHow severe is the problem?",
        "criteria": ["Cosmetic: Appearance only.", "Fully blocked"],
    }


@pytest.mark.asyncio
async def test_history_and_media_are_flat_content_blocks() -> None:
    requests: list[httpx2.Request] = []
    guard = Agent(
        "guard",
        config=_config(requests, {"answer": {"type": "noul", "noul": 0.9, "confidence": 0.9}}),
        response_schema=ResponseSchema(bool, description="Is there a cat?"),
    )

    await guard.ask(
        "Look:", ImageInput(data=b"hi", media_type="image/png"), ImageInput("https://example.com/cat.png"), "Well?"
    )

    assert _body(requests[0])["input"] == [
        {"type": "text", "text": "Look:"},
        {"type": "image", "data": "aGk=", "mime_type": "image/png"},
        {"type": "image", "uri": "https://example.com/cat.png", "mime_type": "image/png"},
        {"type": "text", "text": "Well?"},
    ]


@pytest.mark.asyncio
async def test_adjacent_text_blocks_are_separated() -> None:
    # The API concatenates adjacent text blocks with no separator.
    requests: list[httpx2.Request] = []
    guard = Agent(
        "guard",
        config=_config(requests, {"answer": {"type": "noul", "noul": 0.9, "confidence": 0.9}}),
        response_schema=ResponseSchema(bool, description="Is this a refund request?"),
    )

    reply = await guard.ask("first")
    await reply.ask("second")

    assert _body(requests[1])["input"] == [
        {"type": "text", "text": "first"},
        {"type": "text", "text": '\n\n[Assistant]\n{"data": true}'},
        {"type": "text", "text": "\n\nsecond"},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    ["gs://bucket/cat.png", "https://www.youtube.com/watch?v=abc", "https://example.com/files/abc"],
    ids=["gcs", "youtube", "no-extension"],
)
async def test_unsupported_urls_are_rejected_before_a_request(url: str) -> None:
    requests: list[httpx2.Request] = []
    agent = Agent("guard", config=_config(requests, {}), response_schema=ResponseSchema(bool, description="Cat?"))

    with pytest.raises(UnsupportedInputError):
        await agent.ask(ImageInput(url))
    assert requests == []


@pytest.mark.asyncio
async def test_empty_input_is_rejected_before_a_request() -> None:
    requests: list[httpx2.Request] = []
    agent = Agent("guard", config=_config(requests, {}), response_schema=ResponseSchema(bool, description="Cat?"))

    with pytest.raises(ValueError, match="needs input"):
        await agent.ask()
    assert requests == []


@pytest.mark.asyncio
async def test_blocked_input_raises() -> None:
    requests: list[httpx2.Request] = []
    feedback = {"block_reason": "safety", "block_reason_message": "Unsafe content."}
    guard = Agent(
        "guard",
        config=_config(requests, prompt_feedback=feedback),
        response_schema=ResponseSchema(bool, description="Is this fine?"),
    )

    with pytest.raises(DecisionBlockedError, match=r"blocked the input \(safety\): Unsafe content\.") as info:
        await guard.ask("...")

    assert info.value.reason == "safety"


@pytest.mark.asyncio
async def test_blocked_input_http_error_raises() -> None:
    requests: list[httpx2.Request] = []
    error = {"error": {"code": "prohibited_content", "message": "Request blocked."}}
    guard = Agent(
        "guard",
        config=_config(requests, status=400, **error),
        response_schema=ResponseSchema(bool, description="Is this fine?"),
    )

    with pytest.raises(
        DecisionBlockedError, match=r"blocked the input \(prohibited_content\): Request blocked\."
    ) as info:
        await guard.ask("...")

    assert info.value.reason == "prohibited_content"


@pytest.mark.asyncio
async def test_unlisted_choice_raises() -> None:
    requests: list[httpx2.Request] = []
    answer = {"type": "choice", "choice": "shipping", "confidence": 0.9, "probabilities": {}}
    agent = Agent("router", config=_config(requests, {"answer": answer}), response_schema=Department)

    with pytest.raises(ValueError, match="'shipping', not one of its options"):
        await agent.ask("...")


class Lonely(Enum):
    """Which one?"""

    ONLY = "only"


class Padded(Enum):
    """Which one?"""

    A = " a"
    B = "b"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("schema", "match"),
    [
        (Lonely, "needs 2 to 255 options"),
        (Padded, "cannot be an option ID"),
        (make_dataclass("NonAscii", [("sécurité", Annotated[bool, Question("Safe?")])]), "cannot be a question key"),
    ],
    ids=["one-option", "padded-option-id", "non-ascii-key"],
)
async def test_schemas_the_api_rejects_fail_before_a_request(schema: Any, match: str) -> None:
    requests: list[httpx2.Request] = []
    agent = Agent("router", config=_config(requests, {}), response_schema=schema)

    with pytest.raises(UnsupportedResponseSchemaError, match=match):
        await agent.ask("...")
    assert requests == []


@pytest.mark.asyncio
async def test_missing_or_mismatched_answer_raises() -> None:
    requests: list[httpx2.Request] = []
    agent = Agent(
        "router",
        config=_config(requests, {"answer": {"type": "noul", "noul": 0.9, "confidence": 0.9}}),
        response_schema=Department,
    )

    with pytest.raises(ValueError, match=r"answered question 'answer' \(choice\) with 'noul'"):
        await agent.ask("...")


@pytest.mark.asyncio
async def test_tools_are_rejected_before_a_request() -> None:
    requests: list[httpx2.Request] = []
    agent = Agent("router", config=_config(requests, {}), tools=[lookup], response_schema=Department)

    with pytest.raises(UnsupportedToolError):
        await agent.ask("...")
    assert requests == []


@pytest.mark.asyncio
async def test_free_text_schema_is_rejected_before_a_request() -> None:
    requests: list[httpx2.Request] = []
    agent = Agent("writer", config=_config(requests, {}), response_schema=str)

    with pytest.raises(UnsupportedResponseSchemaError, match="decision-only"):
        await agent.ask("...")
    assert requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("media_type", "match"),
    [
        ("application/zip", r"BinaryInput\(binary\)"),
        ("application/octet-stream", r"BinaryInput\(application/octet-stream\)"),
    ],
    ids=["non-media", "generic-type"],
)
async def test_unsupported_binaries_are_rejected(media_type: str, match: str) -> None:
    requests: list[httpx2.Request] = []
    agent = Agent("router", config=_config(requests, {}), response_schema=Department)

    with pytest.raises(UnsupportedInputError, match=match):
        await agent.ask(BinaryInput(b"x", media_type=media_type))
    assert requests == []


class Tier(Enum):
    """Which tier?"""

    BASIC = "basic"
    LEGACY_BASIC = "basic"
    PRO = "pro"


class Level(IntEnum):
    LOW = 0
    MINIMAL = 0
    HIGH = 1


@dataclass
class Grade:
    tier: Tier
    level: Annotated[Level, Question("How high is it?")]


@pytest.mark.asyncio
async def test_enum_aliases_are_one_option() -> None:
    requests: list[httpx2.Request] = []
    answers = {
        "tier": {"type": "choice", "choice": "pro", "confidence": 0.9, "probabilities": {}},
        "level": {"type": "score", "score": 0.9, "confidence": 0.9, "probabilities": {}, "legend": {}},
    }
    agent = Agent(
        "grader",
        config=_config(requests, answers),
        response_schema=Grade,
    )

    reply = await agent.ask("...")

    assert await reply.content() == Grade(tier=Tier.PRO, level=Level.HIGH)
    questions = _body(requests[0])["questions"]
    assert questions["tier"]["criteria"] == {"basic": "", "pro": ""}
    assert questions["level"]["criteria"] == ["Low", "High"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("part", "block"),
    [
        (AudioInput("https://example.com/call.webm"), {"type": "audio", "mime_type": "audio/webm"}),
        (ImageInput("https://example.com/photo.HEIC"), {"type": "image", "mime_type": "image/heic"}),
        (
            FileIdInput("files/abc123", filename="call.wav"),
            {
                "type": "audio",
                "uri": "https://generativelanguage.googleapis.com/v1beta/files/abc123",
                "mime_type": "audio/wav",
            },
        ),
        (
            UploadedFile("files/doc9", filename="report.pdf", provider=FileProvider.GEMINI),
            {
                "type": "document",
                "uri": "https://generativelanguage.googleapis.com/v1beta/files/doc9",
                "mime_type": "application/pdf",
            },
        ),
    ],
    ids=["webm-audio", "heic-image", "files-api-audio", "files-api-pdf"],
)
async def test_mime_type_follows_the_block_kind(part: Any, block: dict[str, str]) -> None:
    requests: list[httpx2.Request] = []
    guard = Agent(
        "guard",
        config=_config(requests, {"answer": {"type": "noul", "noul": 0.9, "confidence": 0.9}}),
        response_schema=ResponseSchema(bool, description="Is this fine?"),
    )

    await guard.ask(part)

    assert _body(requests[0])["input"] == [IsPartialDict(block)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "part",
    [
        FileIdInput("files/abc123"),
        FileIdInput("files/abc123", filename="notes.docx"),
        UploadedFile("file-abc", filename="report.pdf", provider=FileProvider.OPENAI),
    ],
    ids=["no-filename", "unsupported-extension", "other-provider"],
)
async def test_unusable_file_ids_are_rejected_before_a_request(part: FileIdInput) -> None:
    requests: list[httpx2.Request] = []
    agent = Agent("guard", config=_config(requests, {}), response_schema=ResponseSchema(bool, description="Fine?"))

    with pytest.raises(UnsupportedInputError):
        await agent.ask(part)
    assert requests == []


@pytest.mark.asyncio
async def test_blank_text_is_not_input() -> None:
    requests: list[httpx2.Request] = []
    agent = Agent("guard", config=_config(requests, {}), response_schema=ResponseSchema(bool, description="Fine?"))

    with pytest.raises(ValueError, match="needs input"):
        await agent.ask("  ")
    assert requests == []


@pytest.mark.parametrize("threshold", [float("nan"), -0.1, 1.5])
def test_threshold_outside_zero_to_one_is_rejected(threshold: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        GeminiDecisionsConfig(boolean_threshold=threshold).create()
