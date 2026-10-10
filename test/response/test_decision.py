# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Annotated, Any

import pytest

pytest.importorskip("openai")
pytest.importorskip("typesafe_sdk")

import httpx2

from ag2 import Agent, ResponseSchema
from ag2.config import OpenAIDecisionsConfig, TypeSafeConfig
from ag2.config.openai import UnsupportedResponseSchemaError as OpenAIUnsupported
from ag2.config.typesafe import UnsupportedResponseSchemaError as TypeSafeUnsupported
from ag2.response import Question

USAGE = {
    "input_tokens": 1,
    "output_tokens": 0,
    "total_tokens": 1,
    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
    "output_tokens_details": {"reasoning_tokens": 0},
}


class Department(Enum):
    """Which team should handle this ticket?"""

    BILLING = "billing"
    """Payments, invoicing, refunds."""
    TECHNICAL = "technical"
    """Bugs, outages, integrations."""


class Plain(Enum):
    A = "a"
    B = "b"


class Severity(IntEnum):
    """How severe is this?"""

    LOW = 0
    """Cosmetic."""
    HIGH = 1
    """Outage."""


class Bare(IntEnum):
    LOW = 0
    HIGH = 1


UnreadableChoice = Enum("UnreadableChoice", {"A": "a", "B": "b"})
UnreadableScore = IntEnum("UnreadableScore", {"LOW": 0, "HIGH": 1})


@dataclass
class Asked:
    """What the model was asked, the same shape for both providers."""

    instructions: str | None
    options: dict[str, str | None]


class Provider(ABC):
    """A decision provider behind a fake HTTP API; subclasses supply the wire format."""

    unsupported: type[Exception]
    describe: str  # config field that overrides option descriptions

    def __init__(self) -> None:
        self.questions: list[Any] = []
        self.answer: dict[str, Any] = {}
        self.client = httpx2.AsyncClient(transport=httpx2.MockTransport(self._handle))

    @abstractmethod
    def _handle(self, request: httpx2.Request) -> httpx2.Response: ...

    @abstractmethod
    def _config(self, **config: Any) -> Any: ...

    @abstractmethod
    def predicate(self, probability: float) -> dict[str, Any]: ...

    @abstractmethod
    def choice(self, value: str, options: tuple[str, ...]) -> dict[str, Any]: ...

    @abstractmethod
    def score(self, value: float, levels: int) -> dict[str, Any]: ...

    @abstractmethod
    def asked(self) -> Asked: ...

    async def ask(self, schema: Any, answer: dict[str, Any], *, prompt: str | None = None, **config: Any) -> Any:
        self.answer = answer
        agent = Agent("a", prompt=prompt or [], config=self._config(**config), response_schema=schema)
        return await (await agent.ask("hi")).content()


class OpenAIProvider(Provider):
    unsupported = OpenAIUnsupported
    describe = "descriptions"

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        self.questions.extend(json.loads(request.content)["questions"])
        body = {"model": "m", "answers": [{"name": "answer", **self.answer}], "usage": USAGE}
        return httpx2.Response(200, json=body)

    def _config(self, **config: Any) -> Any:
        return OpenAIDecisionsConfig(api_key="k", http_client=self.client, **config)

    def predicate(self, probability: float) -> dict[str, Any]:
        return {"type": "predicate", "probability": probability}

    def choice(self, value: str, options: tuple[str, ...]) -> dict[str, Any]:
        probabilities = [{"value": o, "probability": 1 / len(options)} for o in options]
        return {"type": "choice", "choice": value, "confidence": 0.9, "probabilities": probabilities}

    def score(self, value: float, levels: int) -> dict[str, Any]:
        probabilities = [{"value": i, "label": str(i), "probability": 1 / levels} for i in range(levels)]
        return {"type": "score", "score": value, "confidence": 0.5, "probabilities": probabilities}

    def asked(self) -> Asked:
        [question] = self.questions
        entries = question.get("choices") or question.get("levels") or []
        return Asked(
            question["instructions"],
            {str(o.get("value", i)): o.get("description") for i, o in enumerate(entries)},
        )


class TypeSafeProvider(Provider):
    unsupported = TypeSafeUnsupported
    describe = "criteria"

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        self.questions.append(json.loads(request.content)["questions"]["answer"])
        body = {"model": "m", "answers": {"answer": self.answer}, "usage": {"input_tokens": 1, "output_tokens": 0}}
        return httpx2.Response(200, json=body)

    def _config(self, **config: Any) -> Any:
        return TypeSafeConfig(api_key="k", http_client=self.client, **config)

    def predicate(self, probability: float) -> dict[str, Any]:
        return {"type": "noul", "noul": probability}

    def choice(self, value: str, options: tuple[str, ...]) -> dict[str, Any]:
        probabilities = {o: 1 / len(options) for o in options}
        return {"type": "choice", "choice": value, "confidence": 0.9, "probabilities": probabilities}

    def score(self, value: float, levels: int) -> dict[str, Any]:
        legend = {str(i): "x" for i in range(levels)}
        probabilities = {str(i): 1 / levels for i in range(levels)}
        return {"type": "score", "score": value, "confidence": 0.5, "legend": legend, "probabilities": probabilities}

    def asked(self) -> Asked:
        [question] = self.questions
        criteria = question.get("criteria") or {}
        if isinstance(criteria, list):
            criteria = {str(i): c for i, c in enumerate(criteria)}
        return Asked(question.get("instructions"), criteria)


@pytest.fixture(params=[OpenAIProvider, TypeSafeProvider], ids=["openai", "typesafe"])
def provider(request: pytest.FixtureRequest) -> Provider:
    provider_type: type[Provider] = request.param
    return provider_type()


@pytest.mark.asyncio
class TestQuestionMarker:
    async def test_question_on_plain_bool(self, provider: Provider) -> None:
        schema = Annotated[bool, Question("Is this a refund request?")]

        assert await provider.ask(schema, provider.predicate(0.9)) is True
        assert provider.asked() == Asked("Is this a refund request?", {})

    async def test_options_on_third_party_enum(self, provider: Provider) -> None:
        schema = Annotated[Plain, Question("Pick one", options={"a": "First", "b": "Second"})]

        assert await provider.ask(schema, provider.choice("b", ("a", "b"))) is Plain.B
        assert provider.asked() == Asked("Pick one", {"a": "First", "b": "Second"})

    async def test_score_levels_keyed_by_int(self, provider: Provider) -> None:
        schema = Annotated[Bare, Question("Rate it", options={0: "Fine", 1: "Broken"})]

        assert await provider.ask(schema, provider.score(0.8, 2)) is Bare.HIGH
        assert provider.asked() == Asked("Rate it", {"0": "Fine", "1": "Broken"})

    async def test_question_without_options_keeps_docstring_options(self, provider: Provider) -> None:
        schema = Annotated[Department, Question("Route it")]

        await provider.ask(schema, provider.choice("billing", ("billing", "technical")))

        assert provider.asked() == Asked(
            "Route it", {"billing": "Payments, invoicing, refunds.", "technical": "Bugs, outages, integrations."}
        )


@pytest.mark.asyncio
class TestPrecedence:
    async def test_docstring_is_the_question(self, provider: Provider) -> None:
        await provider.ask(Department, provider.choice("billing", ("billing", "technical")))

        assert provider.asked().instructions == "Which team should handle this ticket?"

    async def test_marker_beats_docstring(self, provider: Provider) -> None:
        await provider.ask(
            Annotated[Department, Question("From marker")], provider.choice("billing", ("billing", "technical"))
        )

        assert provider.asked().instructions == "From marker"

    async def test_description_beats_marker(self, provider: Provider) -> None:
        schema = ResponseSchema(Annotated[Department, Question("From marker")], description="From description")

        await provider.ask(schema, provider.choice("billing", ("billing", "technical")))

        assert provider.asked().instructions == "From description"

    async def test_marker_options_beat_docstrings(self, provider: Provider) -> None:
        schema = Annotated[Department, Question(options={"billing": "Money"})]

        await provider.ask(schema, provider.choice("billing", ("billing", "technical")))

        assert provider.asked().options == {"billing": "Money", "technical": "Bugs, outages, integrations."}

    async def test_config_beats_marker_options(self, provider: Provider) -> None:
        schema = Annotated[Department, Question(options={"billing": "Money", "technical": "Code"})]

        await provider.ask(
            schema, provider.choice("billing", ("billing", "technical")), **{provider.describe: {"billing": "Cash"}}
        )

        assert provider.asked().options == {"billing": "Cash", "technical": "Code"}

    async def test_prompt_and_question_are_joined(self, provider: Provider) -> None:
        await provider.ask(Annotated[bool, Question("Is it?")], provider.predicate(0.1), prompt="Be strict.")

        assert provider.asked().instructions == "Be strict.\n\nIs it?"


@pytest.mark.asyncio
class TestAnswers:
    @pytest.mark.parametrize(("probability", "expected"), [(0.7, True), (0.2, False)])
    async def test_bool_is_thresholded(self, provider: Provider, probability: float, expected: bool) -> None:
        assert await provider.ask(Annotated[bool, Question("?")], provider.predicate(probability)) is expected

    async def test_probability_is_returned_raw(self, provider: Provider) -> None:
        schema = ResponseSchema.from_schema({"type": "number", "minimum": 0, "maximum": 1}, name="p", description="?")

        assert await provider.ask(schema, provider.predicate(0.37)) == "0.37"

    @pytest.mark.parametrize(("raw", "expected"), [(0.4, Severity.LOW), (0.6, Severity.HIGH), (9.0, Severity.HIGH)])
    async def test_score_snaps_and_clamps(self, provider: Provider, raw: float, expected: Severity) -> None:
        assert await provider.ask(Severity, provider.score(raw, 2)) is expected

    async def test_envelope_does_not_change_the_question(self, provider: Provider) -> None:
        wrapped = ResponseSchema(bool, description="Is it?")
        bare = ResponseSchema(bool, description="Is it?", embed=False)

        assert await provider.ask(wrapped, provider.predicate(0.9)) is True
        first = provider.asked()
        provider.questions.clear()
        assert await provider.ask(bare, provider.predicate(0.9)) is True

        assert provider.asked() == first


@pytest.mark.asyncio
class TestRejections:
    @pytest.mark.parametrize("schema", [str, int, ResponseSchema(str, description="?")])
    async def test_non_decision_schema(self, provider: Provider, schema: Any) -> None:
        with pytest.raises(provider.unsupported, match="decision-only"):
            await provider.ask(schema, provider.predicate(0.5), prompt="Q")

        assert provider.questions == []

    async def test_bare_bool_without_prompt(self, provider: Provider) -> None:
        with pytest.raises(ValueError, match="question|asking"):
            await provider.ask(bool, provider.predicate(0.5))

        assert provider.questions == []


@pytest.mark.asyncio
class TestUnreadableSource:
    async def test_choice_proceeds_without_descriptions(self, provider: Provider) -> None:
        assert await provider.ask(UnreadableChoice, provider.choice("a", ("a", "b")), prompt="Q") is (
            UnreadableChoice.A
        )

        assert provider.asked().options == {"a": None, "b": None}

    async def test_marker_describes_unreadable_choice(self, provider: Provider) -> None:
        schema = Annotated[UnreadableChoice, Question(options={"a": "First"})]

        await provider.ask(schema, provider.choice("a", ("a", "b")), prompt="Q")

        assert provider.asked().options == {"a": "First", "b": None}

    async def test_score_without_descriptions(self) -> None:
        provider = OpenAIProvider()

        assert await provider.ask(UnreadableScore, provider.score(0.9, 2), prompt="Q") is UnreadableScore.HIGH
        assert provider.asked().options == {"0": None, "1": None}

    async def test_typesafe_score_requires_descriptions_and_names_the_way_out(self) -> None:
        provider = TypeSafeProvider()

        with pytest.raises(TypeSafeUnsupported, match=r"Question\(options=") as e:
            await provider.ask(UnreadableScore, provider.score(0.9, 2), prompt="Q")

        assert "criteria" in str(e.value)
        assert provider.questions == []

    async def test_marker_makes_typesafe_score_succeed(self) -> None:
        provider = TypeSafeProvider()
        schema = Annotated[UnreadableScore, Question(options={0: "Fine", 1: "Broken"})]

        assert await provider.ask(schema, provider.score(0.9, 2), prompt="Q") is UnreadableScore.HIGH
        assert provider.asked().options == {"0": "Fine", "1": "Broken"}


@pytest.mark.asyncio
class TestEdges:
    async def test_description_equal_to_docstring_still_beats_marker(self, provider: Provider) -> None:
        schema = ResponseSchema(Annotated[Department, Question("From marker")], description=Department.__doc__)

        await provider.ask(schema, provider.choice("billing", ("billing", "technical")))

        assert provider.asked().instructions == "Which team should handle this ticket?"

    async def test_stdlib_enum_placeholder_is_not_a_question(self, provider: Provider) -> None:
        class Placeholder(Enum):
            """An enumeration."""

            A = "a"
            B = "b"

        await provider.ask(Placeholder, provider.choice("a", ("a", "b")), prompt="Q")

        assert provider.asked().instructions == "Q"

    @pytest.mark.parametrize(
        "schema",
        [
            pytest.param(Annotated[Department, Question(options={"nope": "x"})], id="unknown-choice"),
            pytest.param(Annotated[Severity, Question(options={"0": "zero"})], id="score-as-string"),
            pytest.param(Annotated[bool, Question("?", options={"true": "x"})], id="predicate"),
        ],
    )
    async def test_option_keys_must_name_options(self, provider: Provider, schema: Any) -> None:
        with pytest.raises(ValueError, match=r"Question\(options="):
            await provider.ask(schema, provider.predicate(0.5), prompt="Q")

        assert provider.questions == []

    async def test_annotated_str_is_not_a_decision(self, provider: Provider) -> None:
        with pytest.raises(provider.unsupported, match="not a decision type"):
            await provider.ask(Annotated[str, Question("?")], provider.predicate(0.5), prompt="Q")
