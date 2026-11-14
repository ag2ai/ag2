# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Annotated, Any, Required

import pytest
from pydantic import AliasChoices, AliasPath, BaseModel, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel
from typing_extensions import TypedDict

pytest.importorskip("openai")
pytest.importorskip("typesafe_sdk")

import httpx2

from ag2 import Agent, AgentReply, ResponseSchema
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


class Urgency(IntEnum):
    LOW = 0
    """Can wait a week."""
    HIGH = 1
    """Needs an answer today."""


class Plain(Enum):
    A = "a"
    B = "b"


@dataclass
class TicketTriage:
    is_spam: Annotated[bool, Question("Is this message spam?")]
    department: Department
    urgency: Annotated[Urgency, Field(description="How urgent is this ticket?")]


class TicketTriageDict(TypedDict):
    is_spam: Annotated[bool, Question("Is this message spam?")]
    department: Department
    urgency: Annotated[Urgency, Field(description="How urgent is this ticket?")]


class TicketTriageModel(BaseModel):
    is_spam: Annotated[bool, Question("Is this message spam?")]
    department: Department
    urgency: Annotated[Urgency, Field(description="How urgent is this ticket?")]


class AliasedTriage(BaseModel):
    is_spam: Annotated[bool, Question("Is this message spam?")] = Field(alias="isSpam")


@dataclass
class AliasedTriageDataclass:
    is_spam: Annotated[bool, Question("Is this message spam?")] = Field(alias="isSpam")


class ChoiceTriage(BaseModel):
    is_spam: Annotated[bool, Question("Is this message spam?")] = Field(
        validation_alias=AliasChoices(AliasPath("nested", "spam"), "isSpam")
    )


@dataclass
class MetadataTriage:
    is_spam: Annotated[bool, Question("Is this message spam?")] = field(metadata={"alias": "isSpam"})


@dataclass
class DescribedMetadataTriage:
    is_spam: Annotated[bool, Question("Is this message spam?"), Field(description="Spam?")] = field(
        metadata={"alias": "isSpam"}
    )


@dataclass
class GeneratedAliasDataclass:
    __pydantic_config__ = ConfigDict(alias_generator=to_camel)

    is_spam: Annotated[bool, Question("Is this message spam?")]


class GeneratedAliasModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel)

    is_spam: Annotated[bool, Question("Is this message spam?")]


class AliasedTriageDict(TypedDict):
    is_spam: Annotated[bool, Question("Is this message spam?"), Field(alias="isSpam")]


class ValidatedTriage(BaseModel):
    is_spam: Annotated[bool, Question("Is this message spam?")] = Field(alias="isSpam")

    @model_validator(mode="after")
    def _check(self) -> "ValidatedTriage":
        return self


class NamedTriage(BaseModel):
    model_config = ConfigDict(validate_by_alias=False, validate_by_name=True)

    is_spam: Annotated[bool, Question("Is this message spam?")] = Field(alias="isSpam")


class PathTriage(BaseModel):
    is_spam: Annotated[bool, Question("Is this message spam?")] = Field(validation_alias=AliasPath("flags", 0))


class SharedAlias(BaseModel):
    is_spam: Annotated[bool, Question("Is this message spam?")] = Field(alias="flag")
    is_urgent: Annotated[bool, Question("Is this ticket urgent?")] = Field(alias="flag")


class AliasOfAnotherField(BaseModel):
    is_spam: Annotated[bool, Question("Is this message spam?")] = Field(alias="is_urgent")
    is_urgent: Annotated[bool, Question("Is this ticket urgent?")]


class QualifiedTriage(TypedDict, total=False):
    is_spam: Required[Annotated[bool, Question("Is this message spam?")]]


@dataclass
class Sender:
    is_customer: Annotated[bool, Question("Is the sender a customer?")]


@dataclass
class Nested:
    sender: Annotated[Sender, Question("Who sent it?")]


@dataclass
class Nullable:
    is_spam: Annotated[bool | None, Question("Is this message spam?")]


@dataclass
class Listed:
    flags: Annotated[list[bool], Question("Which flags apply?")]


@dataclass
class Defaulted:
    is_spam: Annotated[bool, Question("Is this message spam?")] = False


@dataclass
class Pick:
    pick: Annotated[Plain, Question("Pick one", options={"a": "First", "b": "Second"})]


@dataclass
class Churn:
    likelihood: Annotated[float, Field(ge=0, le=1), Question("How likely is this customer to churn?")]


class Spam(BaseModel):
    data: Annotated[bool, Question("Is this message spam?")]


class Gapped(IntEnum):
    LOW = 1
    HIGH = 3


@dataclass
class Skipping:
    urgency: Annotated[Gapped, Question("How urgent is this ticket?")]


class Open(BaseModel):
    model_config = ConfigDict(extra="allow")

    is_spam: Annotated[bool, Question("Is this message spam?")]


@dataclass
class Empty:
    pass


@dataclass
class Unasked:
    is_spam: bool


@dataclass
class Asked:
    """What the model was asked for one question, the same shape for both providers."""

    instructions: str | None
    options: dict[str, str | None]


class Provider(ABC):
    """A decision provider behind a fake HTTP API that answers several named questions."""

    unsupported: type[Exception]

    def __init__(self) -> None:
        self.requests = 0
        self.questions: dict[str, Asked] = {}
        self.answers: dict[str, dict[str, Any]] = {}
        self.client = httpx2.AsyncClient(transport=httpx2.MockTransport(self._handle))

    @abstractmethod
    def _handle(self, request: httpx2.Request) -> httpx2.Response: ...

    @abstractmethod
    def config(self, *, boolean_threshold: float) -> OpenAIDecisionsConfig | TypeSafeConfig: ...

    @abstractmethod
    def predicate(self, probability: float) -> dict[str, Any]: ...

    @abstractmethod
    def choice(self, value: str, options: tuple[str, ...]) -> dict[str, Any]: ...

    @abstractmethod
    def score(self, value: float, levels: int) -> dict[str, Any]: ...

    def triage(self, *, spam: float = 0.1) -> dict[str, dict[str, Any]]:
        return {
            "is_spam": self.predicate(spam),
            "department": self.choice("billing", ("billing", "technical")),
            "urgency": self.score(0.8, 2),
        }

    async def ask(
        self,
        schema: Any,
        answers: dict[str, dict[str, Any]],
        *,
        prompt: str | None = None,
        boolean_threshold: float = 0.5,
    ) -> AgentReply[Any]:
        self.answers = answers
        config = self.config(boolean_threshold=boolean_threshold)
        agent = Agent("a", prompt=prompt or [], config=config, response_schema=schema)
        return await agent.ask("I was charged twice for my order.")


class OpenAIProvider(Provider):
    unsupported = OpenAIUnsupported

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests += 1
        for question in json.loads(request.content)["questions"]:
            entries = question.get("choices") or question.get("levels") or []
            self.questions[question["name"]] = Asked(
                question["instructions"],
                {str(o.get("value", i)): o.get("description") for i, o in enumerate(entries)},
            )
        answers = [{"name": name, **answer} for name, answer in self.answers.items()]
        return httpx2.Response(200, json={"model": "m", "answers": answers, "usage": USAGE})

    def config(self, *, boolean_threshold: float) -> OpenAIDecisionsConfig:
        return OpenAIDecisionsConfig(api_key="k", http_client=self.client, boolean_threshold=boolean_threshold)

    def predicate(self, probability: float) -> dict[str, Any]:
        return {"type": "predicate", "probability": probability}

    def choice(self, value: str, options: tuple[str, ...]) -> dict[str, Any]:
        probabilities = [{"value": o, "probability": 1 / len(options)} for o in options]
        return {"type": "choice", "choice": value, "confidence": 0.9, "probabilities": probabilities}

    def score(self, value: float, levels: int) -> dict[str, Any]:
        probabilities = [{"value": i, "label": str(i), "probability": 1 / levels} for i in range(levels)]
        return {"type": "score", "score": value, "confidence": 0.5, "probabilities": probabilities}


class TypeSafeProvider(Provider):
    unsupported = TypeSafeUnsupported

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests += 1
        for name, question in json.loads(request.content)["questions"].items():
            criteria = question.get("criteria") or {}
            if isinstance(criteria, list):
                criteria = {str(i): c for i, c in enumerate(criteria)}
            self.questions[name] = Asked(question.get("instructions"), criteria)
        body = {"model": "m", "answers": self.answers, "usage": {"input_tokens": 1, "output_tokens": 0}}
        return httpx2.Response(200, json=body)

    def config(self, *, boolean_threshold: float) -> TypeSafeConfig:
        return TypeSafeConfig(api_key="k", http_client=self.client, boolean_threshold=boolean_threshold)

    def predicate(self, probability: float) -> dict[str, Any]:
        return {"type": "noul", "noul": probability}

    def choice(self, value: str, options: tuple[str, ...]) -> dict[str, Any]:
        probabilities = {o: 1 / len(options) for o in options}
        return {"type": "choice", "choice": value, "confidence": 0.9, "probabilities": probabilities}

    def score(self, value: float, levels: int) -> dict[str, Any]:
        legend = {str(i): "x" for i in range(levels)}
        probabilities = {str(i): 1 / levels for i in range(levels)}
        return {"type": "score", "score": value, "confidence": 0.5, "legend": legend, "probabilities": probabilities}


@pytest.fixture(params=[OpenAIProvider, TypeSafeProvider], ids=["openai", "typesafe"])
def provider(request: pytest.FixtureRequest) -> Provider:
    provider_type: type[Provider] = request.param
    return provider_type()


@pytest.mark.asyncio
class TestSeveralQuestions:
    async def test_each_field_is_one_question_of_one_request(self, provider: Provider) -> None:
        answers = provider.triage()

        reply = await provider.ask(TicketTriage, answers, prompt="You triage support tickets.")

        assert await reply.content() == TicketTriage(is_spam=False, department=Department.BILLING, urgency=Urgency.HIGH)
        assert provider.requests == 1
        assert provider.questions == {
            "is_spam": Asked("You triage support tickets.\n\nIs this message spam?", {}),
            "department": Asked(
                "You triage support tickets.\n\nWhich team should handle this ticket?",
                {"billing": "Payments, invoicing, refunds.", "technical": "Bugs, outages, integrations."},
            ),
            "urgency": Asked(
                "You triage support tickets.\n\nHow urgent is this ticket?",
                {"0": "Can wait a week.", "1": "Needs an answer today."},
            ),
        }

    async def test_metadata_is_each_answer_by_field(self, provider: Provider) -> None:
        answers = provider.triage()

        reply = await provider.ask(TicketTriage, answers, prompt="You triage support tickets.")
        scalar = await provider.ask(Annotated[bool, Question("Is this message spam?")], {"answer": answers["is_spam"]})

        assert set(reply.response.metadata) == {"is_spam", "department", "urgency"}
        assert reply.response.metadata["is_spam"] == scalar.response.metadata

    @pytest.mark.parametrize(
        ("schema", "expected"),
        [
            (
                TicketTriageDict,
                {"is_spam": False, "department": Department.BILLING, "urgency": Urgency.HIGH},
            ),
            (
                TicketTriageModel,
                TicketTriageModel(is_spam=False, department=Department.BILLING, urgency=Urgency.HIGH),
            ),
            (
                ResponseSchema(TicketTriage),
                TicketTriage(is_spam=False, department=Department.BILLING, urgency=Urgency.HIGH),
            ),
        ],
        ids=["typed-dict", "base-model", "response-schema"],
    )
    async def test_any_flat_class_is_accepted(self, provider: Provider, schema: Any, expected: Any) -> None:
        reply = await provider.ask(schema, provider.triage())

        assert await reply.content() == expected

    async def test_boolean_threshold_applies_to_every_predicate(self, provider: Provider) -> None:
        reply = await provider.ask(TicketTriage, provider.triage(spam=0.6), boolean_threshold=0.7)

        assert await reply.content() == TicketTriage(is_spam=False, department=Department.BILLING, urgency=Urgency.HIGH)

    async def test_probability_field_is_returned_raw(self, provider: Provider) -> None:
        reply = await provider.ask(Churn, {"likelihood": provider.predicate(0.37)})

        assert await reply.content() == Churn(likelihood=0.37)

    async def test_single_field_named_data_is_a_class(self, provider: Provider) -> None:
        reply = await provider.ask(Spam, {"data": provider.predicate(0.9)}, prompt="You triage support tickets.")

        assert await reply.content() == Spam(data=True)
        assert provider.questions == {"data": Asked("You triage support tickets.\n\nIs this message spam?", {})}
        assert set(reply.response.metadata) == {"data"}

    async def test_field_options_come_from_its_question(self, provider: Provider) -> None:
        reply = await provider.ask(Pick, {"pick": provider.choice("b", ("a", "b"))})

        assert await reply.content() == Pick(pick=Plain.B)
        assert provider.questions == {"pick": Asked("Pick one", {"a": "First", "b": "Second"})}

    @pytest.mark.parametrize(
        ("schema", "expected"),
        [
            (AliasedTriage, AliasedTriage(isSpam=True)),
            (AliasedTriageDataclass, AliasedTriageDataclass(is_spam=True)),
            (NamedTriage, NamedTriage(is_spam=True)),
            (ChoiceTriage, ChoiceTriage.model_validate({"isSpam": True})),
            (MetadataTriage, MetadataTriage(is_spam=True)),
            (DescribedMetadataTriage, DescribedMetadataTriage(is_spam=True)),
            (GeneratedAliasDataclass, GeneratedAliasDataclass(is_spam=True)),
            (GeneratedAliasModel, GeneratedAliasModel.model_validate({"isSpam": True})),
            (AliasedTriageDict, {"is_spam": True}),
            (ValidatedTriage, ValidatedTriage(isSpam=True)),
        ],
        ids=[
            "base-model",
            "dataclass",
            "validated-by-name",
            "alias-choices-after-a-path",
            "dataclass-field-metadata",
            "dataclass-field-metadata-with-description",
            "dataclass-alias-generator",
            "base-model-alias-generator",
            "typed-dict",
            "model-validator",
        ],
    )
    async def test_aliased_field_is_asked_by_its_name(self, provider: Provider, schema: Any, expected: Any) -> None:
        reply = await provider.ask(schema, {"is_spam": provider.predicate(0.9)}, prompt="Q")

        assert await reply.content() == expected
        assert provider.questions == {"is_spam": Asked("Q\n\nIs this message spam?", {})}
        assert set(reply.response.metadata) == {"is_spam"}

    async def test_required_typed_dict_field_keeps_its_question(self, provider: Provider) -> None:
        reply = await provider.ask(QualifiedTriage, {"is_spam": provider.predicate(0.9)})

        assert await reply.content() == {"is_spam": True}
        assert provider.questions == {"is_spam": Asked("Is this message spam?", {})}

    async def test_a_missing_answer_names_the_field(self, provider: Provider) -> None:
        answers = provider.triage()
        del answers["urgency"]

        with pytest.raises(ValueError, match="no answer for question 'urgency'"):
            await provider.ask(TicketTriage, answers)


@pytest.mark.asyncio
class TestRejections:
    @pytest.mark.parametrize("schema", [Nested, Nullable, Listed, Defaulted], ids=lambda s: s.__name__)
    async def test_field_that_is_not_a_decision(self, provider: Provider, schema: Any) -> None:
        with pytest.raises(provider.unsupported, match="Field '.+' of `response_schema`"):
            await provider.ask(schema, {}, prompt="Q")

        assert provider.requests == 0

    async def test_field_validated_from_a_nested_path(self, provider: Provider) -> None:
        with pytest.raises(provider.unsupported, match="Field 'is_spam' .+ nested path"):
            await provider.ask(PathTriage, {}, prompt="Q")

        assert provider.requests == 0

    @pytest.mark.parametrize("schema", [SharedAlias, AliasOfAnotherField], ids=lambda s: s.__name__)
    async def test_fields_validated_from_one_key(self, provider: Provider, schema: Any) -> None:
        with pytest.raises(provider.unsupported, match="Fields 'is_spam' and 'is_urgent' .+ both validated from key"):
            await provider.ask(schema, {}, prompt="Q")

        assert provider.requests == 0

    async def test_score_field_not_numbered_from_zero(self, provider: Provider) -> None:
        with pytest.raises(provider.unsupported, match="numbered from 0"):
            await provider.ask(Skipping, {}, prompt="Q")

        assert provider.requests == 0

    async def test_class_open_to_extra_fields(self, provider: Provider) -> None:
        with pytest.raises(provider.unsupported, match="additional properties"):
            await provider.ask(Open, {}, prompt="Q")

        assert provider.requests == 0

    async def test_class_without_fields(self, provider: Provider) -> None:
        with pytest.raises(provider.unsupported, match="no fields"):
            await provider.ask(Empty, {}, prompt="Q")

        assert provider.requests == 0

    async def test_field_without_a_question(self, provider: Provider) -> None:
        with pytest.raises(ValueError, match="Field 'is_spam' of `response_schema` has no question"):
            await provider.ask(Unasked, {}, prompt="You triage support tickets.")

        assert provider.requests == 0
