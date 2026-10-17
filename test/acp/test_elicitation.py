# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""The CLI agent asking the user something: ACP ``elicitation/create``.

Every test drives the public ``Agent.run`` surface with a config bound to the
in-process scripted agent, and asserts only what a caller can observe: the Reply,
the events on the Run's stream, the prompts the human was shown, and the response
the scripted agent got back.
"""

import asyncio
from typing import Any

import pytest
from acp import schema

from ag2 import Agent, Context
from ag2.acp.elicitation import MAX_ATTEMPTS
from ag2.acp.events import ACPElicitation
from ag2.acp.testing import FAKE_SESSION_ID, ACPTurn, ScriptedElicitation, fake_acp_config
from ag2.events import BaseEvent, HumanInputRequest, ModelResponse

AUTH_URL = "https://example.com/authorize"


def _text_update(text: str) -> schema.AgentMessageChunk:
    return schema.AgentMessageChunk(
        session_update="agent_message_chunk",
        content=schema.TextContentBlock(type="text", text=text),
    )


def _url_mode(url: str = AUTH_URL, elicitation_id: str = "elicit-1") -> schema.ElicitationUrlSessionMode:
    return schema.ElicitationUrlSessionMode(session_id=FAKE_SESSION_ID, elicitation_id=elicitation_id, url=url)


def _form_mode(properties: dict[str, Any], required: list[str] | None = None) -> schema.ElicitationFormSessionMode:
    return schema.ElicitationFormSessionMode(
        session_id=FAKE_SESSION_ID,
        requested_schema=schema.ElicitationSchema(type="object", properties=properties, required=required),
    )


def _asks(*elicitations: ScriptedElicitation, reply: str = "done") -> ACPTurn:
    """A turn where the agent asks, then replies once answered."""
    return ACPTurn(elicitations=elicitations, updates=[_text_update(reply)])


class Human:
    """A scripted human, answering the prompts it is shown in order.

    Running out of answers is a failure, not an empty answer: an unexpected extra
    prompt would otherwise be silently absorbed as "take the default".
    """

    def __init__(self, *answers: str) -> None:
        self._answers = list(answers)
        self.prompts: list[str] = []

    def answer(self, event: HumanInputRequest) -> str:
        self.prompts.append(event.content)
        if not self._answers:
            raise AssertionError(f"the agent asked one question too many: {event.content!r}")
        return self._answers.pop(0)

    @property
    def prompt(self) -> str:
        """The only prompt the human was shown (fails if there was not exactly one)."""
        [only] = self.prompts
        return only


async def _reply_body(cfg: Any, human: Human | None = None) -> str:
    """Run one turn against the scripted agent and return the reply's text."""
    agent = Agent("acp", config=cfg, hitl_hook=human.answer if human is not None else None)
    try:
        result = await agent.ask("hello")
    finally:
        await cfg.aclose()
    return result.body


@pytest.mark.asyncio
class TestCapabilityAdvertisement:
    async def test_ask_advertises_form_and_url(self) -> None:
        initialized: list[schema.ClientCapabilities | None] = []
        cfg = fake_acp_config(_asks(), initialize_calls=initialized)

        await _reply_body(cfg)

        [capabilities] = initialized
        assert capabilities is not None
        assert capabilities.elicitation == schema.ElicitationCapabilities(
            form=schema.ElicitationFormCapabilities(),
            url=schema.ElicitationUrlCapabilities(),
        )

    async def test_decline_advertises_nothing(self) -> None:
        # Not "advertise and refuse": a conforming agent must never ask at all.
        initialized: list[schema.ClientCapabilities | None] = []
        cfg = fake_acp_config(_asks(), elicitation_policy="decline", initialize_calls=initialized)

        await _reply_body(cfg)

        [capabilities] = initialized
        assert capabilities is not None
        assert capabilities.elicitation is None


@pytest.mark.asyncio
class TestDeclinePolicy:
    async def test_a_question_asked_anyway_is_declined(self) -> None:
        answers: list[schema.CreateElicitationResponse] = []
        cfg = fake_acp_config(
            _asks(ScriptedElicitation("Authorize GitHub", _url_mode())),
            elicitation_policy="decline",
            elicitation_responses=answers,
        )

        body = await _reply_body(cfg)

        assert [a.action for a in answers] == ["decline"]
        assert body == "done"

    async def test_the_human_is_never_prompted(self) -> None:
        human = Human()  # any prompt at all raises
        cfg = fake_acp_config(
            _asks(ScriptedElicitation("Authorize GitHub", _url_mode())),
            elicitation_policy="decline",
        )

        await _reply_body(cfg, human)

        assert human.prompts == []


@pytest.mark.asyncio
class TestUrlMode:
    async def test_shows_the_message_and_the_url(self) -> None:
        human = Human("yes")
        cfg = fake_acp_config(_asks(ScriptedElicitation("Authorize GitHub", _url_mode())))

        await _reply_body(cfg, human)

        assert "Authorize GitHub" in human.prompt
        assert AUTH_URL in human.prompt

    async def test_confirming_accepts(self) -> None:
        answers: list[schema.CreateElicitationResponse] = []
        cfg = fake_acp_config(
            _asks(ScriptedElicitation("Authorize GitHub", _url_mode()), reply="authorized"),
            elicitation_responses=answers,
        )

        body = await _reply_body(cfg, Human("yes"))

        # url mode requests no fields, so an accept carries no content.
        assert answers == [schema.AcceptElicitationResponse(action="accept", content=None)]
        # The answer reached the agent inside the same turn: only one turn is
        # scripted, and it went on to reply.
        assert body == "authorized"

    async def test_refusing_declines(self) -> None:
        answers: list[schema.CreateElicitationResponse] = []
        cfg = fake_acp_config(
            _asks(ScriptedElicitation("Authorize GitHub", _url_mode())),
            elicitation_responses=answers,
        )

        body = await _reply_body(cfg, Human("no"))

        assert [a.action for a in answers] == ["decline"]
        assert body == "done"

    async def test_the_completion_notification_is_handled(self) -> None:
        answers: list[schema.CreateElicitationResponse] = []
        cfg = fake_acp_config(
            _asks(ScriptedElicitation("Authorize GitHub", _url_mode(), complete=True)),
            elicitation_responses=answers,
        )

        body = await _reply_body(cfg, Human("yes"))

        assert [a.action for a in answers] == ["accept"]
        assert body == "done"


@pytest.mark.asyncio
class TestStreamEvent:
    async def test_the_question_appears_on_the_stream(self) -> None:
        seen: list[BaseEvent] = []
        cfg = fake_acp_config(_asks(ScriptedElicitation("Authorize GitHub", _url_mode())))
        agent = Agent("acp", config=cfg, hitl_hook=Human("yes").answer)

        try:
            async with agent.run("hello") as run:
                run.stream.subscribe(lambda event: seen.append(event))
                await run.result()
        finally:
            await cfg.aclose()

        assert [e for e in seen if isinstance(e, ACPElicitation)] == [
            ACPElicitation("Authorize GitHub", "url", url=AUTH_URL)
        ]

    async def test_a_form_question_names_the_requested_fields(self) -> None:
        seen: list[BaseEvent] = []
        mode = _form_mode({
            "name": schema.ElicitationStringPropertySchema(type="string"),
            "count": schema.ElicitationIntegerPropertySchema(type="integer"),
        })
        cfg = fake_acp_config(_asks(ScriptedElicitation("Name it", mode)))
        agent = Agent("acp", config=cfg, hitl_hook=Human("x", "1").answer)

        try:
            async with agent.run("hello") as run:
                run.stream.subscribe(lambda event: seen.append(event))
                await run.result()
        finally:
            await cfg.aclose()

        assert [e for e in seen if isinstance(e, ACPElicitation)] == [
            ACPElicitation("Name it", "form", fields=["name", "count"])
        ]

    async def test_the_event_is_published_in_the_conversation_context(self) -> None:
        # Per ADR 0004 the event goes out with `context.send`, so a subscriber is
        # dispatched under the run's own context — not some other stream's.
        contexts: list[Context] = []

        def record(event: BaseEvent, ctx: Context) -> None:
            if isinstance(event, ACPElicitation):
                contexts.append(ctx)

        cfg = fake_acp_config(_asks(ScriptedElicitation("Authorize GitHub", _url_mode())))
        agent = Agent("acp", config=cfg, hitl_hook=Human("yes").answer)

        try:
            async with agent.run("hello") as run:
                run.stream.subscribe(record)
                await run.result()
                stream_id = run.stream.id
        finally:
            await cfg.aclose()

        assert [ctx.stream.id for ctx in contexts] == [stream_id]

    async def test_the_event_precedes_the_prompt(self) -> None:
        # An observer must see the question even when something other than an
        # interactive human ends up answering it.
        timeline: list[Any] = []

        def note_prompt(event: HumanInputRequest) -> str:
            timeline.append("prompted")
            return "yes"

        cfg = fake_acp_config(_asks(ScriptedElicitation("Authorize GitHub", _url_mode())))
        agent = Agent("acp", config=cfg, hitl_hook=note_prompt)

        try:
            async with agent.run("hello") as run:
                run.stream.subscribe(
                    lambda event: timeline.append(event) if isinstance(event, ACPElicitation) else None
                )
                await run.result()
        finally:
            await cfg.aclose()

        assert [type(entry).__name__ for entry in timeline] == ["ACPElicitation", "str"]


@pytest.mark.asyncio
class TestFormRendering:
    async def test_one_prompt_per_property_in_schema_order(self) -> None:
        human = Human("refactor", "3")
        mode = _form_mode({
            "strategy": schema.ElicitationStringPropertySchema(type="string"),
            "batch": schema.ElicitationIntegerPropertySchema(type="integer"),
        })
        cfg = fake_acp_config(_asks(ScriptedElicitation("Plan the change", mode)))

        await _reply_body(cfg, human)

        assert len(human.prompts) == 2
        assert "strategy" in human.prompts[0]
        assert "batch" in human.prompts[1]

    async def test_the_title_is_shown_and_falls_back_to_the_name(self) -> None:
        human = Human("a", "b")
        mode = _form_mode({
            "titled": schema.ElicitationStringPropertySchema(type="string", title="Refactoring strategy"),
            "untitled": schema.ElicitationStringPropertySchema(type="string"),
        })
        cfg = fake_acp_config(_asks(ScriptedElicitation("Plan the change", mode)))

        await _reply_body(cfg, human)

        assert "Refactoring strategy" in human.prompts[0]
        assert "untitled" in human.prompts[1]

    async def test_the_description_is_shown(self) -> None:
        human = Human("a")
        mode = _form_mode({
            "strategy": schema.ElicitationStringPropertySchema(
                type="string", description="How aggressively to rewrite call sites"
            )
        })
        cfg = fake_acp_config(_asks(ScriptedElicitation("Plan the change", mode)))

        await _reply_body(cfg, human)

        assert "How aggressively to rewrite call sites" in human.prompt

    async def test_allowed_values_are_shown(self) -> None:
        human = Human("inline")
        mode = _form_mode({
            "strategy": schema.ElicitationStringPropertySchema(type="string", enum=["inline", "extract"])
        })
        cfg = fake_acp_config(_asks(ScriptedElicitation("Plan the change", mode)))

        await _reply_body(cfg, human)

        assert "inline" in human.prompt
        assert "extract" in human.prompt

    async def test_a_value_outside_the_allowed_set_re_prompts(self) -> None:
        # The agent branches on those values; a spelling it never offered is as
        # useless to it as a word where a number was asked for.
        answers: list[schema.CreateElicitationResponse] = []
        human = Human("sideways", "extract")
        mode = _form_mode({
            "strategy": schema.ElicitationStringPropertySchema(type="string", enum=["inline", "extract"])
        })
        cfg = fake_acp_config(_asks(ScriptedElicitation("Plan the change", mode)), elicitation_responses=answers)

        await _reply_body(cfg, human)

        assert len(human.prompts) == 2
        assert answers == [schema.AcceptElicitationResponse(action="accept", content={"strategy": "extract"})]

    async def test_a_default_is_shown_and_taken_on_an_empty_answer(self) -> None:
        answers: list[schema.CreateElicitationResponse] = []
        human = Human("")
        mode = _form_mode({"batch": schema.ElicitationIntegerPropertySchema(type="integer", default=5)})
        cfg = fake_acp_config(_asks(ScriptedElicitation("Plan the change", mode)), elicitation_responses=answers)

        await _reply_body(cfg, human)

        assert "5" in human.prompt
        assert answers == [schema.AcceptElicitationResponse(action="accept", content={"batch": 5})]

    async def test_a_required_property_without_a_default_is_re_prompted(self) -> None:
        answers: list[schema.CreateElicitationResponse] = []
        human = Human("", "", "extract")
        mode = _form_mode(
            {"strategy": schema.ElicitationStringPropertySchema(type="string")},
            required=["strategy"],
        )
        cfg = fake_acp_config(_asks(ScriptedElicitation("Plan the change", mode)), elicitation_responses=answers)

        await _reply_body(cfg, human)

        assert len(human.prompts) == 3
        assert answers == [schema.AcceptElicitationResponse(action="accept", content={"strategy": "extract"})]

    async def test_an_optional_property_left_empty_is_omitted(self) -> None:
        # Not filled in with a guess: the agent asked for it or it is absent.
        answers: list[schema.CreateElicitationResponse] = []
        mode = _form_mode({
            "note": schema.ElicitationStringPropertySchema(type="string"),
            "batch": schema.ElicitationIntegerPropertySchema(type="integer"),
        })
        cfg = fake_acp_config(_asks(ScriptedElicitation("Plan the change", mode)), elicitation_responses=answers)

        await _reply_body(cfg, Human("", "4"))

        assert answers == [schema.AcceptElicitationResponse(action="accept", content={"batch": 4})]

    async def test_every_property_type_is_answered_in_its_own_type(self) -> None:
        answers: list[schema.CreateElicitationResponse] = []
        mode = _form_mode({
            "strategy": schema.ElicitationStringPropertySchema(type="string"),
            "ratio": schema.ElicitationNumberPropertySchema(type="number"),
            "batch": schema.ElicitationIntegerPropertySchema(type="integer"),
            "dry_run": schema.ElicitationBooleanPropertySchema(type="boolean"),
            "skip": schema.ElicitationMultiSelectPropertySchema(
                type="array",
                items=schema.StringMultiSelectItems(type="string", enum=["tests", "docs", "examples"]),
            ),
        })
        cfg = fake_acp_config(_asks(ScriptedElicitation("Plan the change", mode)), elicitation_responses=answers)

        await _reply_body(cfg, Human("extract", "0.5", "12", "yes", "tests, docs"))

        assert answers == [
            schema.AcceptElicitationResponse(
                action="accept",
                content={
                    "strategy": "extract",
                    "ratio": 0.5,
                    "batch": 12,
                    "dry_run": True,
                    "skip": ["tests", "docs"],
                },
            )
        ]

    @pytest.mark.parametrize(
        ("prop", "bad", "good", "expected"),
        [
            (schema.ElicitationIntegerPropertySchema(type="integer"), "twelve", "12", 12),
            (schema.ElicitationNumberPropertySchema(type="number"), "half", "0.5", 0.5),
            (schema.ElicitationBooleanPropertySchema(type="boolean"), "maybe", "no", False),
            (
                schema.ElicitationMultiSelectPropertySchema(
                    type="array", items=schema.StringMultiSelectItems(type="string", enum=["tests"])
                ),
                "docs",
                "tests",
                ["tests"],
            ),
        ],
        ids=["integer", "number", "boolean", "multi-select"],
    )
    async def test_an_uncoercible_answer_re_prompts(self, prop: Any, bad: str, good: str, expected: Any) -> None:
        # Never sent through as a string: the agent asked for a number and must
        # get a number.
        answers: list[schema.CreateElicitationResponse] = []
        human = Human(bad, good)
        cfg = fake_acp_config(
            _asks(ScriptedElicitation("Plan the change", _form_mode({"value": prop}))),
            elicitation_responses=answers,
        )

        await _reply_body(cfg, human)

        assert len(human.prompts) == 2
        assert answers == [schema.AcceptElicitationResponse(action="accept", content={"value": expected})]

    async def test_refusing_part_way_through_declines_the_whole_form(self) -> None:
        # A half-filled form is not what the agent asked for, and not something
        # AG2 should invent the rest of.
        answers: list[schema.CreateElicitationResponse] = []
        mode = _form_mode({
            "strategy": schema.ElicitationStringPropertySchema(type="string"),
            "batch": schema.ElicitationIntegerPropertySchema(type="integer"),
        })
        cfg = fake_acp_config(_asks(ScriptedElicitation("Plan the change", mode)), elicitation_responses=answers)

        body = await _reply_body(cfg, Human("extract", "!decline"))

        assert answers == [schema.DeclineElicitationResponse(action="decline")]
        assert body == "done"

    async def test_an_endlessly_unanswerable_property_declines_rather_than_looping(self) -> None:
        # A human types `!decline` to give up; a programmatic hook cannot, so the
        # re-prompt loop is bounded or the turn never ends.
        answers: list[schema.CreateElicitationResponse] = []
        human = Human(*["twelve"] * MAX_ATTEMPTS)
        cfg = fake_acp_config(
            _asks(
                ScriptedElicitation(
                    "Plan the change",
                    _form_mode({"batch": schema.ElicitationIntegerPropertySchema(type="integer")}),
                )
            ),
            elicitation_responses=answers,
        )

        body = await _reply_body(cfg, human)

        assert len(human.prompts) == MAX_ATTEMPTS
        assert answers == [schema.DeclineElicitationResponse(action="decline")]
        assert body == "done"

    @pytest.mark.parametrize(
        "unrenderable",
        [
            schema.ElicitationOtherPropertySchema(type="swatch"),
            # A multi-select whose option list is in a shape this version cannot
            # read: accepting it blind would send values the agent never offered.
            schema.ElicitationMultiSelectPropertySchema(
                type="array", items=schema.OtherMultiSelectItems(type="swatch")
            ),
        ],
        ids=["unknown-type", "unknown-multi-select-items"],
    )
    async def test_a_field_this_version_cannot_render_declines_before_prompting(self, unrenderable: Any) -> None:
        # The unrenderable field is *second*: a form AG2 cannot fill in completely
        # is refused on arrival, so the human never answers a field whose answer
        # would then be thrown away. `Human()` has no answers, so any prompt fails.
        answers: list[schema.CreateElicitationResponse] = []
        human = Human()
        mode = _form_mode({
            "strategy": schema.ElicitationStringPropertySchema(type="string"),
            "colour": unrenderable,
        })
        cfg = fake_acp_config(_asks(ScriptedElicitation("Pick a colour", mode)), elicitation_responses=answers)

        body = await _reply_body(cfg, human)

        assert human.prompts == []
        assert answers == [schema.DeclineElicitationResponse(action="decline")]
        assert body == "done"

    async def test_declared_bounds_are_shown_and_enforced(self) -> None:
        # Sending a value the agent already said it cannot use is no better than
        # sending the wrong type, and a limit that is enforced must be visible.
        answers: list[schema.CreateElicitationResponse] = []
        human = Human("99", "7")
        mode = _form_mode({"batch": schema.ElicitationIntegerPropertySchema(type="integer", minimum=1, maximum=10)})
        cfg = fake_acp_config(_asks(ScriptedElicitation("Plan the change", mode)), elicitation_responses=answers)

        await _reply_body(cfg, human)

        assert "1" in human.prompts[0] and "10" in human.prompts[0]
        assert len(human.prompts) == 2
        assert answers == [schema.AcceptElicitationResponse(action="accept", content={"batch": 7})]

    async def test_a_number_with_no_json_form_re_prompts(self) -> None:
        # "nan" parses as a float but crosses the wire as `null`, which is not a
        # number the agent can use.
        answers: list[schema.CreateElicitationResponse] = []
        human = Human("nan", "0.5")
        mode = _form_mode({"ratio": schema.ElicitationNumberPropertySchema(type="number")})
        cfg = fake_acp_config(_asks(ScriptedElicitation("Plan the change", mode)), elicitation_responses=answers)

        await _reply_body(cfg, human)

        assert len(human.prompts) == 2
        assert answers == [schema.AcceptElicitationResponse(action="accept", content={"ratio": 0.5})]


@pytest.mark.asyncio
class TestDegradation:
    async def test_an_unrecognised_mode_is_declined(self) -> None:
        # ACP 0.12's `Other`-shaped variants exist so a client can degrade on a
        # mode it does not know. Declining tells the agent to fall back; a
        # transport-level failure would just break its turn.
        answers: list[schema.CreateElicitationResponse] = []
        cfg = fake_acp_config(
            _asks(ScriptedElicitation("Speak your answer", schema.ElicitationOtherPropertySchema(type="voice"))),
            elicitation_responses=answers,
        )

        body = await _reply_body(cfg, Human())

        assert answers == [schema.DeclineElicitationResponse(action="decline")]
        assert body == "done"

    async def test_no_hitl_channel_cancels_rather_than_blocking(self) -> None:
        answers: list[schema.CreateElicitationResponse] = []
        cfg = fake_acp_config(
            _asks(ScriptedElicitation("Authorize GitHub", _url_mode())),
            elicitation_responses=answers,
        )

        # No hitl_hook: there is no human to reach.
        body = await _reply_body(cfg)

        assert answers == [schema.CancelElicitationResponse(action="cancel")]
        assert body == "done"

    async def test_a_request_scoped_question_reaches_the_human_before_a_session_exists(self) -> None:
        # This is how a pre-session authentication flow works, so it must not
        # assume a live session.
        answers: list[schema.CreateElicitationResponse] = []
        human = Human("yes")
        cfg = fake_acp_config(
            _asks(),
            initialize_elicitations=[
                ScriptedElicitation(
                    "Log in first",
                    schema.ElicitationUrlRequestMode(request_id=1, elicitation_id="elicit-1", url=AUTH_URL),
                )
            ],
            elicitation_responses=answers,
        )

        body = await _reply_body(cfg, human)

        assert AUTH_URL in human.prompt
        assert answers == [schema.AcceptElicitationResponse(action="accept", content=None)]
        assert body == "done"

    async def test_cancelling_the_turn_leaves_no_prompt_outstanding(self) -> None:
        human = _NeverAnswers()
        seen: list[BaseEvent] = []
        cfg = fake_acp_config(
            _asks(ScriptedElicitation("Authorize GitHub", _url_mode())),
            turn_timeout=0.3,
            cancel_timeout=0.1,
        )
        agent = Agent("acp", config=cfg, hitl_hook=human.answer)

        try:
            async with agent.run("hello") as run:
                run.stream.subscribe(lambda event: seen.append(event))
                await run.result()
        finally:
            await cfg.aclose()

        # The turn ended rather than hanging on the unanswered question...
        [response] = [e for e in seen if isinstance(e, ModelResponse)]
        assert response.finish_reason == "timeout"
        # ...and the question went with it.
        assert human.prompted, "the human was never asked, so nothing was cancelled"
        assert human.cancelled, "the abandoned run left a prompt waiting for input"


class _NeverAnswers:
    """A human who is shown the question and never answers it."""

    def __init__(self) -> None:
        self.prompted = False
        self.cancelled = False

    async def answer(self, event: HumanInputRequest) -> str:
        self.prompted = True
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return ""  # pragma: no cover - the wait above never completes
