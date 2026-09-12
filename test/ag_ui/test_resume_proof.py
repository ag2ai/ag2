# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Whoever answers an interrupt has to be whoever the interrupt was put to.

An answer attributed to a human is the input an agent trusts most, and without
this the interrupt id is a bare string in a request body. The protocol reserves
an envelope beside the answer for exactly this — signatures and routing keys,
never the answer itself — and requires clients to carry it back.
"""

from typing import Any

import pytest
from dirty_equals import IsPartialDict, IsStr

from ag2.ag_ui import AGUIStream
from ag2.ag_ui.interrupts import AG2_METADATA_KEY, NOT_PROVEN, PROOF_KEY

pytest.importorskip("starlette")

from ag_ui.core import AGUI_METADATA_KEY  # noqa: E402

from test.ag_ui.driving import (  # noqa: E402
    QUESTION,
    answer,
    app_for,
    ask_once,
    asking_agent,
    only,
    outcome_of,
    post_run,
    resolved,
    run_body,
    types_of,
)

pytestmark = pytest.mark.asyncio

SECOND_QUESTION = "And your favourite number?"


def proof_of(interrupt: dict[str, Any]) -> str:
    proof = interrupt["metadata"][AG2_METADATA_KEY][PROOF_KEY]
    assert isinstance(proof, str)
    return proof


async def refused(app: Any, resume: list[dict[str, Any]], *, run_id: str = "r2") -> dict[str, Any]:
    events = await post_run(app, run_body(thread_id="t1", run_id=run_id, text=None, resume=resume))
    assert types_of(events)[-1] == "RUN_ERROR", f"the client was left without a terminating event: {types_of(events)}"
    return only(events, "RUN_ERROR")


class TestTheInterruptCarriesItsProof:
    async def test_it_travels_in_the_envelope_beside_the_question(self) -> None:
        agent, _ = asking_agent()

        interrupt = await ask_once(app_for(AGUIStream(agent)))

        assert interrupt["metadata"] == IsPartialDict({AG2_METADATA_KEY: {PROOF_KEY: IsStr(min_length=16)}})
        # Never beside the answer itself: the question the human reads and the
        # proof the server checks are different things.
        assert QUESTION not in str(interrupt["metadata"])

    async def test_it_stays_out_of_the_key_the_protocol_reserves(self) -> None:
        agent, _ = asking_agent()

        interrupt = await ask_once(app_for(AGUIStream(agent)))

        assert AGUI_METADATA_KEY not in interrupt["metadata"]

    async def test_each_question_is_proven_separately(self) -> None:
        agent, _ = asking_agent(questions=(QUESTION, SECOND_QUESTION))
        app = app_for(AGUIStream(agent))

        first = await ask_once(app)
        second = await post_run(app, run_body(thread_id="t1", run_id="r2", text=None, resume=answer(first, "blue")))
        [second_interrupt] = outcome_of(second)["interrupts"]

        assert proof_of(second_interrupt) != proof_of(first)


async def test_the_proof_it_was_issued_with_is_accepted() -> None:
    agent, asked = asking_agent()
    app = app_for(AGUIStream(agent))

    interrupt = await ask_once(app)
    events = await post_run(app, run_body(thread_id="t1", run_id="r2", text=None, resume=answer(interrupt, "blue")))

    assert asked.answers == ["blue"]
    assert outcome_of(events) == {"type": "success"}


class TestWhatIsRefused:
    async def test_no_proof_at_all(self) -> None:
        agent, asked = asking_agent()
        app = app_for(AGUIStream(agent))

        interrupt = await ask_once(app)
        error = await refused(app, resolved(interrupt["id"], "blue"))

        assert error == IsPartialDict({"code": NOT_PROVEN})
        assert asked.answers == []

    async def test_an_envelope_of_the_wrong_shape(self) -> None:
        agent, asked = asking_agent()
        app = app_for(AGUIStream(agent))

        interrupt = await ask_once(app)
        error = await refused(app, resolved(interrupt["id"], "blue", metadata={AG2_METADATA_KEY: "a-bare-string"}))

        assert error == IsPartialDict({"code": NOT_PROVEN})
        assert asked.answers == []

    async def test_a_guessed_proof(self) -> None:
        agent, asked = asking_agent()
        app = app_for(AGUIStream(agent))

        interrupt = await ask_once(app)
        forged = {AG2_METADATA_KEY: {PROOF_KEY: "not-the-one-that-was-issued"}}
        error = await refused(app, resolved(interrupt["id"], "blue", metadata=forged))

        assert error == IsPartialDict({"code": NOT_PROVEN})
        assert asked.answers == []

    async def test_proof_issued_for_another_question(self) -> None:
        """The proof travels with one interrupt, not with the thread."""
        agent, asked = asking_agent(questions=(QUESTION, SECOND_QUESTION))
        app = app_for(AGUIStream(agent))

        first = await ask_once(app)
        second = await post_run(app, run_body(thread_id="t1", run_id="r2", text=None, resume=answer(first, "blue")))
        [second_interrupt] = outcome_of(second)["interrupts"]

        stale = {AG2_METADATA_KEY: {PROOF_KEY: proof_of(first)}}
        error = await refused(app, resolved(second_interrupt["id"], "7", metadata=stale), run_id="r3")

        assert error == IsPartialDict({"code": NOT_PROVEN})
        assert asked.answers == ["blue"]

    async def test_an_unproven_answer_can_be_followed_by_a_proven_one(self) -> None:
        agent, asked = asking_agent()
        app = app_for(AGUIStream(agent))

        interrupt = await ask_once(app)
        await refused(app, resolved(interrupt["id"], "forged"))

        events = await post_run(app, run_body(thread_id="t1", run_id="r3", text=None, resume=answer(interrupt, "blue")))

        assert asked.answers == ["blue"]
        assert outcome_of(events) == {"type": "success"}

    async def test_giving_up_needs_proof_too(self) -> None:
        """Ending someone else's turn is not a lesser act than answering it."""
        agent, asked = asking_agent()
        app = app_for(AGUIStream(agent))

        interrupt = await ask_once(app)
        error = await refused(app, [{"interruptId": interrupt["id"], "status": "cancelled"}])

        assert error == IsPartialDict({"code": NOT_PROVEN})
        assert asked.ending is None
