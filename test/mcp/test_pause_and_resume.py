# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A served agent asking a *modern-era* client, which has no back-channel to ask over.

Revision 2026-07-28 defines no server-to-client request, so the question comes
back as the result of the call and the client answers by retrying. These tests
drive that from the client's side of the wire and never assert how the pause is
stored, keyed or sealed. The handshake era's inline path is covered in
``test_elicitation.py``.
"""

import pytest
from mcp.server.request_state import RequestStateBoundary, RequestStateSecurity, authenticated_principal
from mcp.shared.exceptions import MCPError
from mcp.types import ElicitRequest, InputRequiredResult, TextContent

from ag2.mcp import MCPServer
from ag2.mcp.pause import SuspendedTurn
from ag2.mcp.testing import connect_modern
from ag2.stream import MemoryStream

from ._helpers import (
    Asked,
    accepting,
    answer,
    ask,
    asking_agent,
    asks_twice,
    declares_elicitation,
    first_text,
    outstanding,
    settle,
)


@pytest.mark.asyncio
class TestModernEraPause:
    async def test_the_question_comes_back_as_the_calls_result(self) -> None:
        """Not an error and not a hang: the outstanding question, plus state."""
        server = MCPServer(asking_agent())

        async with connect_modern(server, elicitation_callback=declares_elicitation) as session:
            first = await ask(session)

        assert isinstance(first, InputRequiredResult)
        assert first.request_state is not None
        _key, request = outstanding(first)
        assert isinstance(request, ElicitRequest)
        assert request.params.message == "What colour?"

    async def test_an_answered_retry_completes_the_turn(self) -> None:
        asked = Asked()
        server = MCPServer(asking_agent(asked))

        async with connect_modern(server, elicitation_callback=declares_elicitation) as session:
            first = await ask(session)
            assert isinstance(first, InputRequiredResult)
            final = await answer(session, first, accepting("blue"))

        assert asked.answers == ["blue"]
        assert not isinstance(final, InputRequiredResult)
        assert final.is_error is False
        reply, _trailer = final.content
        assert reply == TextContent(type="text", text="done")

    async def test_the_run_resumed_rather_than_restarted(self) -> None:
        """The whole point: the work already done — and paid for — is not thrown away."""
        asked = Asked()
        server = MCPServer(asking_agent(asked))

        async with connect_modern(server, elicitation_callback=declares_elicitation) as session:
            first = await ask(session)
            assert isinstance(first, InputRequiredResult)
            await answer(session, first, accepting("blue"))

        assert asked.runs == 1, "the tool body ran again, so the run restarted rather than resumed"

    async def test_a_second_question_pauses_again(self) -> None:
        asked = Asked()
        server = MCPServer(asking_agent(asked, questions=("First?", "Second?")))

        async with connect_modern(server, elicitation_callback=declares_elicitation) as session:
            first = await ask(session)
            assert isinstance(first, InputRequiredResult)
            second = await answer(session, first, accepting("one"))
            assert isinstance(second, InputRequiredResult)
            _key, request = outstanding(second)
            assert request.params.message == "Second?"
            final = await answer(session, second, accepting("two"))

        assert asked.answers == ["one", "two"]
        assert not isinstance(final, InputRequiredResult)
        assert final.is_error is False

    async def test_a_stale_answer_is_not_consumed_and_the_question_is_re_asked(self) -> None:
        """An answer minted for a question the run has moved past must not be applied."""
        server = MCPServer(asking_agent(questions=("First?", "Second?")))

        async with connect_modern(server, elicitation_callback=declares_elicitation) as session:
            first = await ask(session)
            assert isinstance(first, InputRequiredResult)
            stale_key, _request = outstanding(first)
            second = await answer(session, first, accepting("one"))
            assert isinstance(second, InputRequiredResult)
            # Round 1's answer replayed against round 2's state: the key names a
            # question this run is no longer waiting on.
            third = await ask(
                session,
                input_responses={stale_key: accepting("one")},
                request_state=second.request_state,
            )

        assert isinstance(third, InputRequiredResult), "a stale answer completed the round"
        _key, request = outstanding(third)
        assert request.params.message == "Second?"

    async def test_a_client_that_cannot_answer_is_never_asked(self) -> None:
        """No elicitation capability declared, so the existing failure surfaces instead."""
        async with connect_modern(MCPServer(asking_agent()), raise_exceptions=False) as session:
            result = await ask(session)

        assert not isinstance(result, InputRequiredResult)
        assert result.is_error is True
        assert "Human input was requested but not provided" in first_text(result)


@pytest.mark.asyncio
class TestTheRunRefusesAStaleAnswer:
    """The seam the wire test above cannot reach: the run's own refusal.

    The serving path drops an answer whose key names nothing, so it never calls the
    run — leaving the half that would let an answer through untested from outside.
    """

    async def test_an_answer_to_an_earlier_question_is_refused(self) -> None:
        turn = SuspendedTurn(conversation=None, stream=MemoryStream(), created=0.0)
        turn.start(asks_twice(turn))
        await settle()
        assert turn.outstanding is not None
        (first_key, _first) = turn.outstanding

        assert turn.answer(first_key, accepting("one")) is True
        await settle()
        assert turn.outstanding is not None and turn.outstanding[0] != first_key, "the run did not move on"

        assert turn.answer(first_key, accepting("one again")) is False, "an answer to the first question was consumed"
        assert turn.outstanding is not None, "refusing it also cleared the question the run is waiting on"

        turn.reclaim()

    async def test_an_answer_to_a_run_waiting_on_nothing_is_refused(self) -> None:
        turn = SuspendedTurn(conversation=None, stream=MemoryStream(), created=0.0)

        assert turn.answer("whatever", accepting("blue")) is False


class TestTheStateIsBoundToItsCaller:
    """A retry presenting state minted for another caller must not resume this run.

    The first test drives it from the wire; the other two assert the seam, which is
    the only place a default that stopped binding would show.
    """

    @pytest.mark.asyncio
    async def test_a_retry_from_a_different_caller_is_refused(self) -> None:
        """Refused fail-closed, and the run still there for the caller it belongs to."""
        caller = ["alice"]
        asked = Asked()
        server = MCPServer(
            asking_agent(asked),
            request_state_security=RequestStateSecurity(keys=[b"k" * 32], bind_principal=lambda _ctx: caller[0]),
        )

        async with connect_modern(server, raise_exceptions=False, elicitation_callback=declares_elicitation) as session:
            first = await ask(session)
            assert isinstance(first, InputRequiredResult)

            caller[0] = "bob"
            with pytest.raises(MCPError) as raised:
                await answer(session, first, accepting("blue"))

            caller[0] = "alice"
            answered = await answer(session, first, accepting("blue"))

        assert "requestState" in str(raised.value)
        assert not isinstance(answered, InputRequiredResult), "the rightful caller could no longer resume its own run"
        assert asked.answers == ["blue"], "the answer reached the run exactly once"

    def test_the_boundary_is_installed_with_its_principal_binding_intact(self) -> None:
        server = MCPServer(asking_agent(), name="pauser")

        boundaries = [m for m in server.server.middleware if isinstance(m, RequestStateBoundary)]

        assert len(boundaries) == 1, "the lowlevel tier installs none by default; this server must install one"
        (boundary,) = boundaries
        security = boundary._security
        assert security.bind_principal is authenticated_principal, "state would not be bound to its caller"
        assert boundary._audience == "pauser", "state minted by another service sharing these keys would be accepted"

    def test_a_supplied_policy_is_the_one_installed(self) -> None:
        """An operator sharing a key across replicas must actually get their policy."""
        policy = RequestStateSecurity(keys=[b"k" * 32], ttl=45.0)

        server = MCPServer(asking_agent(), request_state_security=policy)

        (boundary,) = [m for m in server.server.middleware if isinstance(m, RequestStateBoundary)]
        assert boundary._security is policy
