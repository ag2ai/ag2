# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``Agent.ask`` and ``Agent.run`` with several messages, as the checker sees them.

Checked by ``test/typing/test_fixtures.py``; not imported by the suite. ``reveal_type``
lines are the assertions — mypy reports them as notes.
"""

from ag2 import Agent, ImageInput
from ag2.response import ResponseSchema


async def several_messages(agent: Agent) -> None:
    image = ImageInput("https://example.com/cat.png")

    # The implementation takes ``*msg``, so every overload must too.
    bare = await agent.ask("Describe", image)
    reveal_type(bare)  # N: Revealed type is "ag2.agent.AgentReply[str, str]"
    none = await agent.ask("Describe", image, response_schema=None)
    reveal_type(none)  # N: Revealed type is "ag2.agent.AgentReply[str, str]"
    typed = await agent.ask("Describe", image, response_schema=int)
    reveal_type(typed)  # N: Revealed type is "ag2.agent.AgentReply[int, str]"
    proto = await agent.ask("Describe", image, response_schema=ResponseSchema(int))
    reveal_type(proto)  # N: Revealed type is "ag2.agent.AgentReply[int, str]"

    async with agent.run("Describe", image) as run:
        reveal_type(run)  # N: Revealed type is "ag2.agent.AgentRun[str, str]"


# Messages are positional: the keyword never reached the implementation. Unformatted, so the
# expectation stays on the line mypy reports.
# fmt: off
Agent("a").ask(msg="x")  # E: Unexpected keyword argument "msg" for overloaded function "ask" of "Agent"  [call-overload]
# fmt: on
