# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``Handoff`` — typed return value for dynamic routing tools.

A tool that needs to decide its target at runtime (load balancing,
state-driven dispatch, conditional pick) returns ``Handoff`` instead
of a string. The framework reads it from the agent's local
``ToolResultEvent`` stream and treats it as the packet's routing
intent.

``target`` is the participant's ``Passport.name`` — *not* a raw
``agent_id``. The framework resolves name → id at packet-finalize
using the hub's name directory, so tool code stays decoupled from
runtime IDs and portable across sessions.
"""

from dataclasses import dataclass

__all__ = ("Handoff",)


@dataclass(slots=True)
class Handoff:
    """Routing intent returned by a tool that picks its target dynamically.

    Example::

        from autogen.beta.network import Handoff


        @coord_agent.tool
        def smart_route(query: str) -> Handoff:
            target = pick_best_specialist(query)
            return Handoff(target=target, reason="routed by load")

    The framework reads this from ``ToolResultEvent.result`` after
    ``Agent.ask`` returns and uses ``target`` (resolved as a
    ``Passport.name``) plus ``reason`` as the active packet's
    ``routing`` field.
    """

    target: str
    reason: str = ""
