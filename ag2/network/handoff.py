# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Typed routing-intent returns: ``Handoff`` (redirect) / ``Finish`` (terminate).

A tool that needs to decide its target at runtime (load balancing,
state-driven dispatch, conditional pick) returns ``Handoff`` instead
of a string. A tool that wants to end the channel cleanly returns
``Finish``. The framework reads either from the agent's local
``ToolResultEvent`` stream and treats it as the packet's routing
intent.

For ``Handoff``, ``target`` is the participant's ``Passport.name`` —
*not* a raw ``agent_id``. The framework resolves name → id at
packet-finalize using the hub's name directory, so tool code stays
decoupled from runtime IDs and portable across channels.
"""

from dataclasses import dataclass

__all__ = ("Finish", "Handoff")


@dataclass(slots=True)
class Handoff:
    """Routing intent returned by a tool that picks its target dynamically.

    Example::

        from ag2.network import Handoff


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


@dataclass(slots=True)
class Finish:
    """Routing intent returned by a tool that ends the channel cleanly.

    Example::

        from ag2.network import Finish


        @coord_agent.tool
        def finish(summary: str) -> Finish:
            return Finish(summary=summary)

    When the framework finds a ``Finish`` on a ``ToolResultEvent`` it
    closes the channel — equivalent to a ``TerminateTarget`` rule
    firing, but driven by the tool's runtime decision rather than a
    static graph transition. ``reason`` populates
    ``ChannelMetadata.close_reason``; ``summary`` rides on the
    packet's ``routing.summary`` field for callers / observability.
    """

    summary: str = ""
    reason: str = "finished"
