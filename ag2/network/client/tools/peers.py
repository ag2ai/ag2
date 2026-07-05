# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``peers(action)`` — discover and describe peers in the network.

Two actions:

* ``find`` — list peers matching ``query``/``capability``/``sort_by``;
  returns a structured list the LLM can rank.
* ``describe`` — full profile (passport, resume, SKILL.md) for a single
  named peer. ``skill_md`` is the registered SKILL.md verbatim, or a
  fallback rendered from passport + resume when no SKILL.md exists.
"""

from typing import TYPE_CHECKING, Any, Literal

from ag2.tools import tool

from ..inject import AgentClientInject
from ..skill_render import render_fallback_skill

if TYPE_CHECKING:
    from ..agent_client import AgentClient

__all__ = ("make_peers_tool",)


def _passport_summary(passport: Any, resume: Any) -> dict[str, Any]:
    completed = sum(stat.completed for stat in resume.observed.values())
    total = sum(stat.n for stat in resume.observed.values())
    success_rate = (completed / total) if total else None
    return {
        "name": passport.name,
        "agent_id": passport.agent_id,
        "summary": resume.summary,
        "capabilities": list(resume.claimed_capabilities),
        "observed_success_rate": success_rate,
        "cost": (
            {
                "input_per_mtok": passport.cost.input_per_mtok,
                "output_per_mtok": passport.cost.output_per_mtok,
                "latency_tier": passport.cost.latency_tier,
            }
            if passport.cost is not None
            else None
        ),
    }


def make_peers_tool(agent_client: "AgentClient") -> object:
    """Return a closure-bound ``peers`` tool."""

    @tool
    async def peers(
        action: Literal["find", "describe"],
        *,
        query: str | None = None,
        capability: str | None = None,
        sort_by: Literal["name", "cost", "track_record"] | None = None,
        name: str | None = None,
        limit: int = 20,
        client: AgentClientInject = None,
    ) -> list[dict] | dict | str:
        """Discover and describe peers.

        ``find``:    args query?, capability?, sort_by?, limit
        ``describe``: args name (or agent_id) → {passport, resume, skill_md}
        """
        actual = client if client is not None else agent_client
        hub = actual._hub_client
        if action == "find":
            passports = await hub.list_agents(capability=capability, query=query, limit=limit)
            results: list[dict] = []
            for p in passports:
                if p.agent_id == actual.agent_id:
                    continue  # don't include the calling agent
                resume = await hub.get_resume(p.agent_id)
                results.append(_passport_summary(p, resume))
            if sort_by == "name":
                results.sort(key=lambda r: r["name"])
            elif sort_by == "cost":
                results.sort(key=lambda r: (r["cost"] or {}).get("input_per_mtok") or float("inf"))
            elif sort_by == "track_record":
                results.sort(
                    key=lambda r: r["observed_success_rate"] or 0.0,
                    reverse=True,
                )
            return results

        if action == "describe":
            if not name:
                return "Error: describe requires `name` (or agent_id)"
            try:
                passport = await hub.get_agent(name)
            except Exception:
                return f"Error: peer {name!r} not found"
            assert passport.agent_id is not None
            resume = await hub.get_resume(passport.agent_id)
            skill_md = await hub.get_skill(passport.agent_id)
            if not skill_md:
                skill_md = render_fallback_skill(passport, resume)
            return {
                "passport": {
                    "name": passport.name,
                    "agent_id": passport.agent_id,
                    "owner": passport.owner,
                    "provider": passport.provider,
                    "model": passport.model,
                },
                "resume": {
                    "summary": resume.summary,
                    "claimed_capabilities": list(resume.claimed_capabilities),
                    "domains": list(resume.domains),
                    "observed": {
                        cap: {
                            "n": s.n,
                            "completed": s.completed,
                            "failed": s.failed,
                            "expired": s.expired,
                        }
                        for cap, s in resume.observed.items()
                    },
                },
                "skill_md": skill_md,
            }

        return f"Error: unknown action {action!r}; choose from find, describe"

    return peers
