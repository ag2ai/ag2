# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import TYPE_CHECKING

from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from autogen.beta.tools.builtin.skills import SkillsTool
from autogen.beta.tools.skills import SkillsToolkit

if TYPE_CHECKING:
    from autogen.beta import Agent


def build_card(
    agent: "Agent",
    *,
    url: str,
    version: str = "0.1.0",
    description: str | None = None,
    capabilities: AgentCapabilities | None = None,
    default_input_modes: Sequence[str] | None = None,
    default_output_modes: Sequence[str] | None = None,
    supports_extended: bool = False,
) -> AgentCard:
    """Build a default `AgentCard` for the given AG2 beta agent.

    Skills are always derived from the agent's installed skills tools
    (`SkillsTool`, `SkillsToolkit`) — they live on the agent, not the card.

    Defaults:
      - `description`     → joined `agent._system_prompt`, or `"AG2 beta agent"`
      - `version`         → `"0.1.0"`
      - `capabilities`    → `AgentCapabilities(streaming=True)`
      - `default_input_modes` / `default_output_modes` → `["text"]`

    `supports_extended=True` flips `supports_authenticated_extended_card` so
    A2A clients know to also fetch `/agent/authenticatedExtendedCard`.
    """
    return AgentCard(
        name=agent.name,
        description=description if description is not None else _default_description(agent),
        url=url,
        version=version,
        capabilities=capabilities if capabilities is not None else AgentCapabilities(streaming=True),
        default_input_modes=list(default_input_modes) if default_input_modes is not None else ["text"],
        default_output_modes=list(default_output_modes) if default_output_modes is not None else ["text"],
        skills=_extract_skills(agent),
        supports_authenticated_extended_card=True if supports_extended else None,
    )


def _default_description(agent: "Agent") -> str:
    return "\n".join(agent._system_prompt) if agent._system_prompt else "AG2 beta agent"


def _extract_skills(agent: "Agent") -> list[AgentSkill]:
    """Derive `AgentCard.skills` from skills tools installed on the agent.

    - `SkillsToolkit` (agentskills.io `SKILL.md` discovery) → one `AgentSkill`
      per discovered skill, using its name + description.
    - `SkillsTool` (provider-managed containers, e.g. Anthropic pptx/xlsx) →
      one `AgentSkill` per declared id; description carries the version pin
      when set.
    """
    out: list[AgentSkill] = []
    for t in agent.tools:
        if isinstance(t, SkillsToolkit):
            for meta in t._runtime.discover():
                out.append(
                    AgentSkill(
                        id=meta.name,
                        name=meta.name,
                        description=meta.description or f"Skill {meta.name}",
                        tags=[],
                    )
                )
        elif isinstance(t, SkillsTool):
            for s in t._skills:
                desc = f"Provider-managed skill {s.id}"
                if s.version:
                    desc += f" (v{s.version})"
                out.append(
                    AgentSkill(
                        id=s.id,
                        name=s.id,
                        description=desc,
                        tags=[],
                    )
                )
    return out
