# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""SKILL.md frontmatter parser + fallback renderer.

The Anthropic SKILL.md format is a Markdown body with optional
``---``-delimited YAML-like frontmatter at the top:

.. code-block:: markdown

   ---
   name: alice
   description: Senior policy analyst.
   ---

   ## What I do

   ...

This module ships:

* :func:`parse_skill_frontmatter` — splits ``(frontmatter, body)`` for
  the ``peers(action="describe")`` tool's structured response.
* :func:`render_fallback_skill` — builds a SKILL.md-style document from
  passport + resume when no ``SKILL.md`` is registered, so the LLM
  always has a uniform read.

Only parses scalar ``key: value`` lines (no nested mappings, lists, or
block scalars). This matches what Anthropic's skill harness recognises
and keeps the parser dependency-free.
"""

from typing import Any

from ..identity import Passport, Resume

__all__ = (
    "ParsedSkill",
    "parse_skill_frontmatter",
    "render_fallback_skill",
)


class ParsedSkill(dict):
    """``{"frontmatter": dict, "body": str}`` with ``.frontmatter`` and ``.body`` attrs.

    Subclasses ``dict`` so it round-trips as JSON without conversion;
    attribute access keeps call-site code readable.
    """

    @property
    def frontmatter(self) -> dict[str, Any]:
        return self["frontmatter"]

    @property
    def body(self) -> str:
        return self["body"]


_FENCE = "---"


def parse_skill_frontmatter(md: str) -> ParsedSkill:
    """Split a SKILL.md body into ``(frontmatter, body)``.

    Returns ``{"frontmatter": {}, "body": md}`` when no frontmatter
    fence is present or the closing fence is missing — the rest of the
    document is treated as the body verbatim.
    """
    if not md.startswith(_FENCE):
        return ParsedSkill(frontmatter={}, body=md)
    # Find the closing fence on its own line after the opening one.
    rest = md[len(_FENCE) :]
    if rest.startswith("\n"):
        rest = rest[1:]
    closing = rest.find(f"\n{_FENCE}")
    if closing == -1:
        return ParsedSkill(frontmatter={}, body=md)
    header = rest[:closing]
    body = rest[closing + len(_FENCE) + 1 :]  # skip "\n---"
    if body.startswith("\n"):
        body = body[1:]

    frontmatter: dict[str, Any] = {}
    for line in header.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        frontmatter[key.strip()] = value.strip()

    return ParsedSkill(frontmatter=frontmatter, body=body)


def render_fallback_skill(passport: Passport, resume: Resume) -> str:
    """Build a SKILL.md-style document from passport + resume.

    Used by ``peers(action="describe")`` when an agent has no
    registered ``SKILL.md``. The output mirrors the frontmatter shape
    Anthropic skills use so the calling LLM gets the same structure
    either way.
    """
    description = resume.summary or "Network-registered agent."
    lines: list[str] = [
        "---",
        f"name: {passport.name}",
        f"description: {description}",
        "---",
        "",
    ]
    if resume.claimed_capabilities:
        lines.append("## Capabilities")
        lines.extend(f"- {cap}" for cap in resume.claimed_capabilities)
        lines.append("")
    if resume.domains:
        lines.append("## Domains")
        lines.extend(f"- {d}" for d in resume.domains)
        lines.append("")
    if resume.observed:
        lines.append("## Track record")
        for cap in sorted(resume.observed.keys()):
            stat = resume.observed[cap]
            lines.append(
                f"- {cap}: {stat.completed} completed / {stat.failed} failed / {stat.expired} expired (n={stat.n})"
            )
        lines.append("")
    if resume.examples:
        lines.append("## Examples")
        for ex in resume.examples:
            lines.append(f"- {ex.title} — {ex.outcome or 'no outcome'}")
        lines.append("")
    # Trim the trailing blank line for cleanliness.
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"
