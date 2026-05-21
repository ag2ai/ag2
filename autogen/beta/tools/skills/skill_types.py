# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SkillMetadata:
    """Metadata parsed from a skill's SKILL.md frontmatter."""

    name: str
    description: str
    path: Path
    has_scripts: bool
    version: str | None = None
    license: str | None = None
    compatibility: str | None = None
