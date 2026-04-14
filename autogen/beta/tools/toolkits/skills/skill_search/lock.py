# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path


class SkillsLock:
    """Manages ``skills-lock.json`` for tracking installed skill hashes."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def read(self) -> dict:
        if self._path.exists():
            return json.loads(self._path.read_text(encoding="utf-8"))
        return {"version": 1, "skills": {}}

    def record(self, name: str, source: str, computed_hash: str) -> None:
        """Record or update a skill entry in the lock file."""
        data = self.read()
        data["skills"][name] = {
            "source": source,
            "sourceType": "github",
            "computedHash": computed_hash,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    def remove(self, name: str) -> None:
        """Remove a skill entry from the lock file."""
        data = self.read()
        data["skills"].pop(name, None)
        self._path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    def get_hash(self, name: str) -> str | None:
        """Return the recorded hash for a skill, or ``None``."""
        return self.read().get("skills", {}).get(name, {}).get("computedHash")
