# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Optional ``ModelConfig`` capability protocols.

Live next to ``ModelConfig`` so that subagent / tool runtime code can detect
them without taking a dependency on any specific provider package.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SubtaskContextSeeder(Protocol):
    """Optional ``ModelConfig`` capability — pre-seed parent ``Context.variables``
    with protocol-level state that must be shared by all subtasks.

    Called by ``subagent_tool.delegate`` once, before a sub-task spawns.
    Implementations should use ``dict.setdefault`` (atomic in CPython) so
    concurrent first-time sub-task spawns converge on the same value.
    """

    def seed_subtask_variables(self, parent_vars: dict[str, Any]) -> None: ...
