# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""An event field's default, as the checker sees it.

Checked by ``test/typing/test_fixtures.py``; not imported by the suite.
"""

from ag2.events import BaseEvent, Field, TaskCancelled, TaskProgress


class Order(BaseEvent):
    label: str = Field(default="")


# A field with a default may be left out.
Order()
TaskProgress(task_id="t", agent_name="a", objective="o")
TaskCancelled(task_id="t", agent_name="a", objective="o")


# The checker only reads ``default=`` by name — a positional default would silently make the
# field required — so the positional form is refused. Unformatted, so the expectation stays on
# the line mypy reports.
# fmt: off
class Positional(BaseEvent):
    label: str = Field("")  # E: Too many positional arguments for "Field"  [call-arg]  # N: "Field" defined in "ag2.events.base"
# fmt: on
