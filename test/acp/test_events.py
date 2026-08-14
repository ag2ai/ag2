# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ag2.acp.events import (
    ACPAvailableCommands,
    ACPElicitation,
    ACPModeChange,
    ACPPlan,
    ACPPlanEntry,
)
from ag2.events import BaseEvent


def test_acp_plan_holds_entries() -> None:
    plan = ACPPlan(entries=[ACPPlanEntry(content="step 1", status="pending", priority="high")])
    assert isinstance(plan, BaseEvent)
    assert plan.entries == [ACPPlanEntry(content="step 1", status="pending", priority="high")]


def test_mode_change() -> None:
    assert ACPModeChange(mode_id="edit").mode_id == "edit"


def test_available_commands() -> None:
    assert ACPAvailableCommands(commands=["/test"]).commands == ["/test"]


def test_elicitation_carries_the_question() -> None:
    url = ACPElicitation("Authorize access", "url", url="https://example.com/auth")
    assert isinstance(url, BaseEvent)
    assert (url.message, url.mode, url.url, url.fields) == ("Authorize access", "url", "https://example.com/auth", [])

    form = ACPElicitation("Which strategy?", "form", fields=["strategy"])
    assert (form.url, form.fields) == (None, ["strategy"])
