# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for packet routing resolution.

Pins the contract that ``WorkflowAdapter._resolve_routing`` walks
agent local-stream events in emission order and applies first-emit-
wins across both static (``ToolCalled`` graph rule) and dynamic
(``Handoff`` typed return) routing.
"""

import json

from ag2.events import DataInput, ToolCallEvent, ToolResult, ToolResultEvent
from ag2.network import (
    AgentTarget,
    Finish,
    Handoff,
    TerminateTarget,
    ToolCalled,
    Transition,
    TransitionGraph,
)
from ag2.network.adapters.workflow import _resolve_routing


def _call(name: str, *, call_id: str = "id1", reason: str = "") -> ToolCallEvent:
    args = json.dumps({"reason": reason}) if reason else "{}"
    return ToolCallEvent(id=call_id, name=name, arguments=args)


def _result(parent_id: str, *, name: str | None = None, value=None) -> ToolResultEvent:
    """Build a ToolResultEvent. ``value`` becomes the ``result.parts[0]``
    payload — a Handoff goes into a DataInput, a string into TextInput
    via Input.ensure_input."""
    return ToolResultEvent(
        parent_id=parent_id,
        name=name,
        result=ToolResult.ensure_result(value if value is not None else "ok"),
    )


def _graph(*tools: str) -> TransitionGraph:
    """Build a minimal graph with ToolCalled rules for each tool name."""
    transitions = [Transition(when=ToolCalled(t), then=AgentTarget(f"agent_{t}")) for t in tools]
    return TransitionGraph(
        initial_speaker="agent_initial",
        transitions=transitions,
        default_target=TerminateTarget(reason="default"),
    )


class TestResolveRouting:
    def test_no_events_returns_text(self) -> None:
        graph = _graph("delegate_a")
        routing = _resolve_routing([], graph, name_to_id={})
        assert routing == {"kind": "text"}

    def test_no_graph_returns_text(self) -> None:
        events = [_call("delegate_a")]
        routing = _resolve_routing(events, None, name_to_id={})
        assert routing == {"kind": "text"}

    def test_single_static_tool_call_routes(self) -> None:
        """A single ToolCallEvent matching a ToolCalled rule sets
        routing.tool; routing.target left unset (fold's select_next
        resolves it)."""
        graph = _graph("delegate_a")
        events = [_call("delegate_a", call_id="c1", reason="go")]
        routing = _resolve_routing(events, graph, name_to_id={})
        assert routing == {
            "kind": "handoff",
            "tool": "delegate_a",
            "reason": "go",
        }

    def test_unmatched_tool_call_returns_text(self) -> None:
        """Tool whose name doesn't match any ToolCalled rule falls
        through to text routing."""
        graph = _graph("delegate_a")
        events = [_call("some_other_tool", call_id="c1")]
        routing = _resolve_routing(events, graph, name_to_id={})
        assert routing == {"kind": "text"}

    def test_two_static_tools_first_emit_wins(self) -> None:
        """Two parallel ToolCallEvents both matching ToolCalled rules:
        the FIRST in emission order wins routing."""
        graph = _graph("delegate_a", "delegate_b")
        events = [
            _call("delegate_a", call_id="c1", reason="first"),
            _call("delegate_b", call_id="c2", reason="second"),
        ]
        routing = _resolve_routing(events, graph, name_to_id={})
        assert routing["tool"] == "delegate_a"
        assert routing["reason"] == "first"

    def test_dynamic_handoff_resolves_target(self) -> None:
        """A ToolResultEvent carrying a Handoff dataclass produces
        dynamic routing with the resolved target id."""
        graph = _graph("delegate_a")
        call = _call("smart_route", call_id="c1")
        result = _result("c1", name="smart_route", value=Handoff(target="researcher", reason="picked"))
        events = [call, result]
        routing = _resolve_routing(
            events,
            graph,
            name_to_id={"researcher": "agent-id-researcher"},
        )
        assert routing == {
            "kind": "handoff",
            "tool": "smart_route",
            "reason": "picked",
            "target": "agent-id-researcher",
        }

    def test_dynamic_handoff_unresolved_name_falls_through_to_name(self) -> None:
        """If the Handoff's target isn't in name_to_id, the name itself
        is used (caller's responsibility to handle UnknownHandoffTarget)."""
        graph = _graph("delegate_a")
        events = [
            _call("smart_route", call_id="c1"),
            _result("c1", name="smart_route", value=Handoff(target="unknown_name")),
        ]
        routing = _resolve_routing(events, graph, name_to_id={})
        assert routing["target"] == "unknown_name"

    def test_dynamic_overrides_static_for_same_tool(self) -> None:
        """A tool registered via ToolCalled AND returning Handoff:
        the dynamic Handoff wins for that tool's call."""
        graph = _graph("delegate_a")
        events = [
            _call("delegate_a", call_id="c1"),
            _result("c1", name="delegate_a", value=Handoff(target="explicit_target")),
        ]
        routing = _resolve_routing(
            events,
            graph,
            name_to_id={"explicit_target": "agent-id-explicit"},
        )
        # Dynamic Handoff present → routing carries target.
        assert routing["target"] == "agent-id-explicit"

    def test_dynamic_first_static_second_first_wins(self) -> None:
        """First emission wins regardless of dynamic vs static."""
        graph = _graph("delegate_b")
        events = [
            _call("smart_route", call_id="c1"),  # dynamic, first
            _result("c1", name="smart_route", value=Handoff(target="alice")),
            _call("delegate_b", call_id="c2"),  # static, second
        ]
        routing = _resolve_routing(events, graph, name_to_id={"alice": "agent-id-alice"})
        assert routing["tool"] == "smart_route"
        assert routing["target"] == "agent-id-alice"

    def test_static_first_dynamic_second_first_wins(self) -> None:
        """First emission wins regardless of dynamic vs static (reversed)."""
        graph = _graph("delegate_a")
        events = [
            _call("delegate_a", call_id="c1"),  # static, first
            _call("smart_route", call_id="c2"),  # dynamic, second
            _result("c2", name="smart_route", value=Handoff(target="bob")),
        ]
        routing = _resolve_routing(events, graph, name_to_id={"bob": "agent-id-bob"})
        assert routing["tool"] == "delegate_a"
        # No target field for static (fold's select_next resolves).
        assert "target" not in routing

    def test_string_result_not_handoff(self) -> None:
        """A tool that returns a plain string produces no dynamic
        routing — only Handoff instances trigger that path."""
        graph = _graph("delegate_a")
        events = [
            _call("delegate_a", call_id="c1"),
            _result("c1", name="delegate_a", value="just a string"),
        ]
        routing = _resolve_routing(events, graph, name_to_id={})
        # Static routing still applies (tool name matches).
        assert routing["tool"] == "delegate_a"
        assert "target" not in routing

    def test_data_input_non_handoff_ignored(self) -> None:
        """A DataInput whose data isn't a Handoff is not routing-relevant."""
        graph = _graph("delegate_a")
        result = ToolResultEvent(
            parent_id="c1",
            name="other_tool",
            result=ToolResult.ensure_result(DataInput(data={"some": "data"})),
        )
        events = [_call("other_tool", call_id="c1"), result]
        routing = _resolve_routing(events, graph, name_to_id={})
        # No matching rule, no Handoff → text routing.
        assert routing == {"kind": "text"}

    def test_results_without_calls_ignored(self) -> None:
        """Orphan ToolResultEvent (no matching call) doesn't drive routing."""
        graph = _graph("delegate_a")
        events = [_result("orphan", value=Handoff(target="alice"))]
        routing = _resolve_routing(events, graph, name_to_id={"alice": "agent-id-alice"})
        # No calls at all → text routing.
        assert routing == {"kind": "text"}

    def test_dynamic_finish_terminates(self) -> None:
        """A ToolResultEvent carrying a Finish dataclass produces
        routing with ``kind: "finish"`` so fold can close the channel."""
        graph = _graph("delegate_a")
        events = [
            _call("finish", call_id="c1"),
            _result("c1", name="finish", value=Finish(summary="all done", reason="done")),
        ]
        routing = _resolve_routing(events, graph, name_to_id={})
        assert routing == {
            "kind": "finish",
            "tool": "finish",
            "reason": "done",
            "summary": "all done",
        }

    def test_finish_default_reason_propagates(self) -> None:
        """Finish() with no args still produces a valid termination routing."""
        graph = _graph("delegate_a")
        events = [
            _call("finish", call_id="c1"),
            _result("c1", name="finish", value=Finish()),
        ]
        routing = _resolve_routing(events, graph, name_to_id={})
        assert routing == {
            "kind": "finish",
            "tool": "finish",
            "reason": "finished",
            "summary": "",
        }

    def test_finish_beats_handoff_in_same_emission(self) -> None:
        """If a single tool somehow returns Finish, it beats a later
        Handoff from another call — first-emitted-wins extends to
        Finish."""
        graph = _graph("delegate_a")
        events = [
            _call("finish", call_id="c1"),
            _result("c1", name="finish", value=Finish(reason="early")),
            _call("delegate_a", call_id="c2"),
            _result("c2", name="delegate_a", value=Handoff(target="alice")),
        ]
        routing = _resolve_routing(events, graph, name_to_id={"alice": "agent-id-alice"})
        assert routing["kind"] == "finish"
        assert routing["tool"] == "finish"

    def test_handoff_before_finish_still_first_wins(self) -> None:
        """First emission still wins — a Handoff that fires before a
        later Finish takes precedence."""
        graph = _graph("delegate_a")
        events = [
            _call("delegate_a", call_id="c1"),
            _result("c1", name="delegate_a", value=Handoff(target="alice")),
            _call("finish", call_id="c2"),
            _result("c2", name="finish", value=Finish(reason="late")),
        ]
        routing = _resolve_routing(events, graph, name_to_id={"alice": "agent-id-alice"})
        assert routing["kind"] == "handoff"
        assert routing["target"] == "agent-id-alice"
