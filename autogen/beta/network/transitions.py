# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``Transition`` vocabulary for orchestrated workflows.

The pieces:

* :class:`TransitionTarget` — Protocol; ``resolve(state, envelope) ->
  TransitionDecision`` says where the next turn goes. Built-ins:
  ``AgentTarget``, ``RoundRobinTarget``, ``StayTarget``,
  ``RevertToInitiatorTarget``, ``TerminateTarget``.
* :class:`TransitionCondition` — Protocol; ``evaluate(state, envelope)
  -> bool`` says when a transition fires. Built-ins: ``Always``,
  ``FromSpeaker``, ``ToolCalled``.
* :class:`Transition` — pairs ``when`` + ``then`` with a ``priority``.
* :class:`TransitionGraph` — ``initial_speaker`` + ordered list of
  ``Transition``s + ``default_target`` + optional ``max_turns``.

The graph is data: ``to_dict()`` / ``loads()`` round-trip via named
registries (``register_target`` / ``register_condition``) so it
persists in ``ChannelMetadata.knobs["graph"]`` and survives
``Hub.hydrate()``. Custom targets / conditions plug in by registering
under a unique ``name``.

``resolve`` / ``evaluate`` deliberately take only ``(state, envelope)``
— no metadata. ``WorkflowState`` carries ``participant_order`` and
``creator_id`` so transitions can be evaluated inside
``WorkflowAdapter.fold``, which has no metadata.
"""

import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

from .envelope import EV_PACKET, Envelope
from .errors import NetworkError

if TYPE_CHECKING:
    from .adapters.workflow import WorkflowState

__all__ = (
    "AgentTarget",
    "Always",
    "ContextEquals",
    "FromSpeaker",
    "RevertToInitiatorTarget",
    "RoundRobinTarget",
    "StayTarget",
    "TerminateTarget",
    "ToolCalled",
    "Transition",
    "TransitionCondition",
    "TransitionDecision",
    "TransitionGraph",
    "TransitionRegistry",
    "TransitionTarget",
    "WorkflowGraphError",
    "register_condition",
    "register_target",
)


class WorkflowGraphError(NetworkError):
    """Raised when a registry lookup fails or graph deserialisation breaks."""


@dataclass(slots=True)
class TransitionDecision:
    """What a target wants the workflow to do next.

    ``next_speaker=None`` terminates the channel; ``close_reason``
    populates ``ChannelMetadata.close_reason`` on the resulting
    ``EV_CHANNEL_CLOSED``.
    """

    next_speaker: str | None
    close_reason: str = ""


class TransitionTarget(Protocol):
    """Where the next turn goes. Pure resolver — no I/O, no awaitables."""

    name: ClassVar[str]

    def resolve(
        self,
        state: "WorkflowState",
        envelope: Envelope,
    ) -> TransitionDecision: ...


class TransitionCondition(Protocol):
    """When a transition fires. Pure predicate — no I/O."""

    name: ClassVar[str]

    def evaluate(
        self,
        state: "WorkflowState",
        envelope: Envelope,
    ) -> bool: ...


@dataclass(slots=True)
class Transition:
    """One rule: if ``when.evaluate`` is true, take ``then``.

    Lower ``priority`` checks first; ties resolve in list order.
    """

    when: TransitionCondition
    then: TransitionTarget
    priority: int = 0


# ── Built-in TransitionTargets ──────────────────────────────────────────────


@dataclass(slots=True)
class AgentTarget:
    """Resolve to a named agent."""

    agent_id: str
    name: ClassVar[str] = "agent"

    def resolve(self, state: "WorkflowState", envelope: Envelope) -> TransitionDecision:
        return TransitionDecision(next_speaker=self.agent_id)


@dataclass(slots=True)
class RoundRobinTarget:
    """Next participant after ``state.last_speaker_id`` in ``participant_order``."""

    name: ClassVar[str] = "round_robin"

    def resolve(self, state: "WorkflowState", envelope: Envelope) -> TransitionDecision:
        order = state.participant_order
        if not order:
            return TransitionDecision(next_speaker=None, close_reason="no_participants")
        anchor = state.last_speaker_id or envelope.sender_id
        if anchor in order:
            idx = order.index(anchor)
            next_idx = (idx + 1) % len(order)
        else:
            next_idx = 0
        return TransitionDecision(next_speaker=order[next_idx])


@dataclass(slots=True)
class StayTarget:
    """Same speaker again."""

    name: ClassVar[str] = "stay"

    def resolve(self, state: "WorkflowState", envelope: Envelope) -> TransitionDecision:
        return TransitionDecision(next_speaker=state.last_speaker_id or envelope.sender_id)


@dataclass(slots=True)
class RevertToInitiatorTarget:
    """Back to the channel creator."""

    name: ClassVar[str] = "revert_to_initiator"

    def resolve(self, state: "WorkflowState", envelope: Envelope) -> TransitionDecision:
        return TransitionDecision(next_speaker=state.creator_id)


@dataclass(slots=True)
class TerminateTarget:
    """End the channel with ``reason``."""

    reason: str = "after_work"
    name: ClassVar[str] = "terminate"

    def resolve(self, state: "WorkflowState", envelope: Envelope) -> TransitionDecision:
        return TransitionDecision(next_speaker=None, close_reason=self.reason)


# ── Built-in TransitionConditions ───────────────────────────────────────────


@dataclass(slots=True)
class Always:
    """Always fires."""

    name: ClassVar[str] = "always"

    def evaluate(self, state: "WorkflowState", envelope: Envelope) -> bool:
        return True


@dataclass(slots=True)
class FromSpeaker:
    """Fires when the just-accepted envelope was sent by ``agent_id``."""

    agent_id: str
    name: ClassVar[str] = "from_speaker"

    def evaluate(self, state: "WorkflowState", envelope: Envelope) -> bool:
        return envelope.sender_id == self.agent_id


@dataclass(slots=True)
class ToolCalled:
    """Fires when the just-accepted envelope's routing tool matches.

    For ``EV_PACKET`` envelopes (the standard packet-model path),
    reads ``event_data["routing"]["tool"]``. The packet's routing
    field is populated by the framework from the agent's local-stream
    ``ToolCallEvent`` whose name matches a registered routing tool.
    """

    tool_name: str
    name: ClassVar[str] = "tool_called"

    def evaluate(self, state: "WorkflowState", envelope: Envelope) -> bool:
        if envelope.event_type != EV_PACKET:
            return False
        routing = envelope.event_data.get("routing", {}) or {}
        return routing.get("tool") == self.tool_name


@dataclass(slots=True)
class ContextEquals:
    """Fires when ``state.context_vars[key]`` equals ``value``.

    The channel-scoped context dict is mutated by ``ag2.context.set``
    envelopes; this condition is the read side. Missing keys compare
    as ``None`` so ``ContextEquals(key="foo", value=None)`` fires
    when ``foo`` has never been set or was deleted.
    """

    key: str
    value: Any
    name: ClassVar[str] = "context_equals"

    def evaluate(self, state: "WorkflowState", envelope: Envelope) -> bool:
        return state.context_vars.get(self.key) == self.value


# ── Registry ────────────────────────────────────────────────────────────────


_BUILTIN_TARGETS: tuple[type[TransitionTarget], ...] = (
    AgentTarget,
    RoundRobinTarget,
    StayTarget,
    RevertToInitiatorTarget,
    TerminateTarget,
)

_BUILTIN_CONDITIONS: tuple[type[TransitionCondition], ...] = (
    Always,
    FromSpeaker,
    ToolCalled,
    ContextEquals,
)


class TransitionRegistry:
    """Per-(process, instance) registry of transition target / condition classes.

    Constructed pre-populated with the built-ins
    (``AgentTarget`` / ``RoundRobinTarget`` / ``StayTarget`` /
    ``RevertToInitiatorTarget`` / ``TerminateTarget`` and
    ``Always`` / ``FromSpeaker`` / ``ToolCalled``).

    Tests / multi-tenant callers that need isolation construct their
    own and pass to ``TransitionGraph.loads(data, registry=)``. The
    module-level ``register_target`` / ``register_condition`` helpers
    delegate to :meth:`default` — a class-cached lazily-initialised
    singleton — for the common single-tenant case.
    """

    _DEFAULT: ClassVar["TransitionRegistry | None"] = None

    def __init__(self) -> None:
        self._targets: dict[str, type[TransitionTarget]] = {cls.name: cls for cls in _BUILTIN_TARGETS}
        self._conditions: dict[str, type[TransitionCondition]] = {cls.name: cls for cls in _BUILTIN_CONDITIONS}

    @classmethod
    def default(cls) -> "TransitionRegistry":
        """Return the lazily-initialised process-wide default registry.

        Mutated by the module-level ``register_target`` /
        ``register_condition`` helpers. Tests that need isolation should
        construct a fresh ``TransitionRegistry`` instead and pass it to
        ``TransitionGraph.loads(..., registry=)``.
        """
        if cls._DEFAULT is None:
            cls._DEFAULT = cls()
        return cls._DEFAULT

    def register_target(self, target_cls: type[TransitionTarget]) -> None:
        """Register a custom :class:`TransitionTarget`. Re-registers replace."""
        self._targets[target_cls.name] = target_cls

    def register_condition(self, condition_cls: type[TransitionCondition]) -> None:
        """Register a custom :class:`TransitionCondition`. Re-registers replace."""
        self._conditions[condition_cls.name] = condition_cls

    def target_from_dict(self, data: dict[str, Any] | None) -> TransitionTarget:
        if data is None:
            return TerminateTarget()
        cls = self._targets.get(data["name"])
        if cls is None:
            raise WorkflowGraphError(f"no transition target registered for {data['name']!r}")
        return cls(**data.get("args", {}))

    def condition_from_dict(self, data: dict[str, Any]) -> TransitionCondition:
        cls = self._conditions.get(data["name"])
        if cls is None:
            raise WorkflowGraphError(f"no transition condition registered for {data['name']!r}")
        return cls(**data.get("args", {}))


def register_target(target_cls: type[TransitionTarget]) -> None:
    """Register a custom :class:`TransitionTarget` on the default registry.

    Equivalent to ``TransitionRegistry.default().register_target(...)``.
    Re-registering the same name replaces the prior class.
    """
    TransitionRegistry.default().register_target(target_cls)


def register_condition(condition_cls: type[TransitionCondition]) -> None:
    """Register a custom :class:`TransitionCondition` on the default
    registry. Re-registers replace."""
    TransitionRegistry.default().register_condition(condition_cls)


# ── TransitionGraph ─────────────────────────────────────────────────────────


@dataclass(slots=True)
class TransitionGraph:
    """The orchestrator script: initial speaker + transitions +
    default fall-through + optional turn cap."""

    initial_speaker: str
    transitions: list[Transition] = field(default_factory=list)
    default_target: TransitionTarget = field(default_factory=TerminateTarget)
    max_turns: int | None = None

    # ── Serialization ──────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dict for ``ChannelMetadata.knobs``."""
        return {
            "initial_speaker": self.initial_speaker,
            "transitions": [_transition_to_dict(t) for t in self.transitions],
            "default_target": _target_to_dict(self.default_target),
            "max_turns": self.max_turns,
        }

    def dumps(self) -> str:
        """JSON string form of :meth:`to_dict`."""
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def loads(
        cls,
        data: str | dict[str, Any],
        *,
        registry: "TransitionRegistry | None" = None,
    ) -> "TransitionGraph":
        """Inverse of :meth:`to_dict` / :meth:`dumps`.

        Accepts either a JSON string or already-parsed dict. ``registry``
        defaults to the process-wide default — pass an explicit instance
        when you need isolation (e.g. multi-tenant tests) or a registry
        seeded with custom targets / conditions.
        """
        if isinstance(data, str):
            data = json.loads(data)
        reg = registry if registry is not None else TransitionRegistry.default()
        return cls(
            initial_speaker=data["initial_speaker"],
            transitions=[_transition_from_dict(t, reg) for t in data.get("transitions", [])],
            default_target=reg.target_from_dict(data.get("default_target")),
            max_turns=data.get("max_turns"),
        )

    # ── Convenience factories ──────────────────────────────────────

    @classmethod
    def round_robin(
        cls,
        participants: list[str],
        *,
        max_turns: int | None = None,
    ) -> "TransitionGraph":
        """Cycle through ``participants`` in order; terminate after ``max_turns``."""
        if not participants:
            raise WorkflowGraphError("round_robin requires at least 1 participant")
        return cls(
            initial_speaker=participants[0],
            transitions=[Transition(when=Always(), then=RoundRobinTarget())],
            default_target=TerminateTarget(reason="round_robin_complete"),
            max_turns=max_turns,
        )

    @classmethod
    def sequence(cls, steps: list[str]) -> "TransitionGraph":
        """Pipeline: ``steps[0]`` → ``steps[1]`` → ... → terminate."""
        if not steps:
            raise WorkflowGraphError("sequence requires at least 1 step")
        transitions: list[Transition] = []
        for i in range(len(steps) - 1):
            transitions.append(
                Transition(
                    when=FromSpeaker(steps[i]),
                    then=AgentTarget(steps[i + 1]),
                )
            )
        return cls(
            initial_speaker=steps[0],
            transitions=transitions,
            default_target=TerminateTarget(reason="sequence_complete"),
            max_turns=len(steps),
        )


# ── Serialization helpers (module-level — no nested fns in hot path) ────────


def _target_to_dict(target: TransitionTarget) -> dict[str, Any]:
    return {"name": target.name, "args": _dataclass_args(target)}


def _condition_to_dict(condition: TransitionCondition) -> dict[str, Any]:
    return {"name": condition.name, "args": _dataclass_args(condition)}


def _transition_to_dict(transition: Transition) -> dict[str, Any]:
    return {
        "when": _condition_to_dict(transition.when),
        "then": _target_to_dict(transition.then),
        "priority": transition.priority,
    }


def _transition_from_dict(data: dict[str, Any], registry: TransitionRegistry) -> Transition:
    return Transition(
        when=registry.condition_from_dict(data["when"]),
        then=registry.target_from_dict(data["then"]),
        priority=data.get("priority", 0),
    )


def _dataclass_args(obj: object) -> dict[str, Any]:
    """``asdict`` for the instance fields only — ``ClassVar``s are skipped
    automatically by ``dataclasses.fields``."""
    if not hasattr(obj, "__dataclass_fields__"):
        return {}
    return asdict(obj)
