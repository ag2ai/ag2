# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Channel data layer — manifests, metadata, expectations.

The channel description splits in two:

* :class:`ChannelManifest` — *data*. Persisted with metadata. Describes
  what the channel is.
* :class:`ChannelAdapter` (in ``adapters/base.py``) — *code*. Registered
  in the hub process; looked up by ``(manifest.type, manifest.version)``.

Manifests are snapshotted into ``ChannelMetadata.manifest`` at create
time. Re-registering an adapter at a new version does **not** mutate
in-flight channels.

Adapters control which event types they accept via ``validate_send``;
there is no hub-level allow-list.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

__all__ = (
    "ChannelManifest",
    "ChannelMetadata",
    "ChannelState",
    "Expectation",
    "Participant",
    "ParticipantRole",
    "ParticipantSchema",
)


class ChannelState(str, Enum):
    """Lifecycle state of a channel.

    The state machine is hub-enforced. Transitions are paired with the
    envelope that drove them under the per-channel lock.
    """

    PENDING = "pending"  # invite sent, waiting on acks
    ACTIVE = "active"
    CLOSING = "closing"
    CLOSED = "closed"
    EXPIRED = "expired"


_TERMINAL_CHANNEL_STATES: frozenset[ChannelState] = frozenset({
    ChannelState.CLOSED,
    ChannelState.EXPIRED,
})


def is_terminal_channel_state(state: ChannelState) -> bool:
    """True when the channel can no longer accept envelopes."""
    return state in _TERMINAL_CHANNEL_STATES


class ParticipantRole(str, Enum):
    """Role of a participant within a channel.

    Consulting uses ``INITIATOR`` + ``RESPONDENT`` (2-party). Discussion
    and conversation use ``INITIATOR`` + ``PARTICIPANT`` (multi-party).
    """

    INITIATOR = "initiator"
    RESPONDENT = "respondent"  # 2-party non-initiator
    PARTICIPANT = "participant"  # multi-party non-initiator


@dataclass(slots=True)
class ParticipantSchema:
    """Adapter-declared bounds on participant count + role names."""

    min: int  # inclusive
    max: int | None = None  # inclusive; None = unbounded
    roles: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Expectation:
    """A protocol-shape contract the hub evaluates over WAL + clock.

    ``name`` selects a built-in evaluator (``acks_within``,
    ``reply_within``, ``max_silence``). Violations are dispatched to one
    of the built-in handlers (``audit``, ``notify_channel``,
    ``auto_close``).
    """

    name: str
    on_violation: str  # "audit" | "warn" | "notify_channel" | "hide" | "remove" | "auto_close"
    params: dict[str, Any] = field(default_factory=dict)
    applies_to: list[str] | None = None  # role names or agent ids; None = all participants


@dataclass(slots=True)
class ChannelManifest:
    """Adapter dispatch key + protocol-shape declaration.

    The manifest is data only — adapters live in code (see
    ``adapters/base.py``). The hub looks up the adapter by
    ``(type, version)`` at channel create time and snapshots the
    manifest into ``ChannelMetadata.manifest``.
    """

    type: str  # adapter dispatch key, e.g. "consulting"
    version: int = 1
    participants: ParticipantSchema = field(default_factory=lambda: ParticipantSchema(min=2))
    knobs_schema: dict[str, str] = field(default_factory=dict)  # name → type hint
    default_view_policy: str = "full_transcript"
    expectations: list[Expectation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChannelManifest":
        payload = dict(data)
        if isinstance(payload.get("participants"), dict):
            payload["participants"] = ParticipantSchema(**payload["participants"])
        if "expectations" in payload:
            payload["expectations"] = [Expectation(**e) if isinstance(e, dict) else e for e in payload["expectations"]]
        return cls(**payload)


@dataclass(slots=True)
class Participant:
    agent_id: str
    role: ParticipantRole
    order: int  # 0 for initiator; insertion order otherwise
    joined_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "role": self.role.value,
            "order": self.order,
            "joined_at": self.joined_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Participant":
        payload = dict(data)
        role = payload.get("role")
        if isinstance(role, str):
            payload["role"] = ParticipantRole(role)
        return cls(**payload)


@dataclass(slots=True)
class ChannelMetadata:
    """Mutable lifecycle record for a channel.

    ``manifest`` is the snapshot taken at create time — re-registering
    an adapter at a new version does not retroactively change this.
    ``knobs`` are adapter-specific (e.g. ``{"ordering": "round_robin"}``);
    ``labels`` is the catch-all for tenant annotations and includes
    ``"intent"`` if the creator passed one.

    ``pending_acks`` and ``rejected_by`` are populated at channel
    creation and frozen once the channel transitions to ``ACTIVE``
    (quorum reached) or fails creation. The handshake is all-or-nothing
    for both 2-party (consulting) and multi-party (discussion,
    conversation, workflow) channels: any reject fails creation.
    """

    channel_id: str
    manifest: ChannelManifest
    creator_id: str
    participants: list[Participant]
    state: ChannelState
    created_at: str
    expires_at: str | None = None
    closed_at: str | None = None
    close_reason: str = ""
    parent_channel_id: str | None = None  # nested channels
    knobs: dict[str, Any] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)

    # Multi-party handshake state (only meaningful while state == PENDING)
    required_acks: int | None = None
    pending_acks: list[str] = field(default_factory=list)
    rejected_by: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel_id": self.channel_id,
            "manifest": self.manifest.to_dict(),
            "creator_id": self.creator_id,
            "participants": [p.to_dict() for p in self.participants],
            "state": self.state.value,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "closed_at": self.closed_at,
            "close_reason": self.close_reason,
            "parent_channel_id": self.parent_channel_id,
            "knobs": dict(self.knobs),
            "labels": dict(self.labels),
            "required_acks": self.required_acks,
            "pending_acks": list(self.pending_acks),
            "rejected_by": list(self.rejected_by),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChannelMetadata":
        payload = dict(data)
        manifest = payload.get("manifest")
        if isinstance(manifest, dict):
            payload["manifest"] = ChannelManifest.from_dict(manifest)
        if "participants" in payload:
            payload["participants"] = [
                Participant.from_dict(p) if isinstance(p, dict) else p for p in payload["participants"]
            ]
        state = payload.get("state")
        if isinstance(state, str):
            payload["state"] = ChannelState(state)
        return cls(**payload)

    def participant_ids(self) -> list[str]:
        return [p.agent_id for p in self.participants]

    def is_terminal(self) -> bool:
        return is_terminal_channel_state(self.state)
