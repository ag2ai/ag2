# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``HubArbiter`` — decision-making seam for access / routing.

A swappable Protocol the hub consults inline before committing
register / channel-open / send / dispatch decisions. Distinct from
:class:`HubListener` (read-only observation, after the fact); the
arbiter is the gatekeeper.

Default impl :class:`RuleBasedArbiter` enforces the per-agent
:class:`Rule` (``access`` + ``limits``) — same behavior the hub had
inline before this seam existed. Tenants replace it with a custom
arbiter to layer JWT scopes, federation routing, etc., on top of (or
in place of) the rule data.

Return type :class:`Decision` is either :class:`Allow` or
:class:`Deny(reason, error=...)`. ``Deny.error`` controls which
:class:`NetworkError` subclass the hub raises back to the caller —
defaults to :class:`AccessDeniedError`. Returning ``None`` from
:meth:`resolve_unknown_audience` means "drop the unknown ids
silently" (preserves the current single-hub behavior); returning a
list means "deliver to these ids instead" (federation hook).
"""

import fnmatch
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..errors import AccessDeniedError, InboxFull, NetworkError

if TYPE_CHECKING:
    from ..channel import ChannelManifest
    from ..envelope import Envelope
    from ..identity import Passport, Resume
    from ..rule import Rule

__all__ = (
    "Allow",
    "BaseHubArbiter",
    "Decision",
    "Deny",
    "HubArbiter",
    "RuleBasedArbiter",
)


@dataclass(slots=True, frozen=True)
class Allow:
    """Decision: the action is permitted."""


@dataclass(slots=True, frozen=True)
class Deny:
    """Decision: the action is not permitted.

    ``error`` selects which :class:`NetworkError` subclass the hub
    raises (defaults to :class:`AccessDeniedError`). ``reason`` is
    the human-readable message.
    """

    reason: str
    error: type[NetworkError] = AccessDeniedError


Decision = Allow | Deny


@runtime_checkable
class HubArbiter(Protocol):
    """Decision-making Protocol the hub consults before committing.

    Implementations replace the default :class:`RuleBasedArbiter` via
    :meth:`Hub.register_arbiter`. Only one arbiter is active at a
    time; the most recent registration wins.

    Methods may inspect any state passed in but should not mutate hub
    state — they are decisions, not side-effecting work.
    """

    async def authorize_send(
        self,
        envelope: "Envelope",
        sender: "Passport",
        sender_rule: "Rule",
        recipients: list["Passport"],
    ) -> Decision:
        """Gate ``post_envelope`` before the WAL append.

        Checks rule-based outbound access (``access.outbound_to``)
        plus delegation depth (``limits.delegation_depth``). Inbox
        capacity is handled separately by :meth:`authorize_inbox`
        because it depends on per-recipient hub-internal counters.
        """

    async def authorize_inbox(
        self,
        envelope: "Envelope",
        recipient: "Passport",
        recipient_rule: "Rule",
        current_pending: int,
    ) -> Decision:
        """Gate ``post_envelope`` per-recipient against inbox capacity.

        Hub iterates the resolved audience, calling this once per
        recipient with their current pending count. Returning
        ``Deny`` (with ``error=InboxFull`` by default) short-circuits
        the send.
        """

    async def authorize_dispatch(
        self,
        envelope: "Envelope",
        sender: "Passport",
        recipient: "Passport",
        recipient_rule: "Rule",
    ) -> Decision:
        """Gate per-recipient dispatch (inbound access).

        Called for every notify frame the hub is about to send.
        Returning ``Deny`` causes the hub to silently skip this
        recipient (the rest of the audience still gets delivery).
        """

    async def authorize_channel_open(
        self,
        manifest: "ChannelManifest",
        creator: "Passport",
        creator_rule: "Rule",
        invitees: list["Passport"],
        invitee_rules: list["Rule"],
        active_creator_channels: int,
    ) -> Decision:
        """Gate ``create_channel`` before any persistence.

        Default impl checks each invitee's ``access.inbound_from``
        against the creator's name, plus the creator's
        ``limits.max_concurrent_channels`` against the running count.
        """

    async def authorize_register(
        self,
        passport: "Passport",
        resume: "Resume",
        rule: "Rule",
    ) -> Decision:
        """Gate ``Hub.register``. Default impl always allows."""

    async def resolve_unknown_audience(
        self,
        envelope: "Envelope",
        unknown_ids: list[str],
    ) -> list[str] | None:
        """Federation hook for audience members the hub doesn't know.

        Default impl returns ``None`` (drop unknown ids, current
        single-hub behavior). A federated arbiter returns a
        (possibly empty) list of agent_ids the envelope should be
        re-delivered to instead — typically the local proxy id of a
        remote peer.
        """


def _match_any(name: str, patterns: list[str]) -> bool:
    """True if ``name`` matches any of the glob patterns (``["*"]`` allows all)."""
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


class BaseHubArbiter:
    """No-op base implementation. Override only the gates you care about.

    Mirrors :class:`BaseHubListener`: every method returns
    :class:`Allow` so subclasses opting in to a single gate (e.g.
    ``authorize_register`` for tenant-quota checks) don't have to
    implement the rest of the surface.

    For the rule-based behavior the hub had inline before this seam
    existed, use :class:`RuleBasedArbiter` (the default) and compose /
    chain via subclassing.
    """

    async def authorize_send(
        self,
        envelope: "Envelope",  # noqa: ARG002
        sender: "Passport",  # noqa: ARG002
        sender_rule: "Rule",  # noqa: ARG002
        recipients: list["Passport"],  # noqa: ARG002
    ) -> Decision:
        return Allow()

    async def authorize_inbox(
        self,
        envelope: "Envelope",  # noqa: ARG002
        recipient: "Passport",  # noqa: ARG002
        recipient_rule: "Rule",  # noqa: ARG002
        current_pending: int,  # noqa: ARG002
    ) -> Decision:
        return Allow()

    async def authorize_dispatch(
        self,
        envelope: "Envelope",  # noqa: ARG002
        sender: "Passport",  # noqa: ARG002
        recipient: "Passport",  # noqa: ARG002
        recipient_rule: "Rule",  # noqa: ARG002
    ) -> Decision:
        return Allow()

    async def authorize_channel_open(
        self,
        manifest: "ChannelManifest",  # noqa: ARG002
        creator: "Passport",  # noqa: ARG002
        creator_rule: "Rule",  # noqa: ARG002
        invitees: list["Passport"],  # noqa: ARG002
        invitee_rules: list["Rule"],  # noqa: ARG002
        active_creator_channels: int,  # noqa: ARG002
    ) -> Decision:
        return Allow()

    async def authorize_register(
        self,
        passport: "Passport",  # noqa: ARG002
        resume: "Resume",  # noqa: ARG002
        rule: "Rule",  # noqa: ARG002
    ) -> Decision:
        return Allow()

    async def resolve_unknown_audience(
        self,
        envelope: "Envelope",  # noqa: ARG002
        unknown_ids: list[str],  # noqa: ARG002
    ) -> list[str] | None:
        return None


class RuleBasedArbiter:
    """Default arbiter — enforces per-agent :class:`Rule` exactly as the
    hub did before this seam existed.

    Tenants extend or replace via :meth:`Hub.register_arbiter`. Pure
    function of the inputs; holds no state.
    """

    async def authorize_send(
        self,
        envelope: "Envelope",
        sender: "Passport",
        sender_rule: "Rule",
        recipients: list["Passport"],
    ) -> Decision:
        # Self-routing is always allowed — protocol broadcasts
        # (``EV_CHANNEL_OPENED`` / ``EV_CHANNEL_CLOSED``) include the
        # creator in their own audience.
        for recipient in recipients:
            if recipient.agent_id == sender.agent_id:
                continue
            if not _match_any(recipient.name, sender_rule.access.outbound_to):
                return Deny(
                    reason=f"sender {sender.name!r} not permitted to send to {recipient.name!r}",
                )
        # Delegation-depth check. ``0`` disables.
        depth_cap = sender_rule.limits.delegation_depth
        if depth_cap > 0 and envelope.depth > depth_cap:
            return Deny(
                reason=f"sender {sender.name!r} exceeded delegation_depth ({envelope.depth} > {depth_cap})",
            )
        return Allow()

    async def authorize_inbox(
        self,
        envelope: "Envelope",
        recipient: "Passport",
        recipient_rule: "Rule",
        current_pending: int,
    ) -> Decision:
        max_pending = recipient_rule.limits.inbox.max_pending
        if max_pending > 0 and current_pending >= max_pending:
            assert recipient.agent_id is not None
            return Deny(
                reason=f"recipient {recipient.agent_id!r} inbox at capacity ({current_pending} >= {max_pending})",
                error=InboxFull,
            )
        return Allow()

    async def authorize_dispatch(
        self,
        envelope: "Envelope",
        sender: "Passport",
        recipient: "Passport",
        recipient_rule: "Rule",
    ) -> Decision:
        if not _match_any(sender.name, recipient_rule.access.inbound_from):
            return Deny(reason=f"recipient {recipient.name!r} blocks inbound from {sender.name!r}")
        return Allow()

    async def authorize_channel_open(
        self,
        manifest: "ChannelManifest",
        creator: "Passport",
        creator_rule: "Rule",
        invitees: list["Passport"],
        invitee_rules: list["Rule"],
        active_creator_channels: int,
    ) -> Decision:
        # Pre-flight inbound check on each invitee. The dispatch path
        # silently filters envelopes whose sender is not in the
        # recipient's whitelist; without this pre-check, an invite to
        # a recipient who blocks the creator would be dropped and the
        # creator would hang on the ack-timeout.
        for invitee, invitee_rule in zip(invitees, invitee_rules, strict=True):
            if invitee.agent_id == creator.agent_id:
                continue
            if not _match_any(creator.name, invitee_rule.access.inbound_from):
                return Deny(reason=f"invitee {invitee.name!r} does not accept inbound from {creator.name!r}")
        # Concurrency cap on the creator.
        max_channels = creator_rule.limits.max_concurrent_channels
        if max_channels > 0 and active_creator_channels >= max_channels:
            assert creator.agent_id is not None
            return Deny(
                reason=f"creator {creator.agent_id!r} exceeded max_concurrent_channels "
                f"({active_creator_channels} >= {max_channels})",
            )
        return Allow()

    async def authorize_register(
        self,
        passport: "Passport",
        resume: "Resume",
        rule: "Rule",
    ) -> Decision:
        return Allow()

    async def resolve_unknown_audience(
        self,
        envelope: "Envelope",
        unknown_ids: list[str],
    ) -> list[str] | None:
        # Single-hub behavior: drop unknown ids silently.
        return None
