# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""AuctionAdapter — request-for-proposal session.

Delivery rules:

* **Phase 1 — RFP.** The initiator posts a text envelope describing the
  task (first send, required).
* **Phase 2 — Bids.** Any non-initiator participant may post a single
  text envelope as their bid. The adapter enforces one-bid-per-bidder
  and rejects bids sent before the RFP.
* **Phase 3 — Select.** The initiator posts an ``ag2.auction.select``
  envelope naming the winner (``event_data = {"winner_id": ...}``).
  Select closes the bid window; everyone except the winner stops being
  an allowed sender.
* **Phase 4 — Continuation.** The winner may post replies back to the
  initiator. The session stays open for bidirectional
  initiator↔winner communication until an explicit close.

The adapter is stateless — phase is derived from the WAL on every send.
"""

from __future__ import annotations

from ..envelope import EV_TEXT, Envelope
from ..errors import SessionTypeError
from ..session_types import (
    ParticipantRole,
    SessionMetadata,
    SessionState,
    SessionType,
)
from .base import AdapterResult


EV_AUCTION_SELECT = "ag2.auction.select"


class AuctionAdapter:
    session_type = SessionType.AUCTION

    # ------------------------------------------------------------------
    # Create-time validation
    # ------------------------------------------------------------------

    def validate_create(self, metadata: SessionMetadata) -> None:
        if len(metadata.participants) < 2:
            raise SessionTypeError(
                "auction requires an initiator and at least one bidder"
            )
        roles = [p.role for p in metadata.participants]
        if roles.count(ParticipantRole.INITIATOR) != 1:
            raise SessionTypeError("auction requires exactly one initiator")

    def _initiator(self, metadata: SessionMetadata) -> str:
        for p in metadata.participants:
            if p.role is ParticipantRole.INITIATOR:
                return p.actor_id
        raise SessionTypeError("auction session has no initiator")

    def _winner_from_prior(
        self, prior_envelopes: list[Envelope]
    ) -> str | None:
        for env in prior_envelopes:
            if env.event_type == EV_AUCTION_SELECT:
                winner = env.event_data.get("winner_id")
                if isinstance(winner, str):
                    return winner
        return None

    # ------------------------------------------------------------------
    # Per-envelope validation
    # ------------------------------------------------------------------

    def validate_send(
        self,
        metadata: SessionMetadata,
        envelope: Envelope,
        prior_envelopes: list[Envelope],
    ) -> None:
        initiator = self._initiator(metadata)
        winner = self._winner_from_prior(prior_envelopes)

        if envelope.event_type == EV_AUCTION_SELECT:
            if envelope.sender_id != initiator:
                raise SessionTypeError(
                    "auction: only the initiator may send select"
                )
            if winner is not None:
                raise SessionTypeError(
                    "auction: select was already posted for this session"
                )
            winner_id = envelope.event_data.get("winner_id")
            if not isinstance(winner_id, str):
                raise SessionTypeError(
                    "auction: select envelope must set event_data.winner_id"
                )
            if not metadata.has_participant(winner_id):
                raise SessionTypeError(
                    "auction: select winner_id must be a session participant"
                )
            if winner_id == initiator:
                raise SessionTypeError(
                    "auction: select winner cannot be the initiator"
                )
            # The first non-RFP envelope must be a bid, not a select.
            user_prior = [
                e
                for e in prior_envelopes
                if e.event_type in (EV_TEXT, EV_AUCTION_SELECT)
            ]
            if not user_prior:
                raise SessionTypeError(
                    "auction: cannot select before the RFP is posted"
                )
            return

        if envelope.event_type != EV_TEXT:
            raise SessionTypeError(
                f"auction carries {EV_TEXT!r} and {EV_AUCTION_SELECT!r} events, "
                f"got {envelope.event_type!r}"
            )

        if not metadata.has_participant(envelope.sender_id):
            raise SessionTypeError("auction: sender is not a participant")

        # Phase 1: RFP. First text envelope must come from initiator.
        user_text_prior = [e for e in prior_envelopes if e.event_type == EV_TEXT]
        if not user_text_prior:
            if envelope.sender_id != initiator:
                raise SessionTypeError(
                    "auction: first send must be the RFP from the initiator"
                )
            return

        # Phase 2/3: either bids (pre-select) or initiator↔winner
        # (post-select).
        if winner is None:
            # Bidding phase. Anyone except the initiator may bid — and
            # exactly once. The initiator may not re-post after the
            # RFP until they send a select.
            if envelope.sender_id == initiator:
                raise SessionTypeError(
                    "auction: initiator cannot speak during bidding — post a select instead"
                )
            prior_bid_senders = {
                e.sender_id for e in user_text_prior[1:]
            }  # skip the RFP
            if envelope.sender_id in prior_bid_senders:
                raise SessionTypeError(
                    "auction: each bidder may post one bid"
                )
            return

        # Phase 4: continuation between initiator and winner.
        if envelope.sender_id not in (initiator, winner):
            raise SessionTypeError(
                "auction: only the initiator and the selected winner may speak after select"
            )

    def on_accepted(
        self,
        metadata: SessionMetadata,
        envelope: Envelope,
        prior_envelopes: list[Envelope],
    ) -> AdapterResult:
        if metadata.state is SessionState.PENDING:
            return AdapterResult(next_state=SessionState.ACTIVE)
        return AdapterResult()
