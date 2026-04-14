# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Built-in named transforms — Phase 5a.1 standard library.

Three entries, installed by default into every :class:`ActorClient`'s
:class:`TransformRegistry`:

- ``redact_pii`` — strip email addresses, phone numbers, and US SSN
  patterns from the text content of ``ag2.msg.text`` envelopes.
- ``truncate_long_content`` — cap text content at 32KB (default) or
  a configurable length.
- ``stamp_audit_header`` — copy ``sender_id`` / ``trace_id`` / stage
  into ``envelope.event_data["_audit"]``.

All three run fully in the actor's address space. They are the
smallest useful proof of the named-transform surface; operators who
need richer behavior ship their own via :class:`PythonTransform` or
by registering a factory through :meth:`ActorClient.register_transform`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from ...envelope import EV_TEXT, Envelope
from .protocol import Transform, TransformContext
from .registry import TransformRegistry

__all__ = (
    "DEFAULT_TRUNCATE_BYTES",
    "PiiRedactor",
    "Truncator",
    "AuditStamper",
    "install_stdlib_transforms",
)


DEFAULT_TRUNCATE_BYTES: Final[int] = 32 * 1024


# ---------------------------------------------------------------------------
# redact_pii
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
)
_PHONE_RE = re.compile(
    r"\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b",
)
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


@dataclass(slots=True)
class PiiRedactor:
    """Regex-based PII stripper.

    Deliberately minimal — three common patterns, each replaced with
    a ``[REDACTED:<kind>]`` marker. Non-text envelopes pass through
    unchanged so this transform is safe to install globally.
    """

    async def __call__(
        self, envelope: Envelope, ctx: TransformContext
    ) -> Envelope | None:
        if envelope.event_type != EV_TEXT:
            return envelope
        content = envelope.event_data.get("content")
        if not isinstance(content, str):
            return envelope
        redacted = _EMAIL_RE.sub("[REDACTED:email]", content)
        redacted = _PHONE_RE.sub("[REDACTED:phone]", redacted)
        redacted = _SSN_RE.sub("[REDACTED:ssn]", redacted)
        if redacted == content:
            return envelope
        new_data = dict(envelope.event_data)
        new_data["content"] = redacted
        envelope.event_data = new_data
        return envelope


# ---------------------------------------------------------------------------
# truncate_long_content
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Truncator:
    """Cap text content at ``max_bytes`` (default 32KB).

    Measures length in the string's character count for simplicity —
    good enough for the MVP. Appends an ``…[truncated]`` marker so
    downstream readers know content was cut. Non-text envelopes pass
    through.
    """

    max_bytes: int = DEFAULT_TRUNCATE_BYTES

    async def __call__(
        self, envelope: Envelope, ctx: TransformContext
    ) -> Envelope | None:
        if envelope.event_type != EV_TEXT:
            return envelope
        content = envelope.event_data.get("content")
        if not isinstance(content, str):
            return envelope
        if len(content) <= self.max_bytes:
            return envelope
        new_data = dict(envelope.event_data)
        new_data["content"] = content[: self.max_bytes] + "…[truncated]"
        envelope.event_data = new_data
        return envelope


# ---------------------------------------------------------------------------
# stamp_audit_header
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AuditStamper:
    """Annotate envelopes with a ``_audit`` sub-dict.

    Pure annotation — never rejects. Idempotent: re-running the stamper
    overwrites the prior audit block, so a transform chain that includes
    this in both ``pre_receive`` and ``post_receive`` is harmless.
    """

    async def __call__(
        self, envelope: Envelope, ctx: TransformContext
    ) -> Envelope | None:
        new_data = dict(envelope.event_data)
        new_data["_audit"] = {
            "stage": ctx.stage.value,
            "sender_id": envelope.sender_id,
            "trace_id": envelope.trace_id,
            "actor_id": ctx.client.actor_id,
            "rule_version": ctx.rule_version,
        }
        envelope.event_data = new_data
        return envelope


# ---------------------------------------------------------------------------
# Registry installer
# ---------------------------------------------------------------------------


def install_stdlib_transforms(registry: TransformRegistry) -> None:
    """Install the three built-in named factories on a registry.

    Called from :meth:`HubClient.register` by default; opt out via
    ``install_stdlib_transforms=False`` on that method.
    """

    registry.register("redact_pii", lambda: PiiRedactor())
    registry.register("truncate_long_content", lambda: Truncator())
    registry.register("stamp_audit_header", lambda: AuditStamper())
