# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Four-stage transform pipeline driver — Phase 5a.1.

A :class:`TransformPipeline` owns the compiled list of transforms for
every stage, dispatches envelopes through them in declaration order,
and handles the three termination outcomes per envelope: mutate
(return the new envelope), pass (return unchanged), or reject
(return ``None``). The pipeline also owns the lifecycle of adapter
instances it built — dropping a pipeline closes every adapter via
``aclose()`` where supported.

Unknown ``apply`` forms (``exec`` / ``ws`` — Phase 5b) log a warning
**once per (stage, form) pair** and are silently skipped. Rules
authored against the full five-form surface therefore do not break
on a Phase 5a hub; they just lose the effect of those specific
entries.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterable

from ...envelope import Envelope
from ...rule import Rule, TransformSpec, TransformStage
from .adapters import HttpTransform, NamedTransform, PythonTransform
from .protocol import Transform, TransformContext, TransformError
from .registry import TransformRegistry
from .when import when_matches

if TYPE_CHECKING:
    from ..actor_client import ActorClient

log = logging.getLogger(__name__)

__all__ = ("CompiledTransform", "TransformPipeline")


_OUTBOUND_STAGES = frozenset(
    {TransformStage.PRE_SEND, TransformStage.POST_SEND}
)


class CompiledTransform:
    """A built :class:`Transform` plus its declared ``when`` filter.

    Wrapping the two together lets the pipeline evaluate the filter
    without re-parsing the ``TransformSpec`` on every envelope, and
    gives the pipeline a single handle to close when the rule changes.
    """

    __slots__ = ("transform", "when", "stage", "origin")

    def __init__(
        self,
        *,
        transform: Transform,
        when: dict[str, Any],
        stage: TransformStage,
        origin: str,
    ) -> None:
        self.transform = transform
        self.when = when
        self.stage = stage
        self.origin = origin

    async def aclose(self) -> None:
        close = getattr(self.transform, "aclose", None)
        if close is not None:
            try:
                await close()
            except Exception:  # pragma: no cover
                log.warning(
                    "CompiledTransform aclose failed for %s",
                    self.origin,
                    exc_info=True,
                )


class TransformPipeline:
    """Per-:class:`ActorClient` transform dispatcher.

    One pipeline is built per ``(ActorClient, Rule)`` pair. On
    :class:`RuleChangedFrame` the owning client constructs a fresh
    pipeline and atomically swaps it in; the old pipeline is drained
    via :meth:`aclose` afterwards.
    """

    def __init__(self, *, rule_version: int) -> None:
        self._stages: dict[TransformStage, list[CompiledTransform]] = {
            TransformStage.PRE_SEND: [],
            TransformStage.POST_SEND: [],
            TransformStage.PRE_RECEIVE: [],
            TransformStage.POST_RECEIVE: [],
        }
        self._rule_version = rule_version
        self._warned_unknown_forms: set[tuple[TransformStage, str]] = set()

    @property
    def rule_version(self) -> int:
        return self._rule_version

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        rule: Rule,
        *,
        registry: TransformRegistry,
    ) -> TransformPipeline:
        """Compile a :class:`Rule`'s transform list into a pipeline.

        Unknown stages were already rejected by
        :meth:`TransformSpec.from_dict` / :meth:`Rule.from_dict` — the
        ``stage`` field is always a valid :class:`TransformStage` by
        the time we get here.
        """

        pipeline = cls(rule_version=rule.version)
        for spec in rule.transforms:
            compiled = pipeline._compile_spec(spec, registry)
            if compiled is None:
                continue
            pipeline._stages[compiled.stage].append(compiled)
        return pipeline

    def _compile_spec(
        self, spec: TransformSpec, registry: TransformRegistry
    ) -> CompiledTransform | None:
        stage = TransformStage(spec.stage)  # validated by from_dict already
        apply = spec.apply
        try:
            transform, origin = _compile_apply(apply, registry=registry)
        except _UnknownForm as unknown:
            key = (stage, unknown.form)
            if key not in self._warned_unknown_forms:
                log.warning(
                    "transform.unsupported_form stage=%s form=%s "
                    "(Phase 5a does not execute this apply form; "
                    "Phase 5b will)",
                    stage.value,
                    unknown.form,
                )
                self._warned_unknown_forms.add(key)
            return None
        except TransformError:
            raise
        return CompiledTransform(
            transform=transform,
            when=dict(spec.when),
            stage=stage,
            origin=origin,
        )

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def run_pre_send(
        self, envelope: Envelope, client: ActorClient
    ) -> Envelope | None:
        """Run ``pre_send`` transforms. ``None`` → reject."""

        return await self._run_mutating(
            TransformStage.PRE_SEND, envelope, client
        )

    async def run_post_send(
        self, envelope: Envelope, client: ActorClient
    ) -> None:
        """Run ``post_send`` side-effect transforms. Return value ignored."""

        await self._run_side_effect(
            TransformStage.POST_SEND, envelope, client
        )

    async def run_pre_receive(
        self, envelope: Envelope, client: ActorClient
    ) -> Envelope | None:
        """Run ``pre_receive`` transforms. ``None`` → reject."""

        return await self._run_mutating(
            TransformStage.PRE_RECEIVE, envelope, client
        )

    async def run_post_receive(
        self, envelope: Envelope, client: ActorClient
    ) -> None:
        """Run ``post_receive`` side-effect transforms. Return value ignored."""

        await self._run_side_effect(
            TransformStage.POST_RECEIVE, envelope, client
        )

    async def _run_mutating(
        self,
        stage: TransformStage,
        envelope: Envelope,
        client: ActorClient,
    ) -> Envelope | None:
        compiled = self._stages[stage]
        if not compiled:
            return envelope
        direction = "outbound" if stage in _OUTBOUND_STAGES else "inbound"
        current = envelope
        for entry in compiled:
            if not when_matches(entry.when, current, client):
                continue
            ctx = TransformContext(
                stage=stage,
                client=client,
                session_id=current.session_id,
                rule_version=self._rule_version,
                direction=direction,
            )
            try:
                result = await entry.transform(current, ctx)
            except Exception:
                log.exception(
                    "transform raised at %s (origin=%s); treating as reject",
                    stage.value,
                    entry.origin,
                )
                return None
            if result is None:
                return None
            current = result
        return current

    async def _run_side_effect(
        self,
        stage: TransformStage,
        envelope: Envelope,
        client: ActorClient,
    ) -> None:
        compiled = self._stages[stage]
        if not compiled:
            return
        direction = "outbound" if stage in _OUTBOUND_STAGES else "inbound"
        for entry in compiled:
            if not when_matches(entry.when, envelope, client):
                continue
            ctx = TransformContext(
                stage=stage,
                client=client,
                session_id=envelope.session_id,
                rule_version=self._rule_version,
                direction=direction,
            )
            try:
                await entry.transform(envelope, ctx)
            except Exception:
                log.exception(
                    "post-stage transform raised at %s (origin=%s)",
                    stage.value,
                    entry.origin,
                )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def iter_compiled(self) -> Iterable[CompiledTransform]:
        for stage_list in self._stages.values():
            yield from stage_list

    async def aclose(self) -> None:
        """Drain every adapter that owns external state.

        Called by the owning :class:`ActorClient` after an atomic
        pipeline swap and on disconnect. Idempotent; safe to call
        multiple times.
        """

        for entry in list(self.iter_compiled()):
            await entry.aclose()
        for stage_list in self._stages.values():
            stage_list.clear()


class _UnknownForm(Exception):
    """Internal — ``apply`` form is not handled in Phase 5a (exec / ws)."""

    def __init__(self, form: str) -> None:
        super().__init__(form)
        self.form = form


def _compile_apply(
    apply: Any, *, registry: TransformRegistry
) -> tuple[Transform, str]:
    """Compile a single ``apply`` into a :class:`Transform` instance.

    Returns ``(transform, origin_label)``. Raises
    :class:`_UnknownForm` for ``exec`` / ``ws`` so the caller logs a
    pass-through warning exactly once per ``(stage, form)``.
    """

    if isinstance(apply, str):
        transform = NamedTransform(apply, registry)
        return transform, f"named:{apply}"
    if isinstance(apply, dict):
        if "python" in apply:
            spec = apply["python"]
            if not isinstance(spec, dict):
                raise TransformError(
                    "python apply form must be a dict"
                )
            return PythonTransform(spec), f"python:{spec.get('module')}.{spec.get('class')}"
        if "http" in apply:
            url = apply["http"]
            if not isinstance(url, str):
                raise TransformError(
                    "http apply form must be a URL string"
                )
            return HttpTransform(url), f"http:{url}"
        for form in ("exec", "ws"):
            if form in apply:
                raise _UnknownForm(form)
    raise TransformError(
        f"unrecognized transform apply form: {apply!r}"
    )
