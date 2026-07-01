# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Ready-made observers for eval lifecycle events.

These are plain callbacks you subscribe to a run's stream — opt-in, no ``run_agent()``
flag. The stream is the seam; this is one renderer you can attach to it::

    from ag2.stream import MemoryStream
    from ag2.eval import console_reporter, run_variants

    stream = MemoryStream()
    stream.subscribe(console_reporter)
    await run_variants(suite, variants=variants, scorers=[...], store_dir="runs", stream=stream)

Want different output (a file, a progress bar, a UI)? Write your own callback and
subscribe it the same way — ``console_reporter`` is just the convenient default.
"""

from ag2.events import BaseEvent

from .events import (
    EvalStarted,
    PairwiseCompared,
    PairwiseCompleted,
    PairwiseStarted,
    TaskEvaluated,
    VariantCompleted,
    VariantStarted,
)

__all__ = ("console_reporter",)


def console_reporter(event: BaseEvent) -> None:
    """Print eval lifecycle progress to stdout. Subscribe to a run's stream to use it."""
    if isinstance(event, EvalStarted):
        tag = f"[{event.label}] " if event.label else ""
        print(f"{tag}running {event.total} task-run(s) over {event.suite!r}...")
    elif isinstance(event, VariantStarted):
        print(f"  → variant {event.variant} ({event.index}/{event.total})")
    elif isinstance(event, TaskEvaluated):
        passed = sum(1 for fb in event.feedback if fb.score is True)
        where = f"[{event.variant}] " if event.variant else ""
        print(f"    {where}{event.task_id}: {passed}/{len(event.feedback)} passed")
    elif isinstance(event, VariantCompleted):
        print(f"  ✓ variant {event.variant} done")
    elif isinstance(event, PairwiseStarted):
        tag = f"[{event.label}] " if event.label else ""
        print(f"{tag}comparing {event.variant_b!r} (B) vs {event.variant_a!r} (A) over {event.total} pair(s)...")
    elif isinstance(event, PairwiseCompared):
        print(f"    {event.task_id} [{event.key}]: {event.winner}")
    elif isinstance(event, PairwiseCompleted):
        print(f"  ✓ pairwise {event.result.run_id} done")
