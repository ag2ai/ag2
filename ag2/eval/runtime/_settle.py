# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Shared settling policy for the per-item fan-out loops in the runtime layer.

``evaluate_traces`` and ``run_agent`` both fan work out with
``asyncio.gather(..., return_exceptions=True)`` and then walk the results,
deciding per item whether an outcome is a value, a recordable failure, or a
signal that must propagate. Both apply the *same* policy; this is its one home
so the two call sites cannot drift.
"""


def reraise_if_not_exception(outcome: BaseException) -> None:
    """Re-raise ``outcome`` unless it is an ordinary :class:`Exception`.

    ``asyncio.gather(return_exceptions=True)`` captures *every* ``BaseException``
    a child raises, including ``CancelledError`` and ``KeyboardInterrupt``. Those
    are control-flow signals, never per-item results — turning them into an error
    entry would silently swallow a cancellation or interrupt. So the caller passes
    each ``BaseException`` outcome through here: non-``Exception`` signals re-raise
    and abort the run; ordinary exceptions return normally so the caller can record
    them as a failed item.
    """
    if not isinstance(outcome, Exception):
        raise outcome
