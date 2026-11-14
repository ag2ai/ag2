# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from a2a.types import TaskState

# The spec's terminal states (§4.1.3): the task accepts nothing further.
TERMINAL_STATES = frozenset({
    TaskState.TASK_STATE_COMPLETED,
    TaskState.TASK_STATE_CANCELED,
    TaskState.TASK_STATE_FAILED,
    TaskState.TASK_STATE_REJECTED,
})

# The interrupted states (§4.1.3): the task hands control back to the caller.
INTERRUPTED_STATES = frozenset({
    TaskState.TASK_STATE_INPUT_REQUIRED,
    TaskState.TASK_STATE_AUTH_REQUIRED,
})

# States after which nothing more arrives for the current turn.
SETTLED_STATES = TERMINAL_STATES | INTERRUPTED_STATES
