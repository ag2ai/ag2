# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Dataset layer: the tasks an evaluation runs over, and the target it drives."""

from .suite import Suite
from .task import Task

__all__ = (
    "Suite",
    "Task",
)
