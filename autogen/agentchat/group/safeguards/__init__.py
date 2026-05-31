# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Safeguards module for agent safety and compliance.

This module provides functionality for applying, managing, and enforcing
safeguards on agent interactions including inter-agent communication,
tool interactions, LLM interactions, and user interactions.
"""

from .api import apply_safeguard_policy, reset_safeguard_policy
from .enforcer import SafeguardEnforcer
from .events import SafeguardEvent

__all__ = [
    "SafeguardEnforcer",
    "SafeguardEvent",
    "apply_safeguard_policy",
    "reset_safeguard_policy",
]
