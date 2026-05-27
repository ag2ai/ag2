# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Based on OpenTelemetry GenAI semantic conventions
# https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/

from .instrumentators import (
    instrument_agent,
    instrument_llm_wrapper,
    instrument_pattern,
)

__all__ = [
    "instrument_agent",
    "instrument_llm_wrapper",
    "instrument_pattern",
]

try:
    from .instrumentators import instrument_a2a_server  # noqa: F401

    __all__.append("instrument_a2a_server")
except ImportError:
    pass
