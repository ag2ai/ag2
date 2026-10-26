# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Everything that depends on which `anthropic` SDK major is installed lives here."""

import warnings
from typing import Any

import anthropic

SDK_2X = int(anthropic.__version__.split(".")[0]) >= 2
"""`anthropic` 2.x sends Messages API version 2026-10-21, which answers 400 to the sampling parameters."""

SAMPLING_FIELDS = ("temperature", "top_p", "top_k")

SAMPLING_DEPRECATION = (
    "The temperature, top_p and top_k parameters are deprecated: Anthropic API version 2026-10-21, "
    "sent by anthropic 2.x, rejects them, so they are ignored there. They will be removed in 1.4.0."
)


def take_sampling_fields(values: dict[str, Any]) -> dict[str, Any]:
    """Pop every sampling key from ``values``, an explicit ``None`` included, returning the ones that were set."""
    return {name: value for name in SAMPLING_FIELDS if (value := values.pop(name, None)) is not None}


def sampling_extra_body(sampling: dict[str, Any], extra_body: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the request's extra body with the deprecated sampling values applied for the installed SDK.

    With 2.x they are dropped. With 1.x they left the method signatures but the API still reads
    them from the body, so they go there; an explicit ``extra_body`` wins on collision.
    """
    if not sampling:
        return extra_body

    warnings.warn(SAMPLING_DEPRECATION, DeprecationWarning, stacklevel=3)

    if SDK_2X:
        return extra_body
    return {**sampling, **(extra_body or {})}
