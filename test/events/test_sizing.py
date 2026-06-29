# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for content-size estimation and prompt rendering (ag2.events.sizing)."""

from ag2.events import (
    ImageInput,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolResult,
    ToolResultEvent,
    ToolResultsEvent,
    Usage,
    UsageEvent,
    estimated_tokens,
    render_for_prompt,
)

BIG = "x" * 10_000


def test_estimate_is_not_truncated_and_symmetric() -> None:
    # The same 10k payload must estimate the same regardless of event type —
    # str(event)/truncate_repr gave 137 / 10023 / 237 here.
    mr = ModelRequest([TextInput(BIG)])
    resp = ModelResponse(ModelMessage(BIG))
    tr = ToolResultsEvent(results=[ToolResultEvent(parent_id="c1", name="t", result=ToolResult(BIG))])

    assert estimated_tokens(mr) == 2500
    assert estimated_tokens(resp) == 2500
    assert estimated_tokens(tr) == 2500


def test_chars_per_token_scales_estimate() -> None:
    assert estimated_tokens(ModelRequest([TextInput(BIG)]), chars_per_token=1) == 10_000


def test_image_counts_a_flat_budget_not_zero() -> None:
    text_only = ModelRequest([TextInput("hi")])
    with_image = ModelRequest([TextInput("hi"), ImageInput(data=b"\x89PNG" + b"0" * 5000, media_type="image/png")])

    assert estimated_tokens(with_image) - estimated_tokens(text_only) == 1000


def test_render_preserves_full_text() -> None:
    rendered = render_for_prompt(ModelRequest([TextInput(BIG)]))
    assert BIG in rendered
    assert rendered.startswith("User: ")


def test_telemetry_is_not_counted_or_rendered() -> None:
    # UsageEvent never reaches the model, so it must size to 0 and render empty
    # for every budget calc (trigger, TokenLimiter) and prompt.
    usage = UsageEvent(Usage(total_tokens=1500), kind="model_call")
    assert estimated_tokens(usage) == 0
    assert render_for_prompt(usage) == ""


def test_render_uses_placeholder_for_non_text() -> None:
    rendered = render_for_prompt(
        ModelRequest([TextInput("look"), ImageInput(data=b"\x89PNG0000", media_type="image/png")])
    )
    assert rendered == "User: look [image]"
    assert "PNG" not in rendered  # raw bytes never leak into the prompt
