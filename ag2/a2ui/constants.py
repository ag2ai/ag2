# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

A2UI_MIME_TYPE = "application/a2ui+json"

# Official A2UI "Standard Prompt Tags" (a2ui-project/a2ui, agent_sdk_guide.md):
# the LLM wraps its A2UI output between these tags for deterministic parsing,
# e.g. ``CONVERSATIONAL TEXT\n<a2ui-json>[ {…} ]</a2ui-json>``. This replaces
# AG2's earlier homegrown ``---a2ui_JSON---`` delimiter.
A2UI_JSON_OPEN_TAG = "<a2ui-json>"
A2UI_JSON_CLOSE_TAG = "</a2ui-json>"

__all__ = (
    "A2UI_JSON_CLOSE_TAG",
    "A2UI_JSON_OPEN_TAG",
    "A2UI_MIME_TYPE",
)
