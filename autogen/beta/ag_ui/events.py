# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ag_ui.core import Event

from autogen.beta.events import BaseEvent, Field


class AGUIEvent(BaseEvent):
    event: Event = Field(kw_only=False)
