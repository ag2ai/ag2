# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""The remote fake config, apart from :mod:`ag2.acp.testing` so that module imports without ``[http]``."""

from dataclasses import dataclass

from .remote import ACPRemoteConfig
from .testing import _FakeConfigViews


@dataclass(slots=True, kw_only=True)
class FakeACPRemoteConfig(_FakeConfigViews, ACPRemoteConfig):
    """:class:`ACPRemoteConfig` bound to the scripted in-process agent."""
