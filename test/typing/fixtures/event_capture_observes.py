# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""An ``EventCapture`` is an observer an agent accepts under the checker.

Checked by ``test/typing/test_fixtures.py``; not imported by the suite. The fixture
carries no expectations: a clean run is the assertion.
"""

from ag2 import Agent
from ag2.eval.runtime._capture import EventCapture
from ag2.observers import Observer
from ag2.testing import TestConfig

observer: Observer = EventCapture()

agent = Agent("capturing", config=TestConfig(), observers=[EventCapture()])
