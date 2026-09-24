# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A remote-agent config can be built under the checker.

Checked by ``test/typing/test_fixtures.py``; not imported by the suite. The fixture
carries no expectations: a clean run is the assertion.
"""

from ag2.a2a import A2AConfig
from ag2.extensions.nlip import NlipConfig

# The remote agent picks its own model, so these configs name none; they still have
# to be constructible.
a2a_model: None = A2AConfig(card_url="http://localhost:8000").model
nlip_model: None = NlipConfig(url="http://localhost:8000").model
