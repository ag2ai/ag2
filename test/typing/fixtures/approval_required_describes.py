# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""The hook ``approval_required`` returns can describe itself under the checker.

Checked by ``test/typing/test_fixtures.py``; not imported by the suite. The fixture
carries no expectations: a clean run is the assertion.
"""

from ag2.middleware import MiddlewareDescription, ToolMiddleware, approval_required

# The hook is an `ApprovalRequired`, so its `describe()` is reachable...
description: MiddlewareDescription = approval_required(timeout=5).describe()

# ...and it is still a hook wherever one is accepted.
hook: ToolMiddleware = approval_required()
