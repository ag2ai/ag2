# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Workflow-specific tool helpers.

Helpers for tools running inside ``WorkflowAdapter`` sessions.
They hide the ``EV_CONTEXT_SET`` envelope shape behind a small
typed API so tool bodies don't need to know about envelope semantics:

```python
from autogen.beta.network import SessionInject
from autogen.beta.network.workflow_helpers import set_context


@coord_agent.tool
async def classify_billing(reason: str, session: SessionInject) -> str:
    await set_context(session, "category", "billing")
    return "ok"
```
"""

from typing import Any

from .adapters.workflow import WORKFLOW_TYPE
from .client.session import Session
from .envelope import EV_CONTEXT_SET

__all__ = ("delete_context", "set_context")


def _ensure_workflow(session: Session, helper_name: str) -> None:
    """Type-guard: helpers only meaningful on workflow sessions."""
    actual = session.metadata.manifest.type
    if actual != WORKFLOW_TYPE:
        raise RuntimeError(
            f"{helper_name}() requires a workflow session "
            f"(manifest.type == {WORKFLOW_TYPE!r}); got {actual!r}. "
            f"Other adapter types do not have context_vars."
        )


async def set_context(session: Session, key: str, value: Any) -> None:
    """Set one workflow ``context_vars`` entry.

    Emits an ``EV_CONTEXT_SET`` envelope under the hood; visible to
    every participant via ``WorkflowState.context_vars`` after fold.
    Loose semantics — any participant may call this regardless of
    turn order.
    """
    _ensure_workflow(session, "set_context")
    await session.send(
        "",
        event_type=EV_CONTEXT_SET,
        event_data={"set": {key: value}},
    )


async def delete_context(session: Session, key: str) -> None:
    """Delete one workflow ``context_vars`` entry.

    Emits an ``EV_CONTEXT_SET`` envelope with ``delete=[key]``.
    No-op if ``key`` was not set.
    """
    _ensure_workflow(session, "delete_context")
    await session.send(
        "",
        event_type=EV_CONTEXT_SET,
        event_data={"delete": [key]},
    )
