# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Network-side DI annotations for tools running inside notify handlers.

The default notify handler (``handlers.py``) stamps the active
``Channel`` / ``AgentClient`` / ``Hub`` into ``context.dependencies``
under the qualified keys defined in ``policies.py``. These ``Annotated``
aliases give tool authors a typed handle to those bindings.

``TaskInject`` lives in framework-core (``ag2.annotations``)
because ``Task`` itself is framework-core; it is re-exported here for
symmetry with the network-only injects so users can ``from
ag2.network.client.inject import TaskInject`` alongside the
others.
"""

from typing import Annotated, Any

from ag2.annotations import Inject

from ..policies import AGENT_CLIENT_DEP, CHANNEL_DEP, CHANNEL_STATE_DEP, HUB_DEP

__all__ = (
    "AgentClientInject",
    "ChannelInject",
    "ChannelStateInject",
    "HubInject",
    "TaskInject",
)


# Network-only injects — resolve to ``None`` outside a network notify
# handler. Default ``None`` keeps the parameter optional so the same
# tool works on- and off-network.
#
# Static type is ``Any`` rather than the concrete class so Pydantic
# (which the ``tool`` decorator uses to schema function signatures) does
# not try to generate a JSON schema for ``Channel`` / ``AgentClient`` /
# ``Hub`` / ``WorkflowState`` etc. — these types are not Pydantic-friendly
# and never appear in the LLM-facing parameter surface anyway (they're
# injected from ``context.dependencies``). Type-checkers lose precision;
# document the resolved type in the docstring of any tool that uses these.
ChannelInject = Annotated[Any, Inject(CHANNEL_DEP, default=None)]
ChannelStateInject = Annotated[Any, Inject(CHANNEL_STATE_DEP, default=None)]
AgentClientInject = Annotated[Any, Inject(AGENT_CLIENT_DEP, default=None)]
HubInject = Annotated[Any, Inject(HUB_DEP, default=None)]

# Framework-core re-export for symmetry. Same definition as
# ``ag2.task.TaskInject`` — both resolve to ``ag2.task``.
TaskInject = Annotated[Any, Inject("ag2.task", default=None)]
