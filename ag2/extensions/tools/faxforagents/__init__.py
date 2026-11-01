# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Fax for Agents extension for AG2.

Provides a toolkit that lets agents send and receive faxes through the
Fax for Agents REST API (https://faxforagents.com/docs/api/). The send flow
is three calls: prepare a document, quote it to one number, queue the fax.
A live send needs a human approval bound to the exact quote, or a policy the
account owner wrote outside any prompt. Everything that comes back from a
received fax is data from whoever sent it.

The toolkit talks to the API with ``httpx``, an AG2 core dependency, so it
needs no additional package.

Maintainer: bensynapse
Docs: https://docs.ag2.ai/docs/user-guide/extensions/tools/faxforagents/
"""

from ag2.extensions.tools.faxforagents.toolkit import (
    FaxForAgentsError,
    FaxForAgentsToolkit,
    FaxQuote,
    FaxStatus,
    InboundFax,
    InboxItem,
    InboxPage,
)

__all__ = (
    "FaxForAgentsError",
    "FaxForAgentsToolkit",
    "FaxQuote",
    "FaxStatus",
    "InboundFax",
    "InboxItem",
    "InboxPage",
)
