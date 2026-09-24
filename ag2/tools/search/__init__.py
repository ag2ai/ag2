# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from ag2.exceptions import missing_optional_dependency

# Imported twice on purpose: the checker is shown only the real import, because
# rebinding a name it has already bound to a class is an error, and the install hint
# is runtime behaviour. See website/docs/contributor-guide/type-checking.mdx.
if TYPE_CHECKING:
    from .duckduckgo import DuckDuckSearchTool
    from .perplexity import PerplexitySearchToolkit
    from .tavily import TavilySearchTool
else:
    try:
        from .tavily import TavilySearchTool
    except ImportError as e:
        TavilySearchTool = missing_optional_dependency("TavilySearchTool", "tavily", e)

    try:
        from .duckduckgo import DuckDuckSearchTool
    except ImportError as e:
        DuckDuckSearchTool = missing_optional_dependency("DuckDuckSearchTool", "ddgs", e)

    try:
        from .perplexity import PerplexitySearchToolkit
    except ImportError as e:
        PerplexitySearchToolkit = missing_optional_dependency("PerplexitySearchToolkit", "perplexity", e)

__all__ = (
    "DuckDuckSearchTool",
    "PerplexitySearchToolkit",
    "TavilySearchTool",
)
