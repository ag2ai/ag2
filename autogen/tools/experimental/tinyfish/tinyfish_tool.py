# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from typing import Annotated, Any

from ....doc_utils import export_module
from ....import_utils import optional_import_block, require_optional_import
from ....llm_config import LLMConfig
from ... import Depends, Tool
from ...dependency_injection import on

logger = logging.getLogger(__name__)

with optional_import_block():
    from tinyfish import TinyFish


@require_optional_import(
    [
        "tinyfish",
    ],
    "tinyfish",
)
def _execute_tinyfish_scrape(
    url: str,
    goal: str,
    tinyfish_api_key: str,
) -> dict[str, Any]:
    """Execute a goal-directed scrape using the TinyFish API.

    Args:
        url (str): The URL to scrape.
        goal (str): A natural language description of what to extract.
        tinyfish_api_key (str): The API key for TinyFish.

    Returns:
        dict[str, Any]: The scrape result from TinyFish.
    """
    client = TinyFish(api_key=tinyfish_api_key)
    response = client.agent.run(url=url, goal=goal)

    if response.status == "COMPLETED" and response.result:
        if isinstance(response.result, dict):
            return response.result
        return {"result": response.result}
    elif response.error:
        return {"status": "error", "error": response.error}
    else:
        return {"status": "no_result", "error": "No result returned."}


def _tinyfish_scrape(
    url: str,
    goal: str,
    tinyfish_api_key: str,
) -> dict[str, Any]:
    """Perform a TinyFish scrape and format the results.

    Args:
        url (str): The URL to scrape.
        goal (str): A natural language description of what to extract.
        tinyfish_api_key (str): The API key for TinyFish.

    Returns:
        dict[str, Any]: The scraped data, or an error dict.
    """
    try:
        result = _execute_tinyfish_scrape(
            url=url,
            goal=goal,
            tinyfish_api_key=tinyfish_api_key,
        )
        return {
            "url": url,
            "goal": goal,
            "data": result,
        }
    except Exception as e:
        logger.error(f"TinyFish scrape failed: {e}")
        return {
            "url": url,
            "goal": goal,
            "data": {},
            "error": str(e),
        }


@export_module("autogen.tools.experimental")
class TinyFishTool(Tool):
    """TinyFishTool is a tool that uses the TinyFish API to deep-scrape web pages with a natural language goal.

    TinyFish performs goal-directed web scraping — you provide a URL and describe what information
    you want to extract, and it returns structured results.

    This tool requires a TinyFish API key, which can be provided during initialization or set as
    an environment variable ``TINYFISH_API_KEY``.

    Attributes:
        tinyfish_api_key (str): The API key used for authenticating with the TinyFish API.
    """

    def __init__(
        self,
        *,
        llm_config: LLMConfig | dict[str, Any] | None = None,
        tinyfish_api_key: str | None = None,
    ):
        """Initializes the TinyFishTool.

        Args:
            llm_config (Optional[Union[LLMConfig, dict[str, Any]]]): LLM configuration.
                (Currently unused but kept for potential future integration).
            tinyfish_api_key (Optional[str]): The API key for the TinyFish API. If not provided,
                it attempts to read from the ``TINYFISH_API_KEY`` environment variable.

        Raises:
            ValueError: If ``tinyfish_api_key`` is not provided either directly or via the environment variable.
        """
        self.tinyfish_api_key = tinyfish_api_key or os.getenv("TINYFISH_API_KEY")

        if self.tinyfish_api_key is None:
            raise ValueError("tinyfish_api_key must be provided either as an argument or via TINYFISH_API_KEY env var")

        def tinyfish_scrape(
            url: Annotated[str, "The URL to scrape."],
            goal: Annotated[str, "A natural language description of what information to extract from the page."],
            tinyfish_api_key: Annotated[str | None, Depends(on(self.tinyfish_api_key))],
        ) -> dict[str, Any]:
            """Deep-scrape a URL using TinyFish with a natural language goal.

            Pass a URL and describe what you want to extract. TinyFish will navigate the page
            and return structured data matching your goal.

            Args:
                url: The URL to scrape.
                goal: A natural language description of what to extract (e.g.,
                    "Find all team members and their roles",
                    "Extract pricing information and plan details").
                tinyfish_api_key: The API key for TinyFish (injected dependency).

            Returns:
                A dictionary containing the URL, goal, and extracted data.

            Raises:
                ValueError: If the TinyFish API key is not available.
            """
            if tinyfish_api_key is None:
                raise ValueError("TinyFish API key is missing.")
            return _tinyfish_scrape(
                url=url,
                goal=goal,
                tinyfish_api_key=tinyfish_api_key,
            )

        super().__init__(
            name="tinyfish_scrape",
            description="Deep-scrape a URL using TinyFish. Pass a URL and a natural language goal describing what to extract.",
            func_or_tool=tinyfish_scrape,
        )
