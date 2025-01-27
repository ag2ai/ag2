# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Any, Callable, Optional, TypeVar

from pydantic import BaseModel

from ....import_utils import optional_import_block, require_optional_import
from ... import Depends, Tool

with optional_import_block():
    from browser_use import Agent
    from browser_use.browser.browser import Browser, BrowserConfig
    from langchain_openai import ChatOpenAI

__all__ = ["BrowserUseResult", "BrowserUseTool"]


class BrowserUseResult(BaseModel):
    """The result of using the browser to perform a task.

    Attributes:
        extracted_content: List of extracted content.
        final_result: The final result.
    """

    extracted_content: list[str]
    final_result: Optional[str]


T = TypeVar("T")


def on(x: T) -> Callable[[], T]:
    def inner(_x: T = x) -> T:
        return _x

    return inner


@require_optional_import(["langchain_openai", "browser_use"], "browser-use")
class BrowserUseTool(Tool):
    """BrowserUseTool is a tool that uses the browser to perform a task."""

    def __init__(  # type: ignore[no-any-unimported]
        self,
        *,
        llm_config: dict[str, Any],
        browser: Optional["Browser"] = None,
        agent_kwargs: Optional[dict[str, Any]] = None,
        browser_config: Optional[dict[str, Any]] = None,
    ):
        """Use the browser to perform a task.

        Args:
            llm_config: The LLM configuration.
            browser: The browser to use. If defined, browser_config must be None
            agent_kwargs: Additional keyword arguments to pass to the Agent
            browser_config: The browser configuration to use. If defined, browser must be None
        """
        if agent_kwargs is None:
            agent_kwargs = {}

        if browser_config is None:
            browser_config = {}

        if browser is not None and browser_config:
            raise ValueError(
                f"Cannot provide both browser and additional keyword parameters: {browser=}, {browser_config=}"
            )

        if browser is None:
            # set default value for headless
            headless = browser_config.pop("headless", True)

            browser_config = BrowserConfig(headless=headless, **browser_config)
            browser = Browser(config=browser_config)

        # set default value for generate_gif
        if "generate_gif" not in agent_kwargs:
            agent_kwargs["generate_gif"] = False

        try:
            model: str = llm_config["config_list"][0]["model"]  # type: ignore[index]
            api_key: str = llm_config["config_list"][0]["api_key"]  # type: ignore[index]
        except (KeyError, TypeError):
            raise ValueError("llm_config must be a valid config dictionary.")

        async def browser_use(  # type: ignore[no-any-unimported]
            task: Annotated[str, "The task to perform."],
            api_key: Annotated[str, Depends(on(api_key))],
            browser: Annotated[Browser, Depends(on(browser))],
            agent_kwargs: Annotated[dict[str, Any], Depends(on(agent_kwargs))],
        ) -> BrowserUseResult:
            agent = Agent(task=task, llm=ChatOpenAI(model=model, api_key=api_key), browser=browser, **agent_kwargs)
            result = await agent.run()

            return BrowserUseResult(
                extracted_content=result.extracted_content(),
                final_result=result.final_result(),
            )

        super().__init__(
            name="browser_use",
            description="Use the browser to perform a task.",
            func_or_tool=browser_use,
        )
