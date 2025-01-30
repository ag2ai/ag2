# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Any, Optional

from pydantic import BaseModel

from ....doc_utils import export_module
from ....import_utils import optional_import_block, require_optional_import
from ... import Depends, Tool
from ...dependency_injection import on

with optional_import_block():
    from browser_use import Agent
    from browser_use.browser.browser import Browser, BrowserConfig
    from langchain_anthropic import ChatAnthropic
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_openai import ChatOpenAI

__all__ = ["BrowserUseResult", "BrowserUseTool"]


@export_module("autogen.tools.experimental.browser_use")
class BrowserUseResult(BaseModel):
    """The result of using the browser to perform a task.

    Attributes:
        extracted_content: List of extracted content.
        final_result: The final result.
    """

    extracted_content: list[str]
    final_result: Optional[str]


@require_optional_import(["langchain_openai", "browser_use"], "browser-use")
@export_module("autogen.tools.experimental")
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

        async def browser_use(  # type: ignore[no-any-unimported]
            task: Annotated[str, "The task to perform."],
            llm_config: Annotated[dict[str, Any], Depends(on(llm_config))],
            browser: Annotated[Browser, Depends(on(browser))],
            agent_kwargs: Annotated[dict[str, Any], Depends(on(agent_kwargs))],
        ) -> BrowserUseResult:
            llm = BrowserUseTool._get_llm(llm_config)
            agent = Agent(task=task, llm=llm, browser=browser, **agent_kwargs)
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

    @staticmethod
    def _get_llm(  # type: ignore[no-any-unimported]
        llm_config: dict[str, Any],
    ) -> Any:
        if "config_list" not in llm_config:
            if "model" in llm_config:
                return ChatOpenAI(model=llm_config["model"])
            raise ValueError("llm_config must be a valid config dictionary.")

        try:
            model = llm_config["config_list"][0]["model"]
            api_type = llm_config["config_list"][0].get("api_type", "openai")
            api_key = llm_config["config_list"][0]["api_key"]
        except (KeyError, TypeError):
            raise ValueError("llm_config must be a valid config dictionary.")

        if api_type == "openai":
            return ChatOpenAI(model=model, api_key=api_key)
        elif api_type == "deepseek":
            return ChatOpenAI(model=model, api_key=api_key, base_url=llm_config["config_list"][0].get("base_url"))
        elif api_type == "anthropic":
            return ChatAnthropic(model=model, api_key=api_key)
        elif api_type == "google":
            return ChatGoogleGenerativeAI(model=model, api_key=api_key)
        else:
            raise ValueError(f"Currently unsupported language model api type for browser use: {api_type}")
