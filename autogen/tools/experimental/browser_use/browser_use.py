# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Any, Awaitable, Callable, Optional

from pydantic import BaseModel

from ....import_utils import optional_import_block, require_optional_import
from ... import Depends, Tool

with optional_import_block():
    from browser_use import Agent
    from browser_use.browser.browser import Browser
    from langchain_openai import ChatOpenAI

__all__ = ["BrowserUseResult", "BrowserUseTool"]


class BrowserUseResult(BaseModel):
    extracted_content: list[str]
    final_result: Optional[str]


@require_optional_import(["langchain_openai", "browser_use"], "browser-use")
def get_browser_use_function(  # type: ignore[no-any-unimported]
    api_key_f: Callable[[], str],
    browser_f: Callable[[], Optional[Browser]],
    generate_gif_f: Callable[[], bool],
) -> Callable[[str, str, Any, bool], Awaitable[BrowserUseResult]]:
    async def browser_use(  # type: ignore[no-any-unimported]
        task: Annotated[str, "The task to perform."],
        api_key: Annotated[str, Depends(api_key_f)],
        browser: Annotated[Optional[Browser], Depends(browser_f)],
        generate_gif: Annotated[bool, Depends(generate_gif_f)],
    ) -> BrowserUseResult:
        agent = Agent(
            task=task,
            llm=ChatOpenAI(model="gpt-4o", api_key=api_key),
            browser=browser,
            generate_gif=generate_gif,
        )
        result = await agent.run()

        return BrowserUseResult(
            extracted_content=result.extracted_content(),
            final_result=result.final_result(),
        )

    return browser_use


@require_optional_import(["langchain_openai", "browser_use"], "browser-use")
class BrowserUseTool(Tool):
    def __init__(self, api_key: str, browser: Optional[Browser] = None, generate_gif: bool = True):  # type: ignore[no-any-unimported]
        def api_key_f() -> str:
            return api_key

        def browser_f() -> Optional[Browser]:  # type: ignore[no-any-unimported]
            return browser

        def generate_gif_f() -> bool:
            return generate_gif

        browser_use = get_browser_use_function(api_key_f, browser_f, generate_gif_f)

        super().__init__(
            name="browser_use",
            description="Use the browser to perform a task.",
            func_or_tool=browser_use,
        )
