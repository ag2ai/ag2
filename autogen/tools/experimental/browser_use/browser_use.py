# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Awaitable, Callable, Union

from ....import_utils import optional_import_block, require_optional_import
from ... import Depends, Tool

with optional_import_block():
    from browser_use import Agent
    from langchain_openai import ChatOpenAI

__all__ = ["BrowserUseTool"]


@require_optional_import(["langchain_openai", "browser_use"], "browser-use")
def get_browser_use_function(api_key_f: Callable[[], str]) -> Callable[[str, str], Awaitable[str]]:
    async def browser_use(
        task: Annotated[str, "The task to perform."],
        api_key: Annotated[str, Depends(api_key_f)],
    ) -> str:
        agent = Agent(
            task=task,
            llm=ChatOpenAI(model="gpt-4o", api_key=api_key),
        )
        result = await agent.run()
        return str(result)

    return browser_use


@require_optional_import(["langchain_openai", "browser_use"], "browser-use")
class BrowserUseTool(Tool):
    def __init__(self, api_key: Union[str, Callable[[], str]]):
        api_key_f: Callable[[], str] = (lambda api_key=api_key: api_key) if isinstance(api_key, str) else api_key  # type: ignore[misc]

        browser_use = get_browser_use_function(api_key_f)

        super().__init__(
            name="browser_use",
            description="Use the browser to perform a task.",
            func_or_tool=browser_use,
        )


# asyncio.run(main())
