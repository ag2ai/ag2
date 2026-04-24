# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from dirty_equals import IsPartialDict

pytest.importorskip("exa_py")

from autogen.beta import Agent, DataInput, Variable
from autogen.beta.context import ConversationContext
from autogen.beta.events import ModelResponse, ToolCallEvent, ToolCallsEvent, ToolResultsEvent
from autogen.beta.testing import TestConfig, TrackingConfig
from autogen.beta.tools.search.exa import (
    ExaAnswerCitation,
    ExaAnswerResult,
    ExaContentResult,
    ExaSearchResponse,
    ExaSearchResult,
    ExaToolkit,
)


def _result(
    *,
    title: str = "AG2 Framework",
    url: str = "https://ag2.ai",
    score: float | None = 0.95,
    published_date: str | None = "2024-01-01",
    author: str | None = "ag2ai",
    text: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        title=title,
        url=url,
        score=score,
        published_date=published_date,
        author=author,
        text=text,
    )


def _response(results: list[SimpleNamespace], autoprompt_string: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(results=results, autoprompt_string=autoprompt_string)


def _tool_call_config(
    arguments: dict,
    *,
    tool_name: str,
    final_reply: str = "done",
) -> TestConfig:
    return TestConfig(
        ModelResponse(
            tool_calls=ToolCallsEvent([
                ToolCallEvent(arguments=json.dumps(arguments), name=tool_name),
            ]),
        ),
        final_reply,
    )


@pytest.mark.asyncio
class TestSchema:
    async def test_default_schemas(self, context: ConversationContext) -> None:
        toolkit = ExaToolkit(client=MagicMock())

        schemas = list(await toolkit.schemas(context))

        names = [s.function.name for s in schemas]
        assert names == ["exa_search", "exa_find_similar", "exa_get_contents", "exa_answer"]

    async def test_search_schema_has_query_param(self, context: ConversationContext) -> None:
        toolkit = ExaToolkit(client=MagicMock())

        schemas = list(await toolkit.schemas(context))
        search_schema = next(s for s in schemas if s.function.name == "exa_search")

        assert search_schema.function.parameters == IsPartialDict({
            "required": ["query"],
            "properties": IsPartialDict({"query": IsPartialDict({"type": "string"})}),
        })

    async def test_custom_tool_name_and_description(self, context: ConversationContext) -> None:
        toolkit = ExaToolkit(client=MagicMock())
        custom = toolkit.search(name="neural_search", description="Custom neural search.")

        [schema] = list(await custom.schemas(context))

        assert schema.function.name == "neural_search"
        assert schema.function.description == "Custom neural search."


@pytest.mark.asyncio
class TestSearchExecution:
    async def test_returns_structured_results(self, mock: MagicMock) -> None:
        mock.search.return_value = _response([
            _result(title="AG2 Framework", url="https://ag2.ai", score=0.95, text=None),
            _result(title="GitHub - AG2", url="https://github.com/ag2ai/ag2", score=0.82, text=None),
        ])
        toolkit = ExaToolkit(client=mock)

        config = TrackingConfig(_tool_call_config({"query": "AG2 framework"}, tool_name="exa_search"))
        agent = Agent("a", config=config, tools=[toolkit])

        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            ExaSearchResponse(
                query="AG2 framework",
                results=[
                    ExaSearchResult(
                        title="AG2 Framework",
                        url="https://ag2.ai",
                        score=0.95,
                        published_date="2024-01-01",
                        author="ag2ai",
                        text=None,
                    ),
                    ExaSearchResult(
                        title="GitHub - AG2",
                        url="https://github.com/ag2ai/ag2",
                        score=0.82,
                        published_date="2024-01-01",
                        author="ag2ai",
                        text=None,
                    ),
                ],
                autoprompt_string=None,
            )
        )

    async def test_empty_results(self, mock: MagicMock) -> None:
        mock.search.return_value = _response([])
        toolkit = ExaToolkit(client=mock)

        config = TrackingConfig(_tool_call_config({"query": "nothing"}, tool_name="exa_search"))
        agent = Agent("a", config=config, tools=[toolkit])

        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            ExaSearchResponse(query="nothing", results=[])
        )

    async def test_none_params_omitted(self, mock: MagicMock) -> None:
        mock.search.return_value = _response([])
        toolkit = ExaToolkit(client=mock)

        agent = Agent("a", config=_tool_call_config({"query": "q"}, tool_name="exa_search"), tools=[toolkit])

        await agent.ask("search")

        mock.search.assert_called_once_with("q")

    async def test_all_params_forwarded(self, mock: MagicMock) -> None:
        mock.search.return_value = _response([])
        toolkit = ExaToolkit(client=mock)
        search_tool = toolkit.search(
            num_results=7,
            search_type="neural",
            category="research paper",
            include_domains=["arxiv.org"],
            exclude_domains=["medium.com"],
            start_published_date="2024-01-01",
            end_published_date="2024-12-31",
            start_crawl_date="2024-01-01",
            end_crawl_date="2024-12-31",
            use_autoprompt=True,
            livecrawl="always",
        )

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="exa_search"),
            tools=[search_tool],
        )
        await agent.ask("search")

        mock.search.assert_called_once_with(
            "q",
            num_results=7,
            type="neural",
            category="research paper",
            include_domains=["arxiv.org"],
            exclude_domains=["medium.com"],
            start_published_date="2024-01-01",
            end_published_date="2024-12-31",
            start_crawl_date="2024-01-01",
            end_crawl_date="2024-12-31",
            use_autoprompt=True,
            livecrawl="always",
        )

    async def test_search_and_contents_when_max_characters_set_on_method(self, mock: MagicMock) -> None:
        mock.search_and_contents.return_value = _response([_result(text="full article text")])
        toolkit = ExaToolkit(client=mock)

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="exa_search"),
            tools=[toolkit.search(max_characters=500)],
        )
        await agent.ask("search")

        mock.search.assert_not_called()
        mock.search_and_contents.assert_called_once_with("q", text={"maxCharacters": 500})

    async def test_toolkit_level_max_characters_applied_to_default_search(self, mock: MagicMock) -> None:
        mock.search_and_contents.return_value = _response([])
        toolkit = ExaToolkit(client=mock, max_characters=800)

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="exa_search"),
            tools=[toolkit],
        )
        await agent.ask("search")

        mock.search.assert_not_called()
        mock.search_and_contents.assert_called_once_with("q", text={"maxCharacters": 800})

    async def test_toolkit_level_num_results_applied_to_default_search(self, mock: MagicMock) -> None:
        mock.search.return_value = _response([])
        toolkit = ExaToolkit(client=mock, num_results=7)

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="exa_search"),
            tools=[toolkit],
        )
        await agent.ask("search")

        mock.search.assert_called_once_with("q", num_results=7)


@pytest.mark.asyncio
class TestFindSimilar:
    async def test_num_results_forwarded_from_method(self, mock: MagicMock) -> None:
        mock.find_similar.return_value = _response([_result(title="Similar page")])
        toolkit = ExaToolkit(client=mock)

        agent = Agent(
            "a",
            config=_tool_call_config({"url": "https://ag2.ai"}, tool_name="exa_find_similar"),
            tools=[toolkit.find_similar(num_results=3)],
        )
        await agent.ask("find similar")

        mock.find_similar.assert_called_once_with("https://ag2.ai", num_results=3)

    async def test_num_results_forwarded_from_toolkit(self, mock: MagicMock) -> None:
        mock.find_similar.return_value = _response([])
        toolkit = ExaToolkit(client=mock, num_results=4)

        agent = Agent(
            "a",
            config=_tool_call_config({"url": "https://ag2.ai"}, tool_name="exa_find_similar"),
            tools=[toolkit],
        )
        await agent.ask("find similar")

        mock.find_similar.assert_called_once_with("https://ag2.ai", num_results=4)

    async def test_exclude_source_domain_forwarded(self, mock: MagicMock) -> None:
        mock.find_similar.return_value = _response([])
        toolkit = ExaToolkit(client=mock)

        agent = Agent(
            "a",
            config=_tool_call_config({"url": "https://ag2.ai"}, tool_name="exa_find_similar"),
            tools=[toolkit.find_similar(num_results=2, exclude_source_domain=True)],
        )
        await agent.ask("find similar")

        mock.find_similar.assert_called_once_with("https://ag2.ai", num_results=2, exclude_source_domain=True)


@pytest.mark.asyncio
class TestGetContents:
    async def test_returns_content(self, mock: MagicMock) -> None:
        mock.get_contents.return_value = _response([
            SimpleNamespace(
                url="https://ag2.ai",
                title="AG2",
                text="full text",
                author="ag2ai",
                published_date="2024-01-01",
            ),
        ])
        toolkit = ExaToolkit(client=mock)

        config = TrackingConfig(_tool_call_config({"urls": ["https://ag2.ai"]}, tool_name="exa_get_contents"))
        agent = Agent("a", config=config, tools=[toolkit])
        await agent.ask("get contents")

        mock.get_contents.assert_called_once_with(["https://ag2.ai"], text=True)

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput([
            ExaContentResult(
                url="https://ag2.ai",
                title="AG2",
                text="full text",
                author="ag2ai",
                published_date="2024-01-01",
            )
        ])


@pytest.mark.asyncio
class TestAnswer:
    async def test_returns_answer_with_citations(self, mock: MagicMock) -> None:
        mock.answer.return_value = SimpleNamespace(
            answer="AG2 is an open-source multi-agent framework.",
            citations=[
                SimpleNamespace(url="https://ag2.ai", title="AG2", text="About AG2"),
                SimpleNamespace(url="https://github.com/ag2ai/ag2", title="GitHub", text="Source code"),
            ],
        )
        toolkit = ExaToolkit(client=mock)

        config = TrackingConfig(_tool_call_config({"query": "What is AG2?"}, tool_name="exa_answer"))
        agent = Agent("a", config=config, tools=[toolkit])
        await agent.ask("answer")

        mock.answer.assert_called_once_with("What is AG2?", text=True)

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            ExaAnswerResult(
                answer="AG2 is an open-source multi-agent framework.",
                citations=[
                    ExaAnswerCitation(url="https://ag2.ai", title="AG2", text="About AG2"),
                    ExaAnswerCitation(url="https://github.com/ag2ai/ag2", title="GitHub", text="Source code"),
                ],
            )
        )


@pytest.mark.asyncio
class TestExaToolkitVariable:
    async def test_resolved(self, mock: MagicMock) -> None:
        mock.search.return_value = _response([])
        toolkit = ExaToolkit(client=mock)
        search_tool = toolkit.search(
            num_results=Variable("user_limit"),
            search_type=Variable(),
        )

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="exa_search"),
            tools=[search_tool],
            variables={"user_limit": 10, "search_type": "keyword"},
        )
        await agent.ask("search")

        mock.search.assert_called_once_with("q", num_results=10, type="keyword")

    async def test_missing_raises(self, mock: MagicMock) -> None:
        mock.search.return_value = _response([])
        toolkit = ExaToolkit(client=mock)
        search_tool = toolkit.search(search_type=Variable())

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="exa_search"),
            tools=[search_tool],
        )

        with pytest.raises(KeyError, match="search_type"):
            await agent.ask("search")


@pytest.mark.asyncio
class TestIndividualTools:
    async def test_search_tool_passed_alone(self, mock: MagicMock) -> None:
        mock.search.return_value = _response([_result()])
        toolkit = ExaToolkit(client=mock)

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="exa_search"),
            tools=[toolkit.search()],
        )
        await agent.ask("search")

        mock.search.assert_called_once_with("q")

    async def test_pick_two_tools_from_toolkit(self, mock: MagicMock) -> None:
        mock.search.return_value = _response([])
        mock.answer.return_value = SimpleNamespace(answer="ok", citations=[])
        toolkit = ExaToolkit(client=mock)

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="exa_search"),
            tools=[toolkit.search(), toolkit.answer()],
        )
        await agent.ask("search")

        mock.search.assert_called_once_with("q")
