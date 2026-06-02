# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A test MCP server that exposes prompts for testing."""

from mcp.server import Server, ServerRequestContext
from mcp.server.stdio import stdio_server
from mcp.types import (
    GetPromptRequestParams,
    GetPromptResult,
    ListPromptsResult,
    PaginatedRequestParams,
    Prompt,
    PromptArgument,
    PromptMessage,
    TextContent,
)


def create_prompts_server() -> Server:
    """Create an MCP server with test prompts."""

    async def handle_list_prompts(
        _context: ServerRequestContext,
        _params: PaginatedRequestParams | None,
    ) -> ListPromptsResult:
        return ListPromptsResult(
            prompts=[
                Prompt(
                    name="greeting",
                    description="A simple greeting prompt",
                ),
                Prompt(
                    name="echo",
                    description="Echo a user-provided message",
                    arguments=[
                        PromptArgument(
                            name="message",
                            description="The message to echo",
                            required=True,
                        ),
                    ],
                ),
                Prompt(
                    name="code_review",
                    description="Review code with optional language specification",
                    arguments=[
                        PromptArgument(
                            name="code",
                            description="The code to review",
                            required=True,
                        ),
                        PromptArgument(
                            name="language",
                            description="The programming language",
                            required=False,
                        ),
                    ],
                ),
            ],
        )

    async def handle_get_prompt(
        _context: ServerRequestContext,
        params: GetPromptRequestParams,
    ) -> GetPromptResult:
        if params.name == "greeting":
            return GetPromptResult(
                description="A greeting",
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(
                            type="text",
                            text="Hello! How can I help you today?",
                        ),
                    ),
                ],
            )
        elif params.name == "echo":
            message = (params.arguments or {}).get("message", "")
            return GetPromptResult(
                description=f"Echo: {message}",
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(
                            type="text",
                            text=f"Echoing: {message}",
                        ),
                    ),
                ],
            )
        elif params.name == "code_review":
            code = (params.arguments or {}).get("code", "")
            language = (params.arguments or {}).get("language", "unknown")
            return GetPromptResult(
                description=f"Code review for {language}",
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(
                            type="text",
                            text=f"Please review this {language} code:\n\n```{language}\n{code}\n```",
                        ),
                    ),
                ],
            )
        else:
            raise ValueError(f"Unknown prompt: {params.name}")

    server = Server(
        "test-prompts-server",
        on_list_prompts=handle_list_prompts,
        on_get_prompt=handle_get_prompt,
    )

    return server


async def main() -> None:
    """Run the prompts test server over stdio."""
    server = create_prompts_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    import anyio

    anyio.run(main)
