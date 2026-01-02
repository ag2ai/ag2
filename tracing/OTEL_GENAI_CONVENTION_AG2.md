# AG2 OpenTelemetry GenAI Semantic Convention Attributes

This document lists all `gen_ai.*` and related attributes used in AG2's instrumentation (`autogen/instrumentation.py`) and indicates whether they are part of the [OpenTelemetry GenAI Semantic Convention](https://opentelemetry.io/docs/specs/semconv/gen-ai/) or custom AG2 extensions.

## References

- [GenAI Spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/)
- [GenAI Agent Spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/)
- [Execute Tool Span](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/#execute-tool-span)

## Attribute Reference Table

| Attribute | OTEL Standard | AG2 Custom | Description |
|-----------|:-------------:|:----------:|-------------|
| **Core Attributes** ||||
| `gen_ai.operation.name` | ✅ | | Operation being performed (`invoke_agent`, `execute_tool`, `conversation`) |
| `gen_ai.agent.name` | ✅ | | Human-readable name of the GenAI agent |
| `gen_ai.agent.remote` | | ✅ | Indicates the agent is a remote A2A agent |
| `server.address` | ✅ | | URL of the remote agent server (standard OTEL attribute) |
| `error.type` | ✅ | | Error class if operation failed (standard OTEL attribute) |
| **AG2 Span Type** ||||
| `ag2.span.type` | | ✅ | AG2-specific span classification (`conversation`, `agent`, `tool`, `llm`, `handoff`, `speaker_selection`) |
| **Message Attributes** ||||
| `gen_ai.input.messages` | ✅ | | Input messages to the agent (JSON array) |
| `gen_ai.output.messages` | ✅ | | Output messages from the agent (JSON array) |
| **Conversation Attributes** ||||
| `gen_ai.conversation.id` | ✅ | | Unique conversation/session identifier (AG2 ChatResult.chat_id) |
| `gen_ai.conversation.max_turns` | | ✅ | Maximum turns configured for conversation |
| `gen_ai.conversation.turns` | | ✅ | Actual number of turns in conversation |
| `gen_ai.conversation.resumed` | | ✅ | Indicates a resumed conversation |
| **Usage Attributes** ||||
| `gen_ai.usage.input_tokens` | ✅ | | Number of input tokens consumed |
| `gen_ai.usage.output_tokens` | ✅ | | Number of output tokens generated |
| `gen_ai.usage.cost` | | ✅ | Total cost of the operation |
| `gen_ai.response.model` | ✅ | | Model used for the response |
| **Tool Attributes** ||||
| `gen_ai.tool.name` | ✅ | | Name of the tool being executed |
| `gen_ai.tool.type` | ✅ | | Type of tool (e.g., `function`) |
| `gen_ai.tool.call.id` | ✅ | | Unique identifier for the tool call |
| `gen_ai.tool.call.arguments` | ✅ | | Arguments passed to the tool (opt-in, may contain sensitive data) |
| `gen_ai.tool.call.result` | ✅ | | Result returned by the tool (opt-in, may contain sensitive data) |
| **Speaker Selection Attributes** ||||
| `ag2.speaker_selection.candidates` | | ✅ | JSON list of candidate agent names for speaker selection |
| `ag2.speaker_selection.selected` | | ✅ | Name of the agent selected as next speaker |
| **Human Input Attributes** ||||
| `ag2.human_input.prompt` | | ✅ | Prompt shown to the human for input |
| `ag2.human_input.response` | | ✅ | Human's response (opt-in, may contain sensitive data) |
| **Code Execution Attributes** ||||
| `ag2.code_execution.exit_code` | | ✅ | Exit code from code execution (0 = success) |
| `ag2.code_execution.output` | | ✅ | Output from code execution (truncated if > 4KB) |
| **Multi-Chat Attributes** ||||
| `ag2.chats.count` | | ✅ | Number of chats in initiate_chats workflow |
| `ag2.chats.mode` | | ✅ | Execution mode (`sequential` or `parallel`) |
| `ag2.chats.recipients` | | ✅ | JSON list of recipient agent names |
| `ag2.chats.ids` | | ✅ | JSON list of chat IDs (ChatResult.chat_id) |
| `ag2.chats.summaries` | | ✅ | JSON list of chat summaries after completion |
| `ag2.chats.prerequisites` | | ✅ | JSON map of chat dependencies (async parallel mode) |

## Span Types

AG2 uses the following span types (via `ag2.span.type`):

| Span Type | `gen_ai.operation.name` | Description |
|-----------|-------------------------|-------------|
| `multi_conversation` | `initiate_chats` | Multi-chat workflow (`initiate_chats`, `a_initiate_chats`) |
| `conversation` | `conversation` | Top-level chat initiation (`initiate_chat`, `a_initiate_chat`, `a_run_chat`) |
| `agent` | `invoke_agent` | Agent reply generation (`generate_reply`, `a_generate_reply`, `a_generate_remote_reply`) |
| `tool` | `execute_tool` | Tool/function execution (`execute_function`, `a_execute_function`) |
| `speaker_selection` | `speaker_selection` | Group chat speaker selection (`a_auto_select_speaker`, `_auto_select_speaker`) |
| `human_input` | `await_human_input` | Human-in-the-loop input (`get_human_input`, `a_get_human_input`) |
| `code_execution` | `execute_code` | Code block execution (`_generate_code_execution_reply_using_executor`) |
| `llm` | `chat` | LLM invocation (TODO) |
| `handoff` | `handoff` | Agent handoff (TODO) |

## Notes

- Attributes marked as **OTEL Standard** follow the [OpenTelemetry GenAI Semantic Convention](https://opentelemetry.io/docs/specs/semconv/gen-ai/).
- Attributes marked as **AG2 Custom** are extensions specific to AG2's multi-agent framework.
- `gen_ai.tool.call.arguments` and `gen_ai.tool.call.result` are opt-in attributes that may contain sensitive information.
- Message attributes (`gen_ai.input.messages`, `gen_ai.output.messages`) follow the OTEL message format with `role` and `parts` structure.
