---
sidebarTitle: AG-UI
title: "AG-UI (Agent-User Interaction) Integration"
description: "Expose AG2 agents to AG-UI-compatible frontends using AGUIStream."
---

## Overview

The Agent-User Interaction (AG-UI) protocol standardizes how frontend applications communicate with agents.
In AG2, `ag2.ag_ui.AGUIStream` bridges an `Agent` to AG-UI event streams.

This solves common integration problems:

- Streaming agent output to UI clients
- Emitting tool-call lifecycle events
- Synchronizing shared state snapshots
- Supporting human-in-the-loop checkpoints through frontend actions and input-required flows

For protocol background, see [AG-UI Protocol introduction](https://docs.ag-ui.com/introduction){.external-link target="_blank"}.

## When to use AG-UI vs direct integration

| Approach | Use it when | Trade-offs |
| --- | --- | --- |
| AG-UI integration (`AGUIStream`) | You need streaming UI, tool rendering, shared state sync, and a protocol-compatible client ecosystem | Adds protocol event semantics you need to expose from your endpoint |
| Direct integration (custom REST/WebSocket contract) | You only need a narrow, app-specific API and will own protocol design end-to-end | You must define and maintain your own streaming/tool/state contract |

Use AG-UI when you want a reusable UI contract across clients and frameworks.

## Supported capabilities

Verified AG-UI features are supported in AG2:

* [x] Streaming text events (`TEXT_MESSAGE_START`, `TEXT_MESSAGE_CONTENT`, `TEXT_MESSAGE_END`, `TEXT_MESSAGE_CHUNK`)
* [x] [Backend tool lifecycle events](https://docs.copilotkit.ai/ag2/generative-ui/backend-tools){.external-link target="_blank"} (`TOOL_CALL_START`, `TOOL_CALL_ARGS`, `TOOL_CALL_RESULT`, `TOOL_CALL_END`)
* [x] [Frontend-tool dispatch](https://docs.copilotkit.ai/ag2/generative-ui/frontend-tools){.external-link target="_blank"} (`TOOL_CALL_CHUNK` for client tools in `RunAgentInput.tools`)
* [x] [Shared-state snapshots](https://docs.copilotkit.ai/ag2/shared-state){.external-link target="_blank"} (`STATE_SNAPSHOT`) from context and agent state
* [x] [Human input checkpoints](https://docs.copilotkit.ai/ag2/human-in-the-loop){.external-link target="_blank"} (`input_required` surfaced as user-visible message events)

## Installation

Install AG2 with AG-UI support, plus the extra for the model provider your agent uses — `openai` in the examples below:

```bash
pip install "ag2[ag-ui,openai]"
```

## Basic server example

Use the manual-dispatch pattern when you want full control over auth, logging, and middleware:

```python title="run_ag_ui.py" linenums="1" hl_lines="14 17-25"
from fastapi import FastAPI, Header
from fastapi.responses import StreamingResponse

from ag2 import Agent
from ag2.ag_ui import AGUIStream, RunAgentInput
from ag2.config import OpenAIConfig

agent = Agent(
    name="support_bot",
    prompt="You help users with billing questions.",
    config=OpenAIConfig(model="gpt-4o-mini"),
)

stream = AGUIStream(agent)
app = FastAPI()

@app.post("/chat")
async def run_agent(
    message: RunAgentInput,
    accept: str | None = Header(None),
) -> StreamingResponse:
    return StreamingResponse(
        stream.dispatch(message, accept=accept),
        media_type=accept or "text/event-stream",
    )
```

Run it:

```bash
uvicorn run_ag_ui:app --reload --port 8000
```

!!! note "Simpler way"
    If you want to use ASGI endpoint without additional logic, you can use the `AGUIStream.build_asgi()` method to build an ASGI endpoint and mount it to your ASGI application.

    ```python linenums="1" hl_lines="5-6"
    from ag2.ag_ui import AGUIStream
    from fastapi import FastAPI

    app = FastAPI()
    stream = AGUIStream(agent)
    app.mount("/chat", stream.build_asgi())
    ```

### Test the endpoint

```bash
curl -N -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "thread_id": "thread-1",
    "run_id": "run-1",
    "messages": [{"id": "m1", "role": "user", "content": "Hello"}],
    "state": {},
    "context": [],
    "tools": [],
    "forwardedProps": null
  }'
```

Example stream (truncated):

```text
data: {"type":"RUN_STARTED","threadId":"thread-1","runId":"run-1",...}
data: {"type":"TEXT_MESSAGE_CHUNK","delta":"Hello! How can I help?",...}
data: {"type":"RUN_FINISHED","threadId":"thread-1","runId":"run-1",...}
```

## Token usage

Both run-terminating events carry the run's token usage, broken down per
provider and model configuration:

```text
data: {"type":"RUN_FINISHED","threadId":"thread-1","runId":"run-1",
       "usage":[{"provider":"openai","model":"gpt-5","inputTokens":1284,
                 "outputTokens":96,"totalTokens":1380,"cachedInputTokens":1024}]}
```

`RUN_ERROR` carries the same field, reporting what the run had spent before it
failed, so a crashed run is not accounted for as free. (`RUN_ERROR` has no
`threadId` / `runId` — the protocol defines neither for it; a client correlates
the run from `RUN_STARTED` on the same event stream.)

The figures cover every model call in the run — the whole tool loop, delegated
sub-agents, history compaction and memory aggregation — and agree with
`AgentReply.usage()` for the same run, because both read the same accounting
events.

An entry carries `inputTokens`, `outputTokens`, `totalTokens`, `reasoningTokens`
and `cachedInputTokens`. Any of them may be missing — three properties are worth
knowing before you sum them:

- **A count the provider did not report is absent, not `0`.** `cachedInputTokens`
  absent means "never measured"; `0` means the provider measured no cache hit.
  The same holds for `reasoningTokens`, which most non-reasoning calls omit.
- **`totalTokens` is never derived.** It appears only when every call behind an
  entry supplied one, so a client is never handed a computed figure dressed up
  as a measurement. Sum `inputTokens` and `outputTokens` yourself if you need a
  figure regardless.
- **Cache *writes* are not reported at all.** `cachedInputTokens` counts a cache
  *read*. Tokens spent creating a cache entry have no field in the protocol, and
  they are omitted rather than folded into a neighbouring count, because
  providers disagree on whether cached tokens already sit in the prompt figure.
  A cost model built on these entries alone under-counts a run that primed a
  cache; read `AgentReply.usage()` server-side if you need that number.

One entry is emitted per distinct provider/model pair, in order of first
appearance in the run. A delegated sub-agent that used a single configuration is
attributed to it; one that spanned several arrives with `provider` and `model`
unset rather than mislabelled. A run that spent nothing omits `usage` entirely.

The A2UI AG-UI transport (`A2UIServer(..., transport=AgUiTransport())`) reports
the same field on the same two events, so what a client can show does not depend
on which endpoint it connected to.

## UI clients

Any AG-UI client works with this endpoint.

For React/Next.js UIs, CopilotKit is the recommended client path in AG2 docs because it provides:

- Streaming chat components
- Tool UI rendering hooks/components
- Shared state patterns for interactive workflows

Start from the [CopilotKit UI quickstart](/docs/user-guide/ag-ui/copilotkit-quickstart).

The same endpoint can also power a bot in Slack and other messaging platforms — see [Channels](/docs/user-guide/ag-ui/channels).

## AG-UI Dojo

For protocol-level testing and event inspection, use the AG2 Dojo profile:

- [AG2 Dojo - agentic_chat](https://dojo.ag-ui.com/ag2/feature/agentic_chat){.external-link target="_blank"}

## Next steps

1. Build the AG-UI endpoint from the minimal example above.
2. Follow the [CopilotKit UI quickstart](/docs/user-guide/ag-ui/copilotkit-quickstart) to connect a React/Next.js client.
3. Validate runtime behavior with the [AG2 Dojo - agentic_chat](https://dojo.ag-ui.com/ag2/feature/agentic_chat){.external-link target="_blank"}.
4. Run the same endpoint as a bot in Slack and other messaging platforms with [Channels](/docs/user-guide/ag-ui/channels).
