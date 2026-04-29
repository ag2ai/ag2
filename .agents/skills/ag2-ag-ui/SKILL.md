---
name: ag2-ag-ui
description: Expose an AG2 beta `Agent` over the AG-UI protocol so a frontend (CopilotKit, custom React/Next.js, or any AG-UI client) can stream responses, render tool calls, sync shared state, and surface human-input checkpoints. Wraps the agent with `AGUIStream(agent)` and mounts it in FastAPI via `stream.dispatch(...)` or `stream.build_asgi()`. Use when the user wants a web frontend in front of an AG2 agent rather than a CLI / script.
license: Apache-2.0
---

# AG-UI integration

## When to use

- The user is building a web UI (React / Next.js / anything HTTP+SSE) that should talk to an AG2 beta agent.
- They want streaming text, tool-call rendering, shared state sync, or HITL checkpoints surfaced to the frontend with a standard protocol — not a custom REST/WebSocket contract.
- They're using or considering CopilotKit (the recommended React client).

For a custom narrow-purpose API where you own the contract end-to-end, skip AG-UI and write a plain endpoint instead.

## Supported AG-UI features

| Feature | Status |
|---|---|
| Streaming text events (`TEXT_MESSAGE_*`, `TEXT_MESSAGE_CHUNK`) | ✓ |
| Backend tool lifecycle (`TOOL_CALL_START` / `_ARGS` / `_RESULT` / `_END`) | ✓ |
| Frontend-tool dispatch (`TOOL_CALL_CHUNK` for client tools in `RunAgentInput.tools`) | ✓ |
| Shared-state snapshots (`STATE_SNAPSHOT`) | ✓ |
| Human input checkpoints (surfaced as user-visible message events) | ✓ |

## Installation

```bash
pip install "ag2[ag-ui]"
```

## 60-second recipe — FastAPI server

```python title="run_ag_ui.py"
from fastapi import FastAPI, Header
from fastapi.responses import StreamingResponse

from autogen.beta import Agent
from autogen.beta.ag_ui import AGUIStream, RunAgentInput
from autogen.beta.config import OpenAIConfig

agent = Agent(
    name="support_bot",
    prompt="You help users with billing questions.",
    config=OpenAIConfig(model="gpt-4o-mini"),
)

stream = AGUIStream(agent)
app = FastAPI()

@app.post("/chat")
async def run_agent(message: RunAgentInput, accept: str | None = Header(None)) -> StreamingResponse:
    return StreamingResponse(
        stream.dispatch(message, accept=accept),
        media_type=accept or "text/event-stream",
    )
```

```bash
uvicorn run_ag_ui:app --reload --port 8000
```

Test:

```bash
curl -N -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"thread_id":"t1","run_id":"r1","messages":[{"id":"m1","role":"user","content":"Hello"}],"state":{},"context":[],"tools":[]}'
```

## Lower-friction alternative — `build_asgi()`

If you don't need custom auth / middleware, mount the ASGI endpoint directly:

```python
from autogen.beta.ag_ui import AGUIStream
from fastapi import FastAPI

app = FastAPI()
stream = AGUIStream(agent)
app.mount("/chat", stream.build_asgi())
```

## CopilotKit frontend (recommended for React / Next.js)

Two ways to start:

```bash
# Option A — CopilotKit bootstrap
npx copilotkit@latest create -f ag2

# Option B — clone the reference starter
# https://github.com/ag2ai/ag2-copilotkit-starter
```

The starter layout:

```
ag2-copilotkit-starter/
├── agent-py/     # Python backend (AG2 + AG-UI)
└── ui-react/     # React + CopilotKit frontend
```

Backend serves `/chat` on port 8008 by default. The Next.js route at `ui-react/app/api/copilotkit/route.ts` bridges the UI to the backend:

```tsx
import { HttpAgent } from "@ag-ui/client";
import { CopilotRuntime, ExperimentalEmptyAdapter, copilotRuntimeNextJSAppRouterEndpoint } from "@copilotkit/runtime";
import { NextRequest } from "next/server";

const agent = new HttpAgent({ url: "http://localhost:8008/chat" });
const runtime = new CopilotRuntime({ agents: { weather_agent: agent } });

export async function POST(req: NextRequest) {
  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime,
    serviceAdapter: new ExperimentalEmptyAdapter(),
    endpoint: "/api/copilotkit",
  });
  return handleRequest(req);
}
```

Wrap your app:

```tsx
import { CopilotKit } from "@copilotkit/react-core";
import "@copilotkit/react-ui/styles.css";

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <CopilotKit agent="weather_agent" runtimeUrl="/api/copilotkit">
          {children}
        </CopilotKit>
      </body>
    </html>
  );
}
```

Render chat (`<CopilotChat />`), and use `useCopilotAction({...})` in client components to render generative UI tied to backend tool calls.

## Production checklist

- **CORS** — allow your frontend origin on the AG-UI backend (`POST`, `OPTIONS`, auth headers).
- **Auth** — protect both `/api/copilotkit` (Next.js route) and backend `/chat`. Don't rely on client-only secrets.
- **SSE buffering** — verify backend response is `Content-Type: text/event-stream` and that proxy layers (Nginx, CloudFront, etc.) don't buffer SSE.
- **Timeouts / retries** — conservative values for long-running tool workflows; only retry idempotent requests.
- **Tool inputs are untrusted** — validate and authorize server-side before invoking privileged tools. Log tool execution with request IDs.
- **Tool name parity** — frontend `useCopilotAction` `name` must match backend `@tool` name exactly.

## Going deeper

- `website/docs/beta/ag-ui/index.mdx` — `AGUIStream`, supported events, basic server.
- `website/docs/beta/ag-ui/copilotkit-quickstart.mdx` — full React + CopilotKit walkthrough (file layout, route, layout, chat component, weather-card example).
- AG-UI protocol: https://docs.ag-ui.com/introduction
- Reference starter: https://github.com/ag2ai/ag2-copilotkit-starter
- For protocol-level testing: AG2 Dojo profile https://dojo.ag-ui.com/ag2/feature/agentic_chat

## Common pitfalls

- **Missing `ag-ui` extra** — `pip install "ag2[ag-ui]"` is required; without it `from autogen.beta.ag_ui import AGUIStream` will fail.
- **CORS blocked** — frontend can't hit the backend in dev. Add `CORSMiddleware` to the FastAPI app for the dev origin.
- **No streaming output** — proxy or `Content-Type` issue. Test with raw `curl -N` first to isolate.
- **Tool UI not rendering** — tool name on the React side (`useCopilotAction({ name: "..." })`) doesn't exactly match the backend `@tool` name.
- **Putting the endpoint behind a SSE-incompatible proxy** — many proxies buffer event-stream responses by default. Disable buffering on the route.
- **Trying to use `AGUIStream` with classic `ConversableAgent`** — the description here covers `autogen.beta.Agent`. Classic AG2 agents can be exposed differently; check the AG2 docs for that path.
