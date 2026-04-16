---
sidebarTitle: CopilotKit Quickstart
title: "CopilotKit UI Quickstart for AG-UI"
description: "Connect a React/Next.js CopilotKit UI to an AG2 AG-UI runtime endpoint."
---

This quickstart shows how to connect a CopilotKit React/Next.js UI to an AG2 backend endpoint that speaks the AG-UI protocol.

## What you'll build

- A Python backend that serves an AG-UI endpoint for an AG2 agent
- A React/Next.js UI powered by CopilotKit

## Prerequisites

- Python 3.10+
- Node.js 18.18+
- An LLM API key for your AG2 agent (for example `OPENAI_API_KEY`)

## Quickstart

You can either bootstrap a template project, or follow the same structure as the AG2 + CopilotKit starter.

<Tabs>
  <Tab title="Bootstrap with CopilotKit">

CopilotKit can bootstrap a template project:

```sh
npx copilotkit@latest create -f ag2
```

After initialization, run the backend and UI that were generated for you.

  </Tab>

  <Tab title="Clone the reference starter">

Runnable reference implementation: [AG2 + CopilotKit starter](https://github.com/ag2ai/ag2-copilotkit-starter){.external-link target="_blank"}.

The updated template structure used by the starter looks like:

```text
ag2-copilotkit-starter/
├── agent-py/     # Python backend (AG2 agent + AG-UI endpoint)
├── ui-react/     # React + CopilotKit frontend
```

  </Tab>
</Tabs>

## 1) Start the AG-UI backend

The starter backend mounts the AG-UI endpoint at `/chat` and runs on port `8008`.

```sh
cd agent-py
pip install -r requirements.txt
export OPENAI_API_KEY="your_openai_api_key"
python backend.py
```

Your AG-UI endpoint will be available at `http://localhost:8008/chat`.

## 2) Start the React + CopilotKit UI

In a new terminal:

```sh
cd ui-react
npm install
npm run dev
```

Then open `http://localhost:3000`.

## 3) Connect CopilotKit runtime to the AG-UI endpoint

CopilotKit uses a Next.js route (typically `/api/copilotkit`) that bridges the UI to your agent runtime. In the template, that route registers an AG-UI HTTP agent client with `CopilotRuntime`.

```tsx title="ui-react/app/api/copilotkit/route.ts"
import { HttpAgent } from "@ag-ui/client";
import {
  CopilotRuntime,
  ExperimentalEmptyAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
} from "@copilotkit/runtime";
import { NextRequest } from "next/server";

const agent = new HttpAgent({ url: "http://localhost:8008/chat" });

const runtime = new CopilotRuntime({
  agents: {
    weather_agent: agent,
  },
});

export async function POST(req: NextRequest) {
  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime,
    serviceAdapter: new ExperimentalEmptyAdapter(),
    endpoint: "/api/copilotkit",
  });
  return handleRequest(req);
}
```

## 4) Add the `CopilotKit` provider

Wrap your app with `<CopilotKit>` and point it at the runtime route.

```tsx title="ui-react/app/layout.tsx"
import { CopilotKit } from "@copilotkit/react-core";
import "@copilotkit/react-ui/styles.css";
import "./globals.css";

export const metadata = {
  title: "AG2 Weather Agent",
  description: "Weather agent powered by AG2 and CopilotKit",
};

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

If your backend enforces CORS/auth, configure those before starting the UI (see Production notes).

## 5) Render a chat UI

```tsx title="ui-react/app/page.tsx"
"use client";

import { CopilotChat } from "@copilotkit/react-ui";
import { useCopilotAction } from "@copilotkit/react-core";

function WeatherCard({
  location,
  temperature,
  feelsLike,
  humidity,
  windSpeed,
  windGust,
  conditions,
  isLoading,
}: {
  location?: string;
  temperature?: number;
  feelsLike?: number;
  humidity?: number;
  windSpeed?: number;
  windGust?: number;
  conditions?: string;
  isLoading: boolean;
}) {
  const tempF = temperature != null ? (temperature * 9 / 5 + 32).toFixed(1) : null;

  return (
    <div
      className={`rounded-lg border border-sky-300 bg-gradient-to-br from-sky-100 to-blue-100 p-4 max-w-xs shadow-md ${isLoading ? "animate-pulse" : ""}`}
    >
      <h3 className="text-lg font-semibold text-sky-700">
        {location || "Loading..."}
      </h3>
      <p className="text-xs text-sky-500 uppercase tracking-wide mb-3">
        {isLoading ? "Fetching weather..." : "Current Weather"}
      </p>

      <div className="flex items-start justify-between mb-3">
        <div>
          <div className="text-4xl font-bold text-gray-800">
            {temperature != null ? temperature : "--"}
            <span className="text-lg text-sky-600">&deg;C</span>
          </div>
          {tempF && <div className="text-sm text-gray-500">{tempF}&deg;F</div>}
        </div>
        <div className="text-sm text-gray-500 text-right max-w-[120px]">
          {conditions || "--"}
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2 pt-3 border-t border-sky-200">
        <div className="text-center">
          <div className="text-[10px] text-gray-500 uppercase">Humidity</div>
          <div className="text-sm font-mono text-gray-800">
            {humidity != null ? `${humidity}%` : "--%"}
          </div>
        </div>
        <div className="text-center">
          <div className="text-[10px] text-gray-500 uppercase">Wind</div>
          <div className="text-sm font-mono text-gray-800">
            {windSpeed != null ? `${windSpeed} km/h` : "-- km/h"}
          </div>
        </div>
        <div className="text-center">
          <div className="text-[10px] text-gray-500 uppercase">Feels Like</div>
          <div className="text-sm font-mono text-gray-800">
            {feelsLike != null ? `${feelsLike}\u00B0` : "--\u00B0"}
          </div>
        </div>
      </div>
    </div>
  );
}

export default function Home() {
  useCopilotAction({
    name: "get_weather",
    description: "Get the weather for a given location.",
    available: "disabled",
    parameters: [{ name: "location", type: "string", required: true }],
    render: ({ args, status, result }) => {
      if (status === "complete" && result) {
        let data = result;
        if (typeof result === "string") {
          try {
            data = JSON.parse(result.replace(/'/g, '"'));
          } catch {
            return <div>{result}</div>;
          }
        }
        return (
          <WeatherCard
            location={data.location}
            temperature={data.temperature}
            feelsLike={data.feelsLike}
            humidity={data.humidity}
            windSpeed={data.windSpeed}
            windGust={data.windGust}
            conditions={data.conditions}
            isLoading={false}
          />
        );
      }
      return <WeatherCard location={args.location} isLoading={true} />;
    },
  });

  return (
    <div className="flex items-center justify-center min-h-screen bg-gradient-to-b from-sky-100 to-blue-200">
      <div className="w-full max-w-2xl h-[80vh] rounded-xl overflow-hidden shadow-2xl border border-sky-300">
        <CopilotChat
          labels={{
            title: "AG2 Weather Agent",
            initial: "Hi! Ask me about the weather in any city.",
            placeholder: "Ask about the weather...",
          }}
          className="h-full"
        />
      </div>
    </div>
  );
}
```

## Expected output

After setup:

- The Next.js page shows a CopilotKit chat UI
- Messages stream from the AG2 runtime via the AG-UI endpoint
- Tool/action calls can be rendered as custom UI (for example, a weather card in the starter)

## Production notes

- CORS: allow your frontend origin on the AG-UI backend (`POST`, `OPTIONS`, auth headers).
- Auth: protect both `/api/copilotkit` and backend `/chat` (token/header/cookie); do not rely on client-only secrets.
- Deployment topology: keep frontend runtime route and AG-UI backend on trusted internal network paths where possible.
- Timeouts/retries: configure conservative client and server timeouts for long-running tool workflows and retry only idempotent requests.

## Troubleshooting

- CORS errors (`blocked by CORS policy`): check backend `Access-Control-Allow-Origin`, `Access-Control-Allow-Headers`, and preflight handling.
- No streaming output: verify backend response `Content-Type` is `text/event-stream` and that proxy layers do not buffer SSE.
- Tool UI not rendering: ensure tool/action names match exactly (for example `get_weather`).

## Security considerations

Treat tool inputs and shared state as untrusted user-controlled data. Validate and authorize server-side before invoking privileged tools, and log tool execution with request identifiers for auditability.

## Version and compatibility notes

- For a complete working example (backend + React UI + HTML UI), see the starter repo referenced above.
