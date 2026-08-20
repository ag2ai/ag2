---
sidebarTitle: Channels
title: "Channels: AG2 Agents in Slack and Other Messaging Platforms"
description: "Run the same AG2 agent that powers your in-app copilot as a bot in Slack and other messaging platforms with the CopilotKit Channels SDK."
---

The same AG2 agent that powers your in-app copilot can also run as a bot in messaging platforms. [CopilotKit Channels](https://docs.copilotkit.ai/slack/ag2){.external-link target="_blank"} connects your AG2 agent to Slack and other messaging platforms, with threads, tool calls, and rich interactive messages handled natively in the channel.

!!! note
    This page covers the AG2 side: how a channel process wires your `AGUIStream` endpoint into a platform thread. Creating the Channel and connecting Slack happen in the CopilotKit dashboard — follow [Configure the Channel in Intelligence](https://docs.copilotkit.ai/slack/ag2/intelligence){.external-link target="_blank"} first, since the code below needs the credentials it gives you.

## How it fits together

Your AG2 server stays where it is, serving the agent through `AGUIStream` exactly as in the [basic server example](/docs/user-guide/ag-ui/index#basic-server-example). A channel is a separate long-running Node process built with [`@copilotkit/channels`](https://www.npmjs.com/package/@copilotkit/channels){.external-link target="_blank"}, connected through [CopilotKit Intelligence](https://docs.copilotkit.ai/slack/ag2/intelligence){.external-link target="_blank"}. Intelligence holds the platform connection and credentials, receives each platform event, and delivers the turn over a persistent gateway connection to your channel process, which runs your agent over AG-UI and streams the reply back into the platform thread.

```text
Slack  ──►  CopilotKit Intelligence  ──►  channel process (Node)  ──►  AG2 server (AGUIStream)
```

The credential split is the point, and it is what makes this safe when your AG2 agent holds its own model keys and tools:

- **You keep** the agent logic, model credentials, tools, and the channel process.
- **CopilotKit Intelligence holds** the platform credentials, message delivery, registration, health, and reconnects.

A web frontend built with the [CopilotKit UI quickstart](/docs/user-guide/ag-ui/copilotkit-quickstart) and the bot are just two clients of one AG-UI endpoint.

## Build the channel process

Install the Channels SDK, the CopilotKit runtime that hosts the channel, and the AG-UI client:

```bash
npm install @copilotkit/channels @copilotkit/runtime @ag-ui/client tsx
```

Build the agent as a per-thread factory so each conversation gets its own session, using `HttpAgent` from [`@ag-ui/client`](https://www.npmjs.com/package/@ag-ui/client){.external-link target="_blank"} pointed at your endpoint — `AGUIStream` speaks plain AG-UI, so no AG2-specific client is needed. The channel's `name` is the **Code** of the Channel you created in Intelligence, whose dashboard also holds the Slack credentials — they never enter this process.

Name the file with an `.mts` extension: the Channels SDK ships ESM only, and `.mts` makes this an ES module without touching `package.json`.

```ts title="channel.mts" linenums="1" hl_lines="13 16-20"
import { createChannel } from "@copilotkit/channels";
import { CopilotRuntime, CopilotKitIntelligence } from "@copilotkit/runtime/v2";
import { createCopilotNodeListener } from "@copilotkit/runtime/v2/node";
import { HttpAgent } from "@ag-ui/client";

const required = (name: string) => {
  const value = process.env[name];
  if (!value) throw new Error(`Missing ${name}`);
  return value;
};

const channel = createChannel({
  name: required("CHANNEL_CODE"), // the Channel's Code from Intelligence
  identifyUser: "platform",
  // A fresh agent per conversation, pointed at your AG-UI endpoint.
  agent: (threadId) => {
    const agent = new HttpAgent({ url: "http://localhost:8000/chat" });
    agent.threadId = threadId;
    return agent;
  },
});

// A mention subscribes the thread and runs the agent; afterwards every message
// in a subscribed thread runs it without needing another mention.
channel.onMention(async ({ thread }) => {
  await thread.subscribe();
  await thread.runAgent();
});
channel.onMessage(async ({ thread }) => {
  if (await thread.isSubscribed()) await thread.runAgent();
});

// The runtime owns the channel's lifecycle: creating the listener starts it.
const runtime = new CopilotRuntime({
  agents: {}, // the channel supplies its own agent; no web-facing agents needed
  intelligence: new CopilotKitIntelligence({
    apiKey: required("INTELLIGENCE_API_KEY"),
  }),
  channels: [channel],
});

const listener = createCopilotNodeListener({ runtime });
await listener.channels.ready({ timeoutMs: 15_000 }); // fail startup loudly on a broken config
```

Run the channel process alongside the AG-UI server from the [basic server example](/docs/user-guide/ag-ui/index#basic-server-example). Both credentials come from the Intelligence dashboard — the API key under **API Keys**, the Code on the Channel you created there:

```bash
uvicorn run_ag_ui:app --reload --port 8000   # terminal 1 — AG2 server
```

```bash
export INTELLIGENCE_API_KEY="cpk-..."        # terminal 2 — channel process
export CHANNEL_CODE="support-bot"            # the Channel's Code — not a Slack channel ID
npx tsx channel.mts
```

Until the channel process is running, the Channel reads **Waiting for runtime** in the dashboard: nothing has declared it yet. That is the state `ready()` waits to leave.

## Slack

Invite the app to a Slack channel, then mention it. It runs your agent and streams the reply back into the thread; the thread stays subscribed, so follow-up messages run without another mention.

The agent receives an ordinary AG-UI `RunAgentInput` and emits ordinary AG-UI events — the platform mechanics stay behind the channel, so the same AG2 agent runs unchanged across every platform. Rich messages are written as JSX and rendered to each platform's native format (Block Kit on Slack, for example), so an interactive card degrades gracefully where a platform has no equivalent.

!!! warning "The channel process needs a long-running host"
    It holds a persistent connection to the Intelligence gateway, which delivers each turn to it. A serverless request handler cannot own that connection.

The Slack app itself is created by pasting the manifest from the Intelligence wizard into Slack's web UI and handing its tokens back to Intelligence — no Slack CLI and no bot framework, since the agent already runs in your AG2 server. For that walkthrough, see [Configure the Channel in Intelligence](https://docs.copilotkit.ai/slack/ag2/intelligence){.external-link target="_blank"}; for the full channel-process one, see [Connect your agent](https://docs.copilotkit.ai/slack/ag2/connect){.external-link target="_blank"}.

## Other platforms

Discord, Teams, Telegram, and WhatsApp connect the same way — a managed connection configured in Intelligence, with your channel code unchanged. See the [CopilotKit Channels documentation](https://docs.copilotkit.ai/slack/ag2){.external-link target="_blank"} for the current platform list and per-platform setup.

!!! note
    By default, interactive actions live in memory and reset on restart. Back the channel with a durable action and state store (Redis or Postgres) so buttons and per-thread state survive restarts and span multiple instances.

## Next steps

1. Build the AG-UI endpoint from the [basic server example](/docs/user-guide/ag-ui/index#basic-server-example) if you have not already.
2. Create the Channel in CopilotKit Intelligence and connect Slack there, then export the API key and the Channel's Code.
3. Start the channel process, invite the app to a Slack channel, and mention it.
4. For a web UI on the same endpoint, follow the [CopilotKit UI quickstart](/docs/user-guide/ag-ui/copilotkit-quickstart).
