---
title: "Building low-latency voice agents in 3 lines of code with GPT Realtime 2"
authors: [Lancetnik]

date: 2026-05-13
slug: voice-agents-with-gpt-realtime-2
tags:
    - AG2 Beta
    - LiveAgent
    - Voice
    - Realtime
    - OpenAI
    - Gemini
categories:
    - AG2 Beta
    - LiveAgent
    - Voice
    - Realtime
    - OpenAI
    - Gemini
date: 2026-05-12
slug: LiveAgent
---

![LiveAgent on AG2 Beta](img/live-agent-hero.png)

OpenAI recently announced [advanced voice intelligence in the API](https://openai.com/index/advancing-voice-intelligence-with-new-models-in-the-api/){.external-link target="_blank"}, including **GPT Realtime 2** for lower-latency, more natural spoken interactions. **AG2 Beta** wraps that class of model behind **`LiveAgent`**: one bidirectional session, continuous audio in and out, and provider-side voice activity detection so users can speak and interrupt like on a phone call—not like a walkie-talkie app.

In this post we walk through why that matters, how **`LiveAgent`** compares to the classic **STT → Agent → TTS** stack, and how to add **tools** and **subagent-style** delegation without giving up the realtime voice surface.

<!-- more -->

## Why voice agents matter

Voice is often the lowest-friction channel for **hands-busy** or **eyes-busy** tasks: driving, cooking, field work, accessibility, and quick “just tell me” moments on a phone. Users expect **low latency**, **natural turn-taking**, and the ability to **cut in** when the model goes wrong or when new information arrives. If every utterance waits for a full record → transcribe → reason → synthesize cycle, it feels like filling out a form, not a conversation.

Provider **realtime** APIs push much of that rhythm into the model and transport layer: streaming audio, built-in **VAD** (Voice Activity Detection), and **barge-in** semantics (where supported) so the application is not re-implementing half of a telephony stack in user space.

## Basic code sample

Once you've installed...

```bash
pip install "ag2[openai] sounddevice[numpy]"
```

..., import **`LiveAgent`**, **`SoundDevicePlayer`**, **`SoundDeviceRecorder`**, and the **`openai`** config helpers from **`autogen.beta.live`**. The **running** session is then just **three lines**: construct the agent, open the session and shared I/O on one **`async with`**, then block until you cancel.

```python linenums="1" hl_lines="1-3"
agent = LiveAgent("assistant", "You are a helpful voice assistant.", config=openai.RealTimeConfig("gpt-realtime-2", output=openai.AudioOutput(voice="ballad")))
async with agent.run() as context, SoundDevicePlayer(context=context), SoundDeviceRecorder(context=context):
    await asyncio.Future()  # run until cancelled
```

That is the whole “wire microphone and speaker to GPT Realtime 2” loop. A self-contained script (imports and `asyncio.run`) looks like this:

```python linenums="1" hl_lines="12-18 22-29"
"""The same 3 lines with formatting and imports"""
import asyncio

from autogen.beta.live import (
    LiveAgent,
    SoundDevicePlayer,
    SoundDeviceRecorder,
    openai,
)

# line 1
agent = LiveAgent(
    name="assistant",
    prompt="You are a helpful voice assistant.",
    config=openai.RealTimeConfig(
        "gpt-realtime-2",
        output=openai.AudioOutput(voice="ballad", speed=1.2),
    ),
)

# line 2
async def main() -> None:
    async with (
        agent.run() as context,
        SoundDevicePlayer(context=context),
        SoundDeviceRecorder(context=context),
    ):
        # line 3
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
```

Beyond the defaults, you can tune how the assistant **sounds** and **when it speaks**: OpenAI realtime **`AudioOutput`** accepts **`voice`** and **`speed`**, and **`InputConfig`** carries **VAD / turn-detection** options (for example semantic VAD with interruption). See the [Providers (OpenAI) documentation](../../../beta/live/live_agent.md#providers){target="_blank"} for voices and configs.

## How it works with tools and subagents

### Tools

`LiveAgent` uses the same **`@agent.tool`** decorator as a text **`Agent`**. Calls go through AG2’s normal tool executor; results are sent back into the realtime session automatically—so middleware and HITL patterns you use elsewhere can stay aligned with the rest of your stack.

```python linenums="1" hl_lines="9-12"
from autogen.beta.live import LiveAgent, OpenAIRealTimeConfig, SoundDevicePlayer, SoundDeviceRecorder

agent = LiveAgent(
    name="assistant",
    prompt="You are a helpful voice assistant.",
    config=OpenAIRealTimeConfig("gpt-realtime-2"),
)

@agent.tool
async def sum_numbers(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b

# async with agent.run() as context, SoundDevicePlayer(...), SoundDeviceRecorder(...): ...
```

More detail: [Tools in a realtime session](../../../beta/live/live_agent.md#tools-in-a-realtime-session){target="_blank"} and [Tools](../../../beta/tools/tools.md){target="_blank"}.

### Subagents (delegation)

**`LiveAgent`** is built around a **realtime audio session**, not the same **`ask()`** loop as a text **`Agent`**. When the voice surface needs **deeper reasoning**, **longer context work**, or **tooling you prefer to keep on the text side**, expose a separate **`Agent`** as a callable tool with **`Agent.as_tool()`** and pass it into **`LiveAgent`** via **`tools=[...]`**. The realtime model issues a tool call; AG2 runs that nested **`Agent`** and sends the result back into the live session.

```python linenums="1" hl_lines="11-13 19"
from autogen.beta import Agent
from autogen.beta.config import OpenAIConfig
from autogen.beta.live import LiveAgent, OpenAIRealTimeConfig, SoundDevicePlayer, SoundDeviceRecorder

researcher = Agent(
    name="researcher",
    prompt="You research topics thoroughly and return concise bullet facts.",
    config=OpenAIConfig("gpt-4o-mini"),
)

research_tool = researcher.as_tool(
    description="Use for multi-step research or when the user needs cited-style facts in text form.",
)

voice = LiveAgent(
    name="voice",
    prompt="You are a low-latency voice assistant. Call research when the user needs deep fact-finding.",
    config=OpenAIRealTimeConfig("gpt-realtime-2"),
    tools=[research_tool],
)

# async with voice.run() as context, SoundDevicePlayer(...), SoundDeviceRecorder(...): ...
```

That gives you **subagent-style delegation** without pretending the realtime session is a full **`Agent.ask`** loop. See [LiveAgent vs Agent](../../../beta/live/live_agent.md#liveagent-vs-agent){target="_blank"}.

## Supported providers

`LiveAgent` accepts any **`RealtimeConfig`**. AG2 Beta ships **OpenAI** and **Gemini** implementations today.

- **OpenAI** — **`gpt-realtime-2`** with **`AudioOutput`** for **`voice`** and **`speed`**, or **`TextOutput`** for text-only replies; **`InputConfig`** controls **VAD and turn detection** (defaults lean toward semantic VAD with interruption).
- **Gemini** — e.g. **`gemini-3.1-flash-live-preview`** with live audio.

OpenAI and Gemini voices and config snippets: [Providers](../../../beta/live/live_agent.md#providers){target="_blank"}.

Broader install and I/O notes: [Voice & Realtime overview](../../../beta/live/live.md){target="_blank"}.

## LiveAgent or STT–TTS

AG2 Beta supports **two** voice patterns:

| | **STT → Agent → TTS** | **`LiveAgent` (realtime)** |
| --- | --- | --- |
| Shape | Discrete turns: transcribe, call a text **`Agent`**, synthesize | One **full-duplex** session for the whole conversation |
| Typical latency | on the order of **1–3 s** per turn | **sub-500 ms** in many setups |
| Turn detection | Your app decides when a “turn” ends | **Provider-driven** (e.g. semantic VAD, interruption) |
| Best when… | You need every **`Agent`** feature each turn (structured output, rich middleware, arbitrary model routing) | You need **phone-call-like** experience: continuous audio, interruption, minimal glue code |

The two are complementary, not competing. Reach for **STT–TTS** when the text agent must be the source of truth for each step. Reach for **`LiveAgent`** when **latency and conversational flow** matter most and you are happy to drive the session through a **realtime** model such as **`gpt-realtime-2`**.

## What's next

- **Read the docs** — start with [LiveAgent — Realtime Voice Sessions](../../../beta/live/live_agent.md){target="_blank"} and the [Voice & Realtime overview](../../../beta/live/live.md){target="_blank"}.
- **Install and try** — `pip install "ag2[openai] sounddevice[numpy]"`, then run the sketches under **`examples/live_playground/`** in the [AG2 repo](https://github.com/ag2ai/ag2/tree/main/examples/live_playground){.external-link target="_blank"}.
- **Star the repo** — if **`LiveAgent`** is useful, a star on [github.com/ag2ai/ag2](https://github.com/ag2ai/ag2){.external-link target="_blank"} helps others find the project.
- **Join the community** — share builds, ask questions, and compare notes on the [AG2 Discord](https://discord.gg/pAbnFJrkgZ){.external-link target="_blank"}.

We are excited to get your feedback on LiveAgent.
