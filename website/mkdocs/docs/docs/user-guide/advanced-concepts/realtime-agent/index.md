---
sidebarTitle: Overview
title: Overview
---

!!! warning "Deprecated"
    `RealtimeAgent` is deprecated as of v0.12 and will be removed in v0.14. It relies on deprecated realtime API endpoints. This documentation and associated notebooks will also be removed in v0.14.

As conversational AI evolves, real-time voice interactions are becoming essential in domains like customer support, healthcare, and virtual assistance. AG2 expands its capabilities by introducing `**`RealtimeAgent`**`, a powerful agent which enables seamless real-time communication through multiple adapters.

## How RealtimeAgent Works

`**`RealtimeAgent`**` acts as the central component for handling voice-based interactions. It integrates with various adapters, allowing developers to choose the best communication channel for their applications. The agent supports:

- **Real-Time Voice Interactions**: Enables natural and immediate conversations with users.
- **Swarm Integration**: Delegates complex tasks to AG2 Swarm teams during live interactions.
- **Developer-Friendly APIs**: Provides straightforward integration for building real-time AI-driven experiences.

## Available Communication Methods for RealtimeAgent

### 1. WebSocket Audio Adapter
For web-based applications, `**`WebSocketAudioAdapter`**` eliminates the need for external telephony services, offering:

- **Direct Browser Integration**: Streams real-time audio via WebSockets.
- **Faster Setup**: Avoids Twilio’s complex configuration and account verification.
- **Reduced Latency & Cost**: Bypasses telephony platforms, minimizing dependency on external services.

🔗 [Explore the Walk-through](./websocket.md)

### 2. WebRTC Integration
For high-quality, peer-to-peer audio streaming, [**`WebRTC`**](https://webrtc.org/) integration provides:

- **Low-Latency Communication**: Direct peer-to-peer audio streaming ensures real-time responsiveness.
- **Adaptive Audio Quality**: Dynamically adjusts based on network conditions.
- **Secure & Widely Supported**: Uses WebRTC encryption and works across all modern browsers.

🔗 [Explore the Walk-through](./webrtc.md)

### 3. Twilio Audio Adapter
AG2’s `**`TwilioAudioAdapter`**` connects `**`RealtimeAgent`**` to Twilio’s telephony services, allowing seamless voice call handling with:

- **Telephony Support**: Direct integration with phone networks via Twilio.
- **Scalability**: Supports enterprise-level deployments for customer service applications.
- **Swarm Team Collaboration**: Allows delegation of tasks to specialized AI agents during live calls.

🔗 [Explore the Walk-through](./twilio.md)
