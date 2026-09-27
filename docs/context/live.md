# Context: Live

Glossary for realtime (speech-to-speech) agents: a long-lived session with a provider
that listens, speaks, and accepts user turns while it runs. Glossary only — no
implementation detail.

## Language

**Live session**:
The span during which a `LiveAgent` holds an open conversation with a realtime provider.
Unlike a turn of an **Ask** or **Run**, it has no single request/reply; it lasts until the
caller leaves it and hears and answers the user many times in between.
_Avoid_: live turn, realtime run

**Response**:
One uninterrupted answer the provider produces inside a live session — speech, text, tool
calls — ending when the provider declares it done. A live session is a sequence of responses.

**Response boundary**:
The moment a response ends. A response that was asked for while the model was busy starts
here, so the model is never interrupted mid-answer.

**Pushed input**:
A user turn handed to a running live session by anyone other than the provider — a human
typing next to the voice channel, or a program — through the **Inbox** or straight onto the
stream. It enters the provider's conversation at once; the provider answers it at the next
response boundary, or immediately if the model is idle. Several pushes waiting on one boundary get one
answer.
_Avoid_: injected message, side input

**Voice turn**:
A user turn the provider itself produced from captured audio. It is published on the stream
for observers and history, but is never pushed back into the provider.
