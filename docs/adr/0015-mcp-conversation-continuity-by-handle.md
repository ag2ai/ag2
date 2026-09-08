---
status: accepted
date: 2026-08-27
---

# 15. MCP conversation continuity is named by a server-minted handle

## Context

`ag2.mcp.MCPServer` exposes an agent as an MCP server whose conversational tool
keeps multi-turn history. Until now that history was keyed by the transport's
`mcp-session-id`, falling back to a per-process sentinel over stdio.

MCP protocol revision 2026-07-28 (the *modern era*) removed the session: each
request is a self-contained POST, no `initialize` handshake, no session id. It
also states that connections do not represent conversations, that clients may
interleave unrelated requests on one transport, and that servers must not use
connection or process identity to establish context.

So the old keying gave a modern-era client one of two wrong behaviours, silently:
over HTTP every call started empty, and over stdio every call on the process
shared one history — the second being a direct non-conformance.

## Decision

Continuity stops riding on the transport and becomes something the caller names.

1. **A handle names the conversation.** The conversational tool gains an optional
   `conversation` argument. Omitted — or presented blank, which names nothing and
   which is what a model tends to send for an optional string — the call starts
   fresh and the server mints an opaque version-4 UUID; presented, the
   conversation continues. Handles are
   minted by the server and **never** adopted from the caller — with a bounded
   LRU registry, a caller-chosen key would let anyone evict other callers'
   conversations.
2. **Each era keeps the mechanism its own revision sanctions.** With no
   conversation named, the handshake era still keys on its MCP session (the
   per-process sentinel over stdio); the modern era, which has neither, starts
   fresh. The process fallback is withdrawn from the modern era only.
3. **The handle travels back twice, and never in `structuredContent`.** In a text
   content block, because the protocol puts recovery from an expired handle on
   the model and the model does not read protocol metadata; and in the result's
   `_meta` under `ai.ag2/conversation`, for programmatic clients. Not in
   `structuredContent`: on this tool that field is the agent's response schema,
   advertised verbatim as `outputSchema`, which MCP requires structured content
   to conform to — a server field mixed in would break the tool's own contract.
4. **An unknown or expired handle is a tool execution error**, not a JSON-RPC
   one. The protocol draws that line so the model can start a new conversation
   rather than fail the turn. It is never a fall-through to the transport
   session, which would reintroduce the silent degradation this exists to remove.
   With conversations off altogether there is no registry to consult and no
   handle this server could have minted, so a presented one is refused as
   *unsupported* rather than accepted and dropped — the same reason, a different
   remedy: retrying without the argument would not restore continuity there, so
   the caller is told continuity is unavailable instead of inferring it from a
   reply that quietly forgot.
5. **A conversation records its principal and revalidates on every call.** The
   access token's subject, falling back to its client id. A mismatch yields the
   same error as an unknown handle, so it does not disclose that the handle
   exists. Session-named conversations need no check of ours: the transport
   already refuses a session id presented with a different credential.

## Consequences

- A modern-era client gets multi-turn behaviour it previously could not have, on
  every transport, and a stdio client's unrelated requests stay unrelated.
- **Handshake-era results change shape**: every reply now carries the handle
  block and `_meta`, whether or not the caller ever names a conversation. This is
  a deliberate resolution of a tension in the spec — one user story asks that the
  handshake era be untouched, another that a handle come back on the first call
  in both eras. The second won: a handshake-era client that wants explicit
  handles can then migrate ahead of changing protocol revision.
- Any call that names no conversation and has no MCP session to fall back on
  occupies an LRU slot — every modern-era call, and every handshake-era call on a
  `stateless=True` transport. Operators serving one-shot traffic should size
  `max_sessions` for the call rate and set a `ttl`.
- With no authentication configured there is no principal to bind to, and the
  handle is the sole credential for the conversation it names — and it travels
  through readable content.
- `stateless=True` with `sessions=True` becomes a coherent configuration ("no
  transport session, conversations by handle") rather than a contradiction, so no
  construction-time validation was added.

## Alternatives considered

- **Mirroring the handle into an HTTP header** (the `x-mcp-header` extension).
  Optional for servers by the protocol's own wording, and there is no sticky
  deployment to serve. Adding it later is additive; removing it after a
  deployment depends on it is not.
- **Retiring transport-keyed continuity entirely.** Handshake-era revisions have
  a protocol session and keying on it is correct there; dropping it would remove
  working multi-turn behaviour from every current client.
- **Rejecting `stateless=True` with `sessions=True` at construction.** The
  pairing was contradictory only while continuity depended on the transport
  issuing a session id.
