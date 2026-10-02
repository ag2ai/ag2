---
status: accepted
date: 2026-09-15
---

# 0017. A served agent derives its DNS rebinding protection, and refuses tokens only when asked

## Context

`MCPServer` serves an agent over streamable HTTP. Two of the transport's
protections are off unless somebody turns them on, and the `mcp` SDK reads an
absent value for each as an instruction to stay off, for backwards
compatibility:

- **DNS rebinding protection.** `TransportSecuritySettings=None` disables it.
  A *default-constructed* `TransportSecuritySettings()` is not the safe middle
  ground either: it enables the protection with empty allow-lists and so refuses
  every request. Turning it on needs the host names the server answers on, and
  an ASGI app mounted behind a proxy cannot see them.
- **The RFC 8707 resource-indicator check.** `mcp` 2.2.0 added
  `resource_server_url=` to its bearer backend: with it set, only a token whose
  `AccessToken.resource` names this server is accepted. The indicator is
  optional on the token an operator's own verifier builds, and an absent value
  fails the check. The SDK makes this check its default in 3.0.

Shipping a served agent with neither is a real exposure. Shipping both on breaks
deployments already in the field — `ag2.mcp.security` shipped in 1.0.4, and
`MCPServer` before it.

## Decision

**DNS rebinding protection is derived, not guessed, and not left off.** When an
authorization requirement is configured, `TransportConfig.security_settings_for`
builds the settings from `security.resource_url` — the endpoint the operator
*declared* this server answers on, so it is a statement rather than a guess.
Any port is allowed when the URL names none, so a proxy forwarding `Host` with
one still reaches the server. An explicit `security_settings` always wins,
including an explicit refusal:
`TransportSecuritySettings(enable_dns_rebinding_protection=False)`. With no
authorization requirement there is no declared URL, nothing to derive from, and
the protection stays off.

**The resource-indicator check stays off unless asked for.**
`require(..., validate_token_resource=True)` turns it on. It is named after the
SDK's own setting so that documentation transfers.

The asymmetry is deliberate: deriving rebinding protection uses a value the
operator already supplied and can be observed to be wrong immediately (a `421`
on the first request), while the resource check depends on a claim the
operator's verifier may simply not populate, and failing it answers `401` to
every request with no local signal about why.

## Consequences

- A deployment that configures `security=` and answers on a `Host` its
  `resource_url` does not name now receives `421` where it previously served.
  That is a breaking change, and the reason the opt-out is spelled out in the
  guide rather than left to be discovered.
- `security_settings=None` no longer means "no protection". The off switch is
  `enable_dns_rebinding_protection=False`, which also frees `None` from carrying
  two meanings.
- Tokens minted for another service are still accepted by default. An operator
  who wants them refused must say so, and until `mcp` 3.0 makes it the default,
  the switch is where the decision lives.
- `TransportSecuritySettings` and `EventStore` are re-exported from `ag2.mcp`
  beside `RequestStateSecurity`, so configuring either protection needs no
  import from inside the SDK.
