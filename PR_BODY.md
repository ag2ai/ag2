## Summary

Agents that execute tools need pre-LLM-send and post-tool-output checks against known attack signatures (prompt injection, secret exfiltration, jailbreak markers). `ConversableAgent` already exposes the right hookable methods (`safeguard_tool_outputs`, `safeguard_llm_inputs`) but ag2 ships no rule pack.

This PR adds `autogen/agentchat/contrib/capabilities/atr_guardrail.py`, a composable `AgentCapability` that subscribes to those two hooks and scans content against the ATR (Agent Threat Rules) open detection ruleset. It lives under `contrib/capabilities/`, the documented landing zone for community extensions, and has no impact on the default install.

## What is ATR

ATR is an MIT-licensed open detection standard for AI agent threats: 419 rules covering OWASP Agentic Top 10 and most of SAFE-MCP. Rules are in production at Microsoft Agent Governance Toolkit, Cisco AI Defense skill-scanner, MISP galaxies, and the OWASP Agentic Top 10 mapping. Repo: https://github.com/Agent-Threat-Rule/agent-threat-rules.

## Design

- `pyatr` is wrapped in `optional_import_block()`; without it the capability is a deterministic no-op rather than raising at import time.
- `add_to_agent(agent)` follows the `AgentCapability` contract used by `Teachability` and registers two hooks: `safeguard_tool_outputs` and `safeguard_llm_inputs`.
- Three action modes: `allow` (record only), `warn` (log and record), `block` (drop the LLM input in line with the existing `_process_llm_input` contract; redact matched tool output content).
- Optional `min_severity` threshold and `on_match` callback for downstream telemetry.
- Tests stub the rule loader, so the suite needs no network and no `pyatr` install.

## Why now

ag2 accepted three security PRs in the last release cycle (CVE-2026-23745, CVE-2026-23950, CVE-2026-24842) plus the SSRF protection in #2784. A pluggable signature layer complements those fixes and the existing `SafeguardEnforcer` in `autogen/agentchat/group/safeguards.py`, which handles policy but ships no rule pack.

## Test plan

- [x] Ruff lint and format clean.
- [x] 13 unit tests pass locally.
- [ ] CI runs `pytest test/agentchat/contrib/capabilities/test_atr_guardrail.py`.
- [ ] Reviewer confirms no-op behavior without `pyatr` installed.
