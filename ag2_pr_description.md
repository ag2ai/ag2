## Summary

Addresses the speaker-selection gap identified by @marklysze in #2430 (comment):
when an agent fails (e.g. `cheap_planner` returning `None`), GroupChat has no
mechanism to exclude it from future rounds.

- Add `AgentEligibilityPolicy` Protocol to `GroupChat`, enabling runtime filtering
  of speaker candidates based on agent health or business logic.
- `SelectionContext` dataclass provides minimal, decoupled context to policies.
- `AgentDescriptionGuard` prepends `[UNAVAILABLE]` to an agent's description to steer LLM-based auto-selection away from unavailable agents.

## Changes

- `autogen/agentchat/eligibility_policy.py` (new): `SelectionContext` frozen dataclass,
  `AgentEligibilityPolicy` Protocol (`@runtime_checkable`), `AgentDescriptionGuard`.
- `autogen/agentchat/groupchat.py`: `eligibility_policies: list[AgentEligibilityPolicy]`
  field (default `[]`) + `_apply_eligibility_policies()` called inside
  `_prepare_and_select_agents`.
- `autogen/agentchat/__init__.py`: export `AgentEligibilityPolicy`, `SelectionContext`, `AgentDescriptionGuard`.
- `autogen/__init__.py`: re-export same symbols so `from autogen import AgentEligibilityPolicy` works.
- `test/agentchat/test_eligibility_policy.py` (new): unit tests for Protocol and AgentDescriptionGuard.
- `test/agentchat/test_groupchat_eligibility.py` (new): integration tests incl.
  @marklysze's `cheap_planner` scenario.
- `notebook/agentchat_groupchat_eligibility.ipynb` (new): CB integration demo.

## Design Decisions

- **Protocol, not ABC**: `AgentEligibilityPolicy` uses `typing.Protocol` for structural
  subtyping -- no inheritance required, zero coupling for callers.
- **AND semantics**: Multiple policies all-must-pass. A single `return False` removes the agent from candidates regardless of other policies.
- **`SelectionContext` is minimal**: Only `round`, `last_speaker` (name, not object),
  and `participants` (names). The `GroupChat` object itself is intentionally excluded.
- **No breaking changes**: `eligibility_policies=[]` is the default. All existing
  `GroupChat` usages are unaffected.
- **Callable bypass**: When `speaker_selection_method` is a Callable that returns an `Agent` directly, policies are not applied -- the caller has explicit control and overrides filtering.

## Out of Scope

- `None` reply semantics change (what happens when an agent reply is `None`) is a separate issue, pending @marklysze input on the failure-handling design.
- Integration with Team/Task orchestration (#2401) is designed as a follow-up.
  `AgentEligibilityPolicy` is intentionally decoupled so it can work with both
  current `GroupChat` and the upcoming orchestration patterns.

## Test Plan

- [x] `pytest test/agentchat/test_eligibility_policy.py` -- Protocol + AgentDescriptionGuard unit tests
- [x] `pytest test/agentchat/test_groupchat_eligibility.py` -- integration tests (msze scenario)
- [x] `pytest test/agentchat/test_groupchat.py` -- no new failures introduced (pre-existing `test_custom_speaker_selection` failure is unrelated to this PR)
