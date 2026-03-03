## Summary

- Add `AgentEligibilityPolicy` Protocol to `GroupChat`, enabling runtime filtering
  of speaker candidates based on agent health or business logic.
- `SelectionContext` dataclass provides minimal, decoupled context to policies.
- `DescriptionMutationMixin` provides soft-signal support for LLM-based selection.

## Changes

- `autogen/agentchat/eligibility_policy.py` (new): `SelectionContext` frozen dataclass,
  `AgentEligibilityPolicy` Protocol (`@runtime_checkable`), `DescriptionMutationMixin`.
- `autogen/agentchat/groupchat.py`: `eligibility_policies: list[AgentEligibilityPolicy]`
  field (default `[]`) + `_apply_eligibility_policies()` called inside
  `_prepare_and_select_agents`.
- `autogen/agentchat/__init__.py`: export `AgentEligibilityPolicy`, `SelectionContext`.
- `test/agentchat/test_eligibility_policy.py` (new): unit tests for Protocol and mixin.
- `test/agentchat/test_groupchat_eligibility.py` (new): integration tests incl.
  @marklysze's `cheap_planner` scenario.
- `notebook/agentchat_groupchat_eligibility.ipynb` (new): CB integration demo.

## Design Decisions

- **Protocol, not ABC**: `AgentEligibilityPolicy` uses `typing.Protocol` for structural
  subtyping -- no inheritance required, zero coupling for callers.
- **AND semantics**: Multiple policies all-must-pass, consistent with
  `safeguard_llm_inputs` hook list behavior.
- **`SelectionContext` is minimal**: Only `round`, `last_speaker` (name, not object),
  and `participants` (names). The `GroupChat` object itself is intentionally excluded.
- **No breaking changes**: `eligibility_policies=[]` is the default. All existing
  `GroupChat` usages are unaffected.
- **veronica-core is optional**: AG2 core has zero dependency on it. CB integration
  is shown in the example notebook only.

## Out of Scope

`None` reply semantics change is a separate issue, pending @marklysze input on
`on_agent_failure` parameter design.

## Test Plan

- [x] `pytest test/agentchat/test_eligibility_policy.py` -- Protocol + mixin unit tests
- [x] `pytest test/agentchat/test_groupchat_eligibility.py` -- integration tests (msze scenario)
- [x] `pytest test/agentchat/test_groupchat.py` -- existing tests unaffected
