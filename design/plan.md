# Phased Plan: Decomposing `planet-satellite-network`

## Goal

Decompose a 45k-LOC merge of framework-core harness primitives, the Agent/Actor merge, and V3 network layer into small, reviewable PRs landed against `main`. Work happens on `planet-satellite-network` first; published PRs are cherry-picked from the green tip once everything passes.

**Branch scope (vs main):**
- ~9k LOC: framework-core harness promotion (knowledge, state, watch, observer, assembly, compact, aggregate, policies, alert/lifecycle events, scheduler)
- ~700 LOC: Agent/Actor merge (and the `memory` tool removal)
- ~27k LOC: V3 network layer (`autogen/beta/network/` — hub, sessions, tasks, rules, transforms, auth, HTTP/WS, clients, 45 test files)
- Misc: design docs, conflict resolution, test updates

---

## Strategy

1. **Land everything on `planet-satellite-network` first.** Resolve conflicts, do the Actor merge, make `test/beta` green.
2. **Cherry-pick from the green tip** into fresh branches off `main`. Prefer `git checkout <source> -- <paths>` file-checkout over commit cherry-pick for large additive chunks (network, harness primitives). Commit cherry-pick is fine for the targeted Actor-merge PR.
3. **Each published PR must pass tests standalone.** If a split puts test-X in PR-1 but the code-X it depends on in PR-2, shuffle the test accordingly.

---

## Decision log

| # | Decision | Taken | Rationale |
|---|---|---|---|
| 1 | **Actor is the final class name.** `Agent` is retired (no alias). | ✅ | User wants one agentic unit, not two. |
| 2 | **Keep both `run_subtask` and `as_tool` on Actor.** `run_subtask` stays auto-injected with the current default `TaskConfig(config=self.config, prompt=default)`. | ✅ | Different mechanisms: `as_tool` wraps *this* Actor; `run_subtask` spawns a throwaway child. Keep both working as-is for now. |
| 3 | **Observer resolution: fold `BaseObserver` onto main's `Observer(register(stack, ctx))` protocol.** Branch's `ObserverAlert` / `AlertPolicy` / `HaltEvent` / lifecycle events survive. | ✅ | Main's `@observer` decorator + `StreamObserver` is more ergonomic for the common case; branch's `BaseObserver` semantics (Watch + alerts) layer cleanly on top. One protocol, two shapes. |
| 4 | **Network V3 may split into 5a/5b/5c** if the combined PR is too large. Decide once the Actor merge lands. | ⏳ | Network is self-contained; defer split until everything else is settled. |
| 5 | **Do conflict-resolution + Actor merge + test fixes on this branch, then cherry-pick.** Do not create intermediate PR-0 base branch. | ✅ | Cleaner than rebasing multiple PRs against a moving base. Guarantees every downstream PR starts from known-good. |
| 6 | **`memory` tool removal.** v2_iteration_review.md already flagged this; confirmed in the Actor-merge PR scope. | ⏳ | Happens as part of Phase C. |
| 7 | **Emit `ObserverStarted`/`Completed` around `super()._execute`.** Agent's `ExitStack` handles actual register/unregister inside; Actor only emits lifecycle events. | ✅ | Forced by (3): Actor no longer manages its own observer stack. |

### Open decisions

- **#4 network split.** Suggested breakdown if we decide to split:
  - **5a** — primitives (ids, identity, rule, envelope, session_types, task, errors, events, auth, `KnowledgeStore` extensions)
  - **5b** — transport + hub core + adapters + in-process link
  - **5c** — HTTP/WS, clients, transforms, integration tests
- **Precise test shuffling for the split.** Some tests cross boundaries (e.g. `test_alert_halt.py` depends on both harness primitives and AlertPolicy). Audit before cherry-picking.

---

## Phase plan (on `planet-satellite-network`)

### ✅ Phase A — resolve main conflicts, get `test/beta` green

**Status: COMPLETE.** 1904 passed, 12 skipped, 1 xfailed. Nothing committed — changes in working tree for user review.

Delivered:
- 8 UU/AA merge conflicts resolved mechanically (`_typos.toml`, `pyproject.toml`, `test_convert_messages.py`, `gemini_client.py`, `events/__init__.py`, `stream.py`, `observer.py`, `__init__.py`).
- Observer refactor (originally planned for Phase B) folded into Phase A because the `observer.py` AA conflict couldn't be resolved without it. Branch's `BaseObserver` now implements `register(stack, ctx)` and relies on Agent's `ExitStack` for lifecycle.
- `Context → ConversationContext` rename propagated through 12 branch source files + 11 test files that directly imported the raw dataclass (main renamed the class in #2621; files importing `Context` from `.annotations` were untouched).
- `ModelRequest(content=str)` → `ModelRequest([TextInput(str)])` migration across 5 production sites and 99 test sites (main restructured `ModelRequest` from `content: str` to `inputs: list[Input]` in #2582).
- `.content` → `.inputs[0].content` assertion rewrites in ~15 tests.
- `Actor.__init__` now passes observers to `super().__init__(observers=...)` instead of maintaining a parallel `_observers` list. `Actor._execute` accepts `additional_observers` and forwards it to `Agent._execute`; emits `ObserverStarted`/`Completed` events around the super call but leaves actual register/unregister to Agent.
- `TokenMonitor.detach` / `LoopDetector.detach` stub overrides deleted.
- 3 framework test files (`test_observer.py`, `test_observers.py`, `test_edge_cases.py`) updated to `obs.register(ExitStack(), ctx)` in place of `obs.attach(stream, ctx)`.

Key files touched (working tree, un-staged):
- `autogen/beta/observer.py` — merged
- `autogen/beta/actor.py` — `__init__` + `_execute` rewrites
- `autogen/beta/stream.py`, `autogen/beta/__init__.py`, `autogen/beta/events/__init__.py`
- `autogen/beta/config/gemini/gemini_client.py`
- `autogen/beta/{watch,scheduler,assembly,compact,aggregate}.py` and `autogen/beta/policies/*.py` — Context rename
- `autogen/beta/network/policies/session_inbox.py` — Context rename + `ModelRequest.ensure_request`
- `autogen/beta/{compact,aggregate}.py` — `ModelRequest.ensure_request`
- `autogen/beta/observers/{token_monitor,loop_detector}.py` — `detach` override removed
- ~20 test files under `test/beta/` — Context rename, `ModelRequest([TextInput(...)])`, `register(stack, ctx)`, `.inputs[0].content` assertions
- `test/beta/config/anthropic/test_convert_messages.py` — main baseline + branch's `TestOrphanedToolResults` rewritten in new style
- `_typos.toml`, `pyproject.toml` — word lists unioned

### ⏳ Phase B — observer refactor

**Merged into Phase A.** No standalone work remains.

### 🔜 Phase C — Agent/Actor merge

**Goal:** make `Actor` the only agentic class. Retire `Agent`. Keep both `run_subtask` and `as_tool` (per decision #2).

Tasks:
1. **Move all `Agent.__init__` fields into `Actor.__init__`.** `Actor` already extends `Agent` — inline the full constructor. Rename the class to `Actor`. Keep backward-compat imports temporarily only if tests grep for `Agent` in a way we can't fix.
2. **Delete `autogen/beta/agent.py`** (or keep as a thin re-export shim if large amounts of test code still import from it — resolve on a case-by-case basis; prefer deletion).
3. **Move `AgentReply`, `Plugin`, `_execute_turn`, `_wrap_prompt_hook`** into `autogen/beta/actor.py`.
4. **Drop the `memory` tool** from Actor (per decision #6). Keep the `knowledge` CRUD tool.
5. **Gate Assembler/HaltCheck middleware injection** on `self._policies` being non-empty so a bare `Actor(name, config=cfg)` has zero harness middleware and matches current plain-Agent behavior.
6. **Update all imports across `autogen/beta/` and `test/beta/`** — `from autogen.beta.agent import Agent` → `from autogen.beta.actor import Actor`. The public `autogen.beta` re-export stays via `__init__.py`.
7. **Playground / demo / docs updates** — find any remaining `Agent(` call sites and migrate. The playground was previously flagged as needing an update.
8. **Run `test/beta` green.** Any test that imported `Agent` directly needs a rename pass.

Expected diff: ~700 LOC + wide test-file churn (mostly mechanical imports).

**Exit criteria:** `test/beta` green, `grep -r "from autogen.beta.agent" autogen/beta test/beta` returns nothing (or only `actor.py` itself re-exporting).

### 🔜 Phase D — verify

Run the full `test/beta` suite once more, fix any stragglers, ensure linting passes. This is the "cherry-pick source of truth" — the tip of `planet-satellite-network` becomes the reference for every downstream PR.

Also: run any integration smoke tests that require network/anthropic/openai credentials (skipped in the default suite). Not blocking but useful to know.

### 🔜 Phase E — cherry-pick into published PRs

Once Phase D is green, create fresh branches off `main` and populate each using `git checkout planet-satellite-network -- <paths>` or targeted commit cherry-picks. Order and contents:

| # | PR | Contents | Depends on |
|---|---|---|---|
| **1** | `feat(beta): framework-core harness primitives` | `knowledge.py`, `state.py`, `watch.py`, `observer.py` (merged shape), `events/alert.py`, `events/lifecycle.py`, `observers/{loop_detector,token_monitor}.py`, `watchdog` dep. Tests for each. Does **not** touch Agent/Actor. | main |
| **2** | `feat(beta): assembly + compact + aggregate + policies` | `assembly.py`, `compact.py`, `aggregate.py`, `policies/*` (Conversation/SlidingWindow/TokenBudget/EpisodicMemory/WorkingMemory/Alert). Still no Actor/Agent change. | PR 1 |
| **3** | `refactor(beta): merge Agent into Actor` | Move Actor fields into Agent's `__init__`, rename class to `Actor`, delete `actor.py` + old `agent.py`, kill the `memory` tool, gate harness middleware on non-empty `_policies`, switch observer lifecycle to Agent's `ExitStack`. Widespread import rewrites in `autogen/beta/` + `test/beta/`. | PR 2 |
| **4** | `feat(beta): scheduler` | `scheduler.py` (Scheduler + WatchStatus, standalone). Can land in parallel with PR 3. | PR 1 |
| **5** | `feat(beta): network v3` | Entire `autogen/beta/network/` tree + 45 test files. **Decision #4: may split into 5a/5b/5c** if review is unwieldy. | PR 3, PR 4 |

**Rationale for this order.** PRs 1 and 2 are pure additions to framework core — no Agent/Actor touch. PR 3 is where the rename and semantic change happen. PR 4 is independent. PR 5 is the network layer which depends on `Actor` being the single entry point.

**Cherry-pick method, per PR:**
- PR 1, 2, 4, 5 — additive chunks. Use `git checkout planet-satellite-network -- <paths>` onto a fresh branch off `main`. Single clean commit per PR.
- PR 3 — the Actor merge. Also file-checkout; after the merge, `agent.py` is gone and `actor.py` holds the merged class.

---

## Ripple effects captured during Phase A

These are not "decisions" but they shaped the resolution and are worth remembering:

1. **Main renamed `Context` → `ConversationContext`** (PR #2621) to avoid collision with the DI-annotated `Context` from `autogen.beta.annotations`. The DI-annotated `Context` (the public API surface for tools, middleware, observers) is unchanged and still exported from `autogen.beta` and `autogen.beta.annotations`. Only direct imports of the raw dataclass from `autogen.beta.context` had to migrate.
2. **Main restructured `ModelRequest`** (PR #2582) from `content: str` to `inputs: list[Input]`, introducing a full multimodal input hierarchy (`TextInput`, `ImageInput`, `DocumentInput`, `AudioInput`, `VideoInput`, `BinaryInput`, `FileIdInput`, URL variants, etc.). `ModelRequest.ensure_request(Iterable[str | Input])` is the idiomatic constructor. Production code uses `ensure_request`; tests use `ModelRequest([TextInput(...)])` for readability.
3. **Main added an `@observer(EventType)` decorator + `StreamObserver` dataclass + lightweight `Observer(register(stack, ctx))` protocol** (PR #2572) on `autogen/beta/observer.py`. Branch's `BaseObserver(name, watch)` is now a subclass of `BaseObserver(ABC)` that implements the same `register(stack, ctx)` protocol — arms the Watch in `register`, calls `stack.callback(self._disarm)` for teardown.
4. **Main added `observers=` param to `Agent.__init__` and `additional_observers=` to `Agent._execute`** (PR #2572), with registration happening inside Agent's own `ExitStack`. Actor now delegates observer registration entirely and only emits `ObserverStarted`/`Completed` lifecycle events around the super call.
5. **Main's `Agent.as_tool()`** (PR #2520) already shipped and is inherited by Actor. No collision with branch's `run_subtask` — they serve different purposes (delegate to sibling vs spawn throwaway child). Both are kept per decision #2.
6. **Main's `Plugin` system** (PR #2622) is already on the branch via an earlier merge. No conflict, but `Plugin` will need to be moved to `actor.py` during Phase C.

---

## Current state (as of this document)

- Branch: `planet-satellite-network`
- Merge source: `origin/main` at `869aadb2fb feat(beta): Search skills (#2615)` (present in working tree as `MERGE_HEAD`, 53 commits ahead of local `main` ref)
- Working tree: Phase A complete, ~40+ files modified, nothing staged or committed per user request
- Tests: `test/beta` → 1904 passed, 12 skipped, 1 xfailed, 0 failed
- `git ls-files --unmerged` → empty (all 8 conflicts resolved at git level)
- Next step waiting for user: review the Phase A diff, commit the merge, then start Phase C

## Quick resume checklist (for future sessions)

1. `git status` — confirm you're on `planet-satellite-network` mid-merge or post-merge.
2. Read this file, then `design/v2_iteration_review.md` and `design/network_v3_redesign.md` for the design context.
3. `uv run --all-extras pytest test/beta -q` or `.venv-beta/bin/python -m pytest test/beta -q` to confirm current test status.
4. Determine which phase is next from the phase plan above.
5. Check the decision log before making any architectural call — if a decision is listed, it's locked.
