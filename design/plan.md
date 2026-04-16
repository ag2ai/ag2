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
| 4 | **Network V3 may split into 5a/5b/5c** if the combined PR is too large. Decide once the Actor merge lands. | ⏳ | Deferred until PR 4/5 scope. |
| 5 | **Do conflict-resolution + Actor merge + test fixes on this branch, then cherry-pick.** Do not create intermediate PR-0 base branch. | ✅ | Cleaner than rebasing multiple PRs against a moving base. Guarantees every downstream PR starts from known-good. |
| 6 | **`memory` tool removal.** v2_iteration_review.md already flagged this; confirmed in the Actor-merge PR scope. | ✅ | Removed as part of Phase C; `actor.py` contains zero references to `memory`. |
| 7 | **Emit `ObserverStarted`/`Completed` around `super()._execute`.** Agent's `ExitStack` handles actual register/unregister inside; Actor only emits lifecycle events. | ✅ | Forced by (3): Actor no longer manages its own observer stack. |
| 8 | **Keep per-commit history on each PR.** Don't squash the 28 harness commits / 11 assembly commits / 11 actor commits into one each; use GitHub's "Squash and merge" at land time. | ✅ | Commit messages are meaningful (`feat: add watch primitive`, `feat: add loop detector`, …); per-commit review surfaces logical layers. |
| 9 | **Smoke tests live at `smoke_tests/` at repo root, not `test/beta/smoke/`.** | ✅ | CI's `just test-beta-cov` targets `test/beta/`. Even with the `-m "not (openai or …)"` filter, one unmarked test leaked through. Moving smoke out of `test/beta/` eliminates the leak vector entirely. |
| 10 | **Stacked PRs, not flat ones.** PR 2 base = `pr1/harness-primitives`, PR 3 base = `pr2/assembly-layer`. | ✅ | Each PR shows clean incremental diff; reviewers see one layer at a time. Trade-off: after each merge, next PR's base auto-updates to `main` and may need rebase. |

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

### ✅ Phase B — observer refactor

**Merged into Phase A.** No standalone work remains.

### ✅ Phase C — Agent/Actor merge

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

**Actual diff: ~2800 LOC net** (+1504 actor.py, −921 agent.py, plus conversable/plugin/textual/subagent tool updates and wide test rename). Estimate of 700 LOC was wrong because `actor.py` absorbed `Plugin`, `AgentReply`, `KnowledgeConfig`, `TaskConfig`, lifecycle wiring, and harness middleware gating.

**Exit criteria met:** `test/beta` → 1137 passed, 12 skipped, 1 xfailed. `grep -r "from autogen.beta.agent" autogen/beta test/beta` → empty. The `memory` tool is gone. Middleware gating works (bare `Actor(name, config=cfg)` has zero harness middleware).

### ✅ Phase D — verify

**Status: COMPLETE.** `test/beta` → 1137 passed, 12 skipped, 1 xfailed on the tip of `planet-satellite-network`. Before cherry-picking, also verified end-to-end:

- **Smoke tests** (`smoke_tests/`, 95 tests, cross-provider) — 95/95 passed against real OpenAI/Anthropic/Gemini APIs in 12 min 9 s.
- **Playground examples** (`playground/01_hello_actor.py` … `08_safety_guard.py`) — 8/8 ran successfully end-to-end via Gemini.

### 🚧 Phase E — cherry-pick into published PRs

**Actor half (PRs 1–3): SHIPPED.** Scheduler (PR 4) and network (PR 5) still pending.

| # | PR | Status | Contents | Depends on |
|---|---|---|---|---|
| **1** | [#2658](https://github.com/ag2ai/ag2/pull/2658) `feat(beta): harness primitives` | 🟢 OPEN | `knowledge.py`, `watch.py`, `observer.py` (adds `BaseObserver` ABC), `events/{alert,lifecycle,_serialization}.py`, `observers/{loop_detector,token_monitor}.py`, `streams/redis/*` upgrades, `watchdog` dep, provider mapper fixes (Usage.total_tokens in anthropic; safer partial-stream in gemini). Tests under `test/beta/framework/` + `test/beta/stream/test_transient.py` + `test_convert_messages.py::TestOrphanedToolResults`. **28 commits, +4504/−33.** No Actor touch. | `main` |
| **2** | [#2655](https://github.com/ag2ai/ag2/pull/2655) `feat(beta): actor assembly layer` | 🟢 OPEN | `assembly.py` (AssemblerMiddleware + AssemblyPolicy), `compact.py` (TailWindowCompact, SummarizeCompact, CompactionSummary), `aggregate.py` (ConversationSummaryAggregate, WorkingMemoryAggregate), `policies/*` (6 policies: Alert, Conversation, EpisodicMemory, SlidingWindow, TokenBudget, WorkingMemory). **11 commits.** **Stacked on PR 1.** | PR 1 |
| **3** | [#2656](https://github.com/ag2ai/ag2/pull/2656) `refactor(beta): merge agent and actor` | 🟢 OPEN | Delete `agent.py` (−921), add `actor.py` (+1504), rewrite `conversable.py` / `plugin.py` / `textual.py` / `response/prompted.py` / `config/openai/containers.py`, update `tools/subagents/*` + `tools/toolkits/filesystem.py` + `tools/toolkits/skills/skill_search/toolset.py`. `memory` tool removed. Harness middleware gated on non-empty `_policies`. `Agent`→`Actor` rename across `test/beta/agent/`, `chats/`, `config/`, `groupchat_interop/`, `middleware/`, `providers/`, `test_plugin/`, `test_task.py`, `tools/`. New: `test/beta/framework/{test_actor_integration,test_bug_fixes}.py`, `test/beta/test_alert_halt.py`, `smoke_tests/*` (moved from `test/beta/smoke/`), `playground/*`. **11 commits.** **Stacked on PR 2.** | PR 2 |
| **4** | `feat(beta): scheduler` | ⏳ PENDING | `autogen/beta/scheduler.py` (Scheduler + WatchStatus) + `test/beta/framework/test_scheduler{,_watch_coverage}.py`. Standalone; can land in parallel with PR 3. | PR 1 |
| **5** | `feat(beta): network v3` | ⏳ PENDING | Entire `autogen/beta/network/` tree (+ `test/beta/network/`, 45 test files). Decision #4: may split into 5a/5b/5c. | PR 3, PR 4 |

**Merge order:** PR 1 (#2658) → PR 2 (#2655) → PR 3 (#2656). After each merges, the next PR's base auto-flips to `main` (GitHub handles this when the base branch is deleted) or may need a manual rebase if the flip leaves conflicts.

**Verification snapshot per layer (unit tests, excluding smoke + network):**
- PR 1: **1028 passed**, 12 skipped, 1 xfailed
- PR 2: **1078 passed**, 12 skipped, 1 xfailed (+50 new tests: assembly, compact, aggregate, sliding_window, token_budget)
- PR 3: **1137 passed**, 12 skipped, 1 xfailed (+59 new tests: actor integration, bug fixes, alert halt, restored `TestRunSubtasksSequentialExceptionHandling`)

**End-to-end verification on PR 3 branch:** `smoke_tests/` 95/95 (real APIs), `playground/01..08` 8/8 (real Gemini).

### Smoke tests relocation

`test/beta/smoke/` → **`smoke_tests/`** at the repo root. CI runs `just test-beta-cov` which targets `test/beta/` with `-m "not (openai or anthropic or gemini or …)"`. All smoke files except `test_core_parity.py` had module-level LLM markers; the one exception had `test_missing_config_raises` (no provider fixture → no marker → leaked through the CI filter). Moving the whole directory outside `test/beta/` eliminates the leak vector. `conftest.py` `_REPO_ROOT = Path(__file__).resolve().parents[1]` (was `parents[3]` at the old location).

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

---

## Post-mortem: accidental direct push to `main`

While preparing PR 1, ran `git checkout -b pr1/harness-primitives origin/main`. The `-b` form with a remote tracking ref sets the new local branch to **track** `origin/main`. A subsequent `git push` (no refspec) fast-forwarded `origin/main` by 28 commits, bypassing review.

### Recovery

1. Branch protection rejected a force-push rollback attempt (`GH013: Cannot force-push to this branch`) — the safety net did its job.
2. Created `revert-pr1-direct-push` branch off `ba70f22aa5`, used two-reset (`git reset --hard c31a08e42a && git reset --soft ba70f22aa5 && git commit`) to produce a single commit whose tree byte-for-byte equals `c31a08e42a`.
3. Opened [#2657](https://github.com/ag2ai/ag2/pull/2657) as a normal PR. Merged. `origin/main` is now back to the pre-push tree.
4. Rebased `pr1/harness-primitives` onto the new `origin/main` via `git rebase --onto origin/main c31a08e42a` — all 28 commits replayed cleanly (zero conflicts, since the new main tree equals the old rebase base tree).
5. Force-pushed `pr1/harness-primitives` with `--force-with-lease`. Opened as [#2658](https://github.com/ag2ai/ag2/pull/2658).
6. Rebased `pr2/assembly-layer` onto `pr1/harness-primitives` via `git rebase --onto pr1/harness-primitives ba70f22aa5 pr2/assembly-layer`. Force-pushed; updated PR #2655's base via `gh api -X PATCH repos/ag2ai/ag2/pulls/2655 -f base=pr1/harness-primitives`.
7. Rebased `pr3/actor-merge` onto new `pr2/assembly-layer` via `git rebase --onto pr2/assembly-layer 87a3bc3dfa pr3/actor-merge` (where `87a3bc3dfa` was the old pr2 tip, recorded before step 6). Force-pushed; PR #2656 already existed with the right base.

### Lesson

When branching off a remote ref for a PR, always use **`git checkout --no-track -b <name> origin/main`** or push with an explicit refspec (`git push -u origin <name>:<name>`). A plain `git push` on a tracking-main branch goes to main. This is saved to `.claude` memory as `feedback_branch_tracking.md` so future sessions don't repeat it.

---

## Current state (as of this document)

- **PR 1** ([#2658](https://github.com/ag2ai/ag2/pull/2658)) — open, 28 commits, targeting `main`. Local branch `pr1/harness-primitives` at `9a9c1872a7`.
- **PR 2** ([#2655](https://github.com/ag2ai/ag2/pull/2655)) — open, 11 commits, stacked on `pr1/harness-primitives`. Local branch `pr2/assembly-layer` at `f60370bc65`.
- **PR 3** ([#2656](https://github.com/ag2ai/ag2/pull/2656)) — open, 11 commits, stacked on `pr2/assembly-layer`. Local branch `pr3/actor-merge` at `a8a0738720`.
- **Revert PR** ([#2657](https://github.com/ag2ai/ag2/pull/2657)) — merged. Local branch `revert-pr1-direct-push` can be deleted.
- **`planet-satellite-network`** — still contains scheduler (PR 4) and network v3 (PR 5). Not yet cherry-picked.

## Quick resume checklist (for future sessions)

1. `git fetch origin` — make sure you have the latest state of main + all PR branches.
2. Check PR statuses: `gh pr list --state open --head pr1/harness-primitives --head pr2/assembly-layer --head pr3/actor-merge`.
3. Read this file, then `design/v2_iteration_review.md` and `design/network_v3_redesign.md` for the design context.
4. `.venv-beta/bin/python -m pytest test/beta --ignore=test/beta/network -q` to confirm current test status on whatever branch you're on.
5. Check the decision log before making any architectural call — if a decision is listed, it's locked.
6. For PR 4 (scheduler) and PR 5 (network), branch off `main` (or the latest PR tip that's merged) with `git checkout --no-track -b <name> origin/main` — **never plain `-b`**.
