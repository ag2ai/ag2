# Status bar middleware: executable examples

These eleven examples use AG2's public `statusbar_middleware`, History queries,
and dependency injection. Each file includes a complete agent, scripted model
responses, and assertions about what the model receives. All examples run
without provider credentials or network access.

`TestConfig` scripts model responses, and `TrackingConfig.calls` captures
complete, detached model-input snapshots. These checks demonstrate delivery of
state; they do not establish that a real model will follow the instructions.

The scenarios build on [Agent Status Bar in chapter 2 of AI Agents in
Depth](https://bojieli.github.io/ai-agent-book/book-en/chapter2/#agent-status-bar-keeping-the-model-aware-of-task-progress),
including experiments 2-8 and 2-9, with additional AG2 lifecycle cases.

| File | Scenario | Observable behavior |
| --- | --- | --- |
| [01_phone_call_limit.py](01_phone_call_limit.py) | Merchant call limit | Three scalar calls are counted across a continuation; only the current dynamic observation appears |
| [02_time_and_timeline.py](02_time_and_timeline.py) | Time and elapsed intervals | Injected clock, user-input timestamps, latest tool result, and a new day |
| [03_todo_plan.py](03_todo_plan.py) | TODO plan | Tools update conversation Variables; model snapshots preserve earlier plan values |
| [04_error_recovery.py](04_error_recovery.py) | Detailed failure and recovery | Scoped latest result, call/result correlation by ID, error count, and removal of a recovered warning |
| [05_environment.py](05_environment.py) | Workspace state | Injected per-conversation directory and current files; process CWD stays unchanged |
| [06_user_turn_counters.py](06_user_turn_counters.py) | Conversation and input counters | Accumulator and seen IDs, scoped counts, and fresh-conversation isolation |
| [07_external_multimodal.py](07_external_multimodal.py) | External dashboard | Async Depends/Inject/Variable composition and ordered text/data/image parts |
| [08_persistent_progress.py](08_persistent_progress.py) | Persistent progress | Retained snapshots, equal updates, Reply continuation, and a fresh isolated trail |
| [09_mixed_persistence.py](09_mixed_persistence.py) | Mixed modes | Persistent progress in chronological order and one current dynamic directory |
| [10_conditional_progress.py](10_conditional_progress.py) | Conditional updates | Empty list and None skip new insertions while previous records remain |
| [11_retry_snapshot.py](11_retry_snapshot.py) | Retry snapshot | Three provider attempts, two callback computations, and two persistent insertions |

Run a scenario from the checkout root, using an environment with AG2's test
dependencies installed:

```sh
PYTHONPATH=. python examples/status_middleware_usecases/01_phone_call_limit.py
```

The normal public test suite invokes each file's async `main()` entry point:

```sh
PYTHONPATH=. python -m pytest test/middleware/test_statusbar_usecases.py
```

## Callback and insertion contract

- Import `statusbar_middleware` from `ag2.middleware` and register the decorated
  callable with `middleware=[...]` on an Agent or invocation/Plugin.
- `@statusbar_middleware` and `@statusbar_middleware(persist=False)` compute
  dynamic input. `@statusbar_middleware(persist=True)` retains observations.
- Sync and async callbacks use normal Context, Inject, Variable, and Depends
  resolution. Return one Input, an ordered Sequence of Inputs, or None. Strings
  are not coerced into Input; invalid results fail before the provider call and
  before that callback's new observation is saved.
- Each logical model call evaluates the callback, including calls following
  tools. DI resources remain open through the wrapped call and its inner
  retries. Errors and cancellation propagate through the normal async scope.
- Dynamic results are appended to a separate request list. They are not
  published or stored, so the next call starts without previous dynamic
  observations. Static prompts and the actual transcript remain intact.
- Persistent nonempty results publish one retained service observation before
  the provider call. Earlier observations remain in chronological position;
  equal values create separate observations. Their parts are snapshots: later
  application mutations do not rewrite them. They are not user requests or
  conversational/usage records and do not trigger another model call.
- None and an empty sequence skip the new insertion. They leave previously
  retained observations available. Example 10 implements application-level
  suppression of unchanged values; the framework does not deduplicate them.
- Multiple status middleware run in registration order. Ordinary downstream
  middleware and reducers can still transform their inputs. Example 09 places
  retained progress before the dynamic environment.
- To reuse a snapshot on retry, place status outside RetryMiddleware:
  `middleware=[status, RetryMiddleware(...)]`. Reversing those wrappers follows
  normal middleware behavior and reevaluates a dynamic callback per attempt.
  A persistent status inside RetryMiddleware is unsupported: its published
  observation trips retry's publication guard, so the failed attempt re-raises
  instead of retrying. Once committed, a persistent observation remains if the
  later provider call fails.
- Keep conversation state in variables/dependencies. Dependencies deliberately
  shared by the application remain shared; AG2 does not isolate arbitrary
  external clients or roll back callback side effects.

## History selection and boundaries

```python
history = context.stream.history
calls = await history.where(ToolCallEvent.name == "phone_call").all()
last_error = await history.where(ToolErrorEvent).last()
current_calls = await history.since_last_user_input().where(ToolCallEvent).all()
```

Query construction is synchronous and immutable. Async all/first/last/count
terminals each read one finite retained snapshot in storage order. Empty results
are [], None, None, and zero, respectively. Queries never wait for future events;
storage errors propagate.

Queries select top-level stored occurrences, without expanding batches or child
streams and without implicit deduplication. Class filters include subclasses;
chained filters combine with AND. A filtered stream shares its parent's History.

The current-input interval starts after the latest retained actual user input,
excluding that input. Ask, Run, and Reply continuations establish boundaries.
Persistent observations and framework-enqueued messages do not reset the
interval, including both mid-turn and final inbox drains. ModelRequest origin
preserves user/service provenance through storage, independently of provider
wire roles; ordinary legacy requests default to user input. No manual turn ID
is required. If no actual user-input boundary remains retained, the interval is
empty.

## Limits

These examples keep their history. Counts cover retained events rather than
lifetime totals after compaction. There is no archive query, incremental cursor,
automatic delta/cadence policy, or snapshot refresh after compaction. The
application accumulator in example 06 keeps explicit seen call IDs. Restoring
arbitrary variables/dependencies after process restart is outside these examples.

Media follows ordinary provider capabilities and errors; the offline dashboard
example does not claim universal image support. Dynamic input changes do not
promise prompt-cache hits or preserved provider reasoning. Persisted records
have ordinary retained visibility; this API does not add one-request visibility
for physically saved records or configurable instruction roles.

The implementation acceptance checks exercise these examples against the real
public API, without reference adapters or private runtime patches. Additional
public middleware, provider-mapping, DI, retention, and stream tests cover the
framework behavior beyond the scripted examples.
