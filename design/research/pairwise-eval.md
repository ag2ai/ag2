# Pairwise Evaluation (eval item 5)

Status: **design locked** (2026-05-25), building. Compares two agent variants
(A vs B) on the same tasks and reports a win-rate — the partner's
"highest-leverage lever" for matching real user preference. Trace-based:
grades pre-produced traces, never runs the agent itself.

## Locked decisions
- **Two sources** — `evaluate_pairwise(source_a, source_b)` pairs by `task_id`.
  Variant = which source; pairing = `ag2.eval.task_id` (stamped at produce time
  via `TelemetryMiddleware(span_attributes=…)` — existing mechanism, #2876).
- **Distinct `pairwise_judge()` + a `PairwiseComparator` protocol** — LLM judge,
  human, and BYO comparators are interchangeable behind the protocol. No `Judge`
  class hierarchy/registry (the protocol *is* the swappable seam; same ethos as
  `Scorer` for single-trace).
- **Human evaluation: both modes** — offline export/import (scales, UI-agnostic)
  + inline HITL (`context.input`, dev/small-N). Plus judge-vs-human agreement.
- **Two entry points** — `evaluate_pairwise` (decoupled foundation) and
  `run_pairwise` (sugar: produce both variants + compare, stamping the keys).
- **Deferred:** Bradley-Terry/Elo for >2 variants; length-controlled win-rate.

## Industry-standard methodology (what makes it standard)
- Win / Loss / Tie per comparison (not absolute scores).
- **Position-bias mitigation** (LLM judges have 60–75% positional bias): the LLM
  comparator runs **dual-order swap** and only declares a win if the same answer
  wins in *both* orders; a flip → tie. Humans get a **single blinded randomized**
  presentation (asking twice is wasteful/biasing), order recorded.
- **Win-rate + Wilson confidence interval** (ties = 0.5); small-N friendly.
- **Self-preference**: judge should be a *different model family* than the
  variants (GPT-4 ≈ +10%, Claude ≈ +25% self-bias) — documented/surfaced.
- Sources: position-bias study arXiv:2406.07791; MT-Bench/Chatbot Arena; AlpacaEval.

## Contracts

```python
@runtime_checkable
class PairwiseComparator(Protocol):
    key: str
    async def compare(self, *, task, trace_a, trace_b, reference_outputs) -> PairwiseOutcome: ...

@dataclass(frozen=True, slots=True)
class PairwiseOutcome:
    winner: Literal["a", "b", "tie"]
    reasoning: str | None = None
    detail: dict | None = None      # e.g. both-order verdicts for audit

class PairwiseVerdict(BaseModel):    # one judge call, position-based + OpenAI-strict-safe
    preferred: Literal["first", "second", "tie"]
    reasoning: str
```

Each comparator encapsulates its own position strategy inside `compare`:
`pairwise_judge` does the dual-order swap; `human_labels` de-blinds a label;
inline-HITL prompts. `evaluate_pairwise` just calls `compare()` and tallies.

## Comparators
- **`pairwise_judge(config, *, criterion, key, include_trace=False, swap=True)`** —
  Agent-as-judge over two responses, single criterion, dual-order swap →
  `PairwiseOutcome`. Reuses the `agent_judge` machinery (compose Agent + schema).
- **`human_labels(path, *, key)`** — reads an imported label file; de-blinds the
  recorded order → `PairwiseOutcome`.
- **inline HITL comparator** — prompts via `context.input()` during the run.
- **`export_pairwise_cases(source_a, source_b, suite, criteria, out)`** — writes a
  blinded labeling manifest (Response 1/2 randomized + recorded) for humans.

## Entry points
```python
async def evaluate_pairwise(source_a, source_b, *, comparators, variant_a="A",
    variant_b="B", suite=None, store_dir, concurrency=4, run_id=None) -> PairwiseRunResult
async def run_pairwise(suite, *, variant_a, variant_b, comparators, store_dir, ...) -> PairwiseRunResult
```
`run_pairwise` = produce A's traces → produce B's traces (stamping
`ag2.eval.task_id`/`variant`) → `evaluate_pairwise`.

## Result
`PairwiseRunResult` per comparator key: A_wins / B_wins / ties, **win-rate(B)** =
(B_wins + 0.5·ties)/n with **Wilson CI**, position-flip→tie count, per-case
records (both-order verdicts). `result.agreement(key_x, key_y)` → **Cohen's κ** +
agreement-rate + disagreement list (judge-vs-human calibration; engagement item 9).

```
Pairwise  "v2" (B) vs "v1" (A) · 12 cases · swap: on
  criterion          B  A  tie  win-rate(v2)   95% CI
  correctness        7  3   2      66.7%     [41%,86%]
agreement(correctness@judge, correctness@human): 83%  κ 0.66  (n=12)
```

## Build sequence
1. `PairwiseComparator` / `PairwiseOutcome` / `PairwiseVerdict` + `pairwise_judge` (swap). ← keystone
2. `PairwiseRunResult` + win-rate / Wilson CI / Cohen's κ + `summary()` + persistence.
3. `evaluate_pairwise` (pair by task_id, run comparators, tally).
4. Human: `export_pairwise_cases` + `human_labels` + inline HITL.
5. `run_pairwise` (produce both, stamp keys, then evaluate).
6. Tests: mock (incl. a position-flip→tie case + agreement), then live vs Tempo (two variants).
