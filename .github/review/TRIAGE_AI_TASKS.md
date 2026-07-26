# AI Triage Tasks

Recurring triage work that automation owns. Every task follows the same shape:

- **Query** — which issues/PRs to pick up (a search filter, so the task is re-runnable at any time);
- **Criteria** — the checks to evaluate, **in order**;
- **Decision** — labels to set and which [canned reply](replies/) to post.

The rules of the game are defined by the [Triage Policy](TRIAGE_POLICY.md) — this file only operationalizes it.

## Global guardrails

These override anything below:

- **Never**: set `priority:*`, set `status:confirmed` (except the maintainer shortcut in T1), close items (except the stale flow in T6), merge, or edit someone's code.
- **One check — one message.** Criteria are evaluated in the listed order; the first failing check decides the outcome, and the task **stops there** — no piling several demands into one pass. Gate precedence: **CLA > template > conflicts**.
- **Idempotency**: before acting, check current labels and comments — never repeat a comment that is already present and still applicable.
- **Sync rule**: any change to a `type:*` label or an Issue Type updates the other side (mapping in [Policy §1](TRIAGE_POLICY.md#type--classification)).
- **Invariants**: exactly one `type:*`; exactly one `status:*` (plus `status:stale` stacked on a gate); permission matrix in [Policy §8](TRIAGE_POLICY.md#8-who-may-do-what).
- When in doubt — label nothing, comment with the observation, and mention the maintainers.

---

## T1. Issue intake

**Query:** open issues with no `status:*` label.

**Criteria, in order — first hit decides:**

| # | Check | Decision |
|---|---|---|
| 1 | Looks like a security vulnerability | add `security` + `status:needs-triage`, post [`security-report`](replies/security-report.md), notify maintainers; **stop** — no further public analysis |
| 2 | Usage question, not a work item | set `type:question` (no Issue Type), post [`redirect-question`](replies/redirect-question.md) suggesting a Discussion, add `status:needs-triage` — do not close |
| 3 | Duplicate of an existing open/recently-closed item | add `resolution:duplicate` as a **proposal** + `status:needs-triage`, post [`duplicate-suspect`](replies/duplicate-suspect.md) with `{{original}}` — **never close**; the author, Triage Team, or a maintainer closes if they agree |
| 4 | Description doesn't follow the template — sections missing/placeholder; **bug reports must include a minimal reproducible example (MRE)**, required by the bug form | set `status:needs-template`, post [`needs-template-issue`](replies/needs-template-issue.md) with `{{missing_sections}}` |
| 5 | All checks passed | author is a maintainer / Triage Team member → `status:confirmed` (shortcut, no comment); otherwise `status:needs-triage` |

**Non-blocking enrichment** (same pass, no extra comments): sync Issue Type ↔ `type:*` if misfiled; infer `area:*` from the text.

## T2. PR intake

**Query:** open PRs with no `status:*` label.

**Criteria, in order — first failing check decides, later checks are not evaluated:**

| # | Check | Decision |
|---|---|---|
| 1 | `license/cla` check not passing (drafts included) | set `status:needs-cla`, post [`needs-cla`](replies/needs-cla.md) |
| 2 | Description doesn't follow [the PR template](../PULL_REQUEST_TEMPLATE.md) — no real "why", doesn't reflect the diff, no validation info ([AI policy](../AI_POLICY.md)) | set `status:needs-template`, post [`needs-template-pr`](replies/needs-template-pr.md); if the cause is unverified AI-generated content, post [`ai-slop`](replies/ai-slop.md) with `{{observations}}` instead |
| 3 | Merge conflicts with `main` — **skip this check for drafts** | set `status:changes-requested`, post [`merge-conflict`](replies/merge-conflict.md) |
| 4 | All checks passed | draft → `status:in-progress`; ready → `status:needs-review`. No comment — except a first-time contributor, who gets [`welcome-first-pr`](replies/welcome-first-pr.md) |

**Non-blocking enrichment:** verify `type:*` matches the diff (`fix:` → `type:bug`, docs-only → `type:docs`); add `area:*` the path-based labeler can't infer.

## T3. Return from `needs-cla` / `needs-template`

**Query:** open issues and PRs with `status:needs-cla` or `status:needs-template` where the author acted **after** the gate was set (signed the CLA, edited the description, pushed).

**Criteria:** re-run the specific check behind the gate (`license/cla` state; template completeness incl. MRE for bugs).

**Decision:**

- Check now passes → remove the gate label and `status:stale` if present, then **re-run the intake cascade** (T1/T2 criteria) — the next gate in precedence order may apply (CLA signed, but the description is still gutted), or the item lands in `status:needs-triage` / `status:in-progress` / `status:needs-review`.
- Check still fails → one reply stating exactly what is still missing (only if not already said); the gate stays.

## T4. Return from `awaiting-response`

**Query:** issues with `status:awaiting-response` where the author replied after the question was asked.

**Decision:** remove the gate and `status:stale` if present → `status:needs-triage` (back into the Triage Team queue). No reply from the author → do nothing; the clock is T5's job.

## T5. Mark gated items stale

**Query:** open issues **and** PRs with a gate label (`needs-template` / `needs-cla` / `awaiting-response` / `changes-requested`), without `status:stale`, where the last author activity is older than the gate's window ([Policy §5](TRIAGE_POLICY.md#5-gates-and-the-stale-mechanism)):

| Gate | Window |
|---|---|
| `status:needs-template` | 7 days |
| `status:needs-cla` | 3 days |
| `status:awaiting-response` | 10 days |
| `status:changes-requested` | 30 days |

**Decision:** add `status:stale` **on top of** the gate (the gate stays — it is the visible reason), post [`stale-warning`](replies/stale-warning.md) with `{{gate_label}}`.

## T6. Resolve stale items: close or unmark

**Query:** open issues and PRs with `status:stale`.

**Criteria, in order:**

| # | Check | Decision |
|---|---|---|
| 1 | Author acted after the stale mark | remove `status:stale`; the gate stays — its fate is T3/T4's job |
| 2 | 7+ days of silence since the stale mark | close as *not planned*, post [`stale-close`](replies/stale-close.md); all labels remain for auditability |

## T7. Conflict sweep of working PRs

**Query:** open PRs with `status:needs-review` or `status:in-progress`, **excluding drafts**; run on every push to `main` and daily. GitHub computes mergeability lazily — re-query while the state is `UNKNOWN`.

**Criteria:** `mergeable == CONFLICTING`.

**Decision:** replace the status with `status:changes-requested`, post [`merge-conflict`](replies/merge-conflict.md). Gated PRs are **not** in the query — their gate has precedence; the conflict resurfaces via T3's re-run of the intake cascade.

## T8. Return from `changes-requested`

**Query:** PRs with `status:changes-requested` where, after the trigger that set the label, the author pushed commits or answered all review threads — or, when the label came from a merge conflict, the conflict is resolved.

**Decision:** replace with `status:needs-review`, no comment — the PR reappears in the reviewers' queue.

---

## Stays manual

By design these remain with humans (see [Policy §8](TRIAGE_POLICY.md#8-who-may-do-what)):

- **Closing** anything outside the stale flow — including confirmed duplicates (T1 only proposes).
- `status:confirmed` and `priority:*` — Triage Team / maintainers, during triage.
- `status:needs-review` → `status:changes-requested` on review — the reviewer sets it together with the review itself.
- Merging (Policy §4 P4 conditions).

## Future automation (not scheduled)

- Extension routing: ping the named maintainer from the module docstring on `area:extensions` items ([`extension-maintainer-ping`](replies/extension-maintainer-ping.md)); archival watchlist.
- Weekly consistency audit: label invariants, Issue Type ↔ label drift, `confirmed` without priority.
- Weekly queue-health digest for maintainers.
