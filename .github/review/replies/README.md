# Canned replies

Template responses for the situations defined in the [Triage Policy](../TRIAGE_POLICY.md). Used by the AI Triage Bot verbatim (with placeholders substituted) and by maintainers / the Triage Team as copy-paste starting points — personalize when it helps.

Conventions:

- Placeholders use `{{name}}` syntax. Common ones: `{{author}}` (GitHub login without `@`), `{{missing_sections}}`, `{{original}}` (issue/PR reference), `{{reason}}`, `{{maintainer}}`, `{{extension}}`.
- The HTML comment at the top of each file documents when to use it and which label/action accompanies the message. It is not part of the reply.
- Gate messages embed their own stale timeline (per-gate windows from [Triage Policy §5](../TRIAGE_POLICY.md#5-gates-and-the-stale-mechanism)); the `+status:stale` flip posts `stale-warning.md`.

| File | Situation | Accompanying action |
|---|---|---|
| `needs-template-issue.md` | Issue doesn't follow the template | `status:needs-template` |
| `needs-template-pr.md` | PR description doesn't follow the template | `status:needs-template` |
| `needs-cla.md` | CLA not signed (drafts included) | `status:needs-cla` |
| `merge-conflict.md` | PR conflicts with `main` | `status:changes-requested` |
| `stale-warning.md` | Gated item passed its grace period | add `status:stale` on top of the gate |
| `stale-close.md` | Closing a gated item after the stale window | close as *not planned* |
| `needs-reproduction.md` | Bug cannot be reproduced / info missing | `status:awaiting-response` |
| `redirect-question.md` | Usage question filed as an issue | close as *not planned* |
| `duplicate-suspect.md` | Intake found a likely duplicate (stays open) | `resolution:duplicate` as a proposal; humans close |
| `close-duplicate.md` | Duplicate of an existing item | `resolution:duplicate` + close as *duplicate* |
| `close-wontfix.md` | Valid but deliberately not planned | `resolution:wontfix` + close as *not planned* |
| `close-invalid.md` | Not actionable / not a real issue | `resolution:invalid` + close as *not planned* |
| `security-report.md` | Possible vulnerability filed publicly | `security` label, notify maintainers |
| `extension-suggestion.md` | Feature belongs as an Extension, not in Core | — |
| `extension-maintainer-ping.md` | Issue/PR touches a community Extension | `status:awaiting-response` |
| `ai-slop.md` | Signs of unverified AI-generated content | `status:needs-template` |
| `welcome-first-pr.md` | First-time contributor opened a PR | — |
