<!--
Use when: a PR description doesn't follow .github/PULL_REQUEST_TEMPLATE.md (missing "Why are these changes needed?", no validation info, unchecked mandatory sections).
Action: set `status:needs-template`. Stale window: 7 days, then closed 7 days later.
-->

Hi @{{author}}, thanks for the contribution!

The PR description doesn't follow our [pull request template](https://github.com/ag2ai/ag2/blob/main/.github/PULL_REQUEST_TEMPLATE.md), and PRs aren't reviewed until it does — this is what's missing:

{{missing_sections}}

In particular, per our [AI-assisted contribution policy](https://github.com/ag2ai/ag2/blob/main/.github/AI_POLICY.md), the description must explain the real problem this solves, accurately reflect the diff, and include how the change was validated and tested.

Please **edit the PR description** to fill in the template. Once it's fixed, the `status:needs-template` label will be removed and the PR will enter the review queue.

⏳ If there's no update within **7 days**, this PR will be marked stale and closed **7 days** after that.
