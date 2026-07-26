<!--
Use when: a PR or issue shows signs of unverified AI-generated content — description doesn't match the diff, invented APIs, boilerplate reasoning, claims of testing with no evidence.
Action: set `status:needs-template`. Stale window: 7 days, then closed 7 days later. Stay respectful: AI assistance is welcome, unverified output is not.
-->

Hi @{{author}}. AI-assisted contributions are welcome in AG2 — but per our [AI policy](https://github.com/ag2ai/ag2/blob/main/.github/AI_POLICY.md), **you remain responsible for everything you submit**, and this submission shows signs that its content wasn't verified:

{{observations}}

To move this forward, please:

1. explain in your own words what problem this solves and how the change works,
2. make the description accurately reflect the actual diff,
3. include how you validated it — commands run, tests executed, observed output.

We ask this of every contribution flagged this way; unverified submissions cost reviewers more than they save authors. Once the description meets the bar, the `status:needs-template` label is removed and the normal flow continues.

⏳ Without an update within **7 days**, this will be marked stale and closed **7 days** after that.
