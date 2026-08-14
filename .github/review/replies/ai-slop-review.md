<!--
Use when: a PR *review* (not the PR/issue itself) shows signs of unverified AI-generated content — restates the PR description as findings, generic praise with no specific lines referenced, questions the diff already answers, "LGTM" alongside an unresolved question, or the same template reused across unrelated PRs. See AI_POLICY.md#reviews.
Action: hide/minimize the review comment first (GitHub review UI: "..." -> "Hide comment", pick the closest reason), then post this reply so the author and future readers know why. This does not set or change any status label — reviews don't gate the PR, only a maintainer's approval does.
-->

Hi @{{reviewer}}. Thanks for taking the time to review — community reviews are genuinely welcome here. This one has been minimized, though: per our [AI policy](https://github.com/ag2ai/ag2/blob/main/.github/AI_POLICY.md#reviews), it shows signs of being unverified AI output rather than a read of this specific diff:

{{observations}}

You're welcome to leave another review — just make sure it reflects your own read of the change, not a template applied to the PR description. Accounts that repeat this pattern across multiple PRs may have their ability to comment or review on this repository restricted.
