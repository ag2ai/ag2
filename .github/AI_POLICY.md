# AI Policy

AG2 is an agent framework, and we welcome contributors who use AI tools —
including AG2 itself — to help build it. With that comes a high bar: **you
remain responsible for everything you publish** — code, issue descriptions,
pull request bodies, comments, and reviews — and we take care with what we
merge and release.

## Code: ownership and licensing

If you used AI to generate code, make sure you understand what it does and
have tested it. If your tests were also AI-generated, check that they actually
exercise the behavior — passing tests written alongside the code can mask bugs.

By submitting a pull request, you certify under AG2's Contributor License
Agreement (signed via cla-assistant.io when you open a PR) that you have the
right to license your contribution under the project's
[Apache 2.0](../LICENSE) license. AI-assisted contributions are no exception —
please confirm that any AI-generated code can be released under Apache 2.0,
and does not include third-party material you don't have the right to
relicense, before you submit it.

## Issues, PRs, and discussion

If you are opening an issue, we expect you to understand the problem well
enough to describe it clearly. If AI helps you draft the description, please
review and edit it before posting so it reflects your own understanding.

If you are opening a pull request, we expect you to be able to explain the
proposed changes — in the PR body and in responses to reviewer questions. If
you used AI to draft text, make sure it accurately reflects your intent, and
verify any specific claims (file paths, behaviors, error messages) against the
actual diff before submitting.

**Be prepared to discuss and revise your contribution in your own words based
on reviewer feedback.** Reviewers may ask follow-up questions; pasting AI
responses back without engaging with the question doesn't move the
conversation forward.

We appreciate when contributors mention significant AI assistance in the PR
body — it helps reviewers calibrate their attention.

If you have reason to share raw AI output in a comment, place it in a quote
block (e.g., using `>`), disclose it as such, and add your own commentary
explaining its relevance. Please avoid sharing long snippets.

## Reviews

We welcome reviews from anyone in the community, not just maintainers — a
second pair of eyes on a diff is valuable regardless of who it comes from. AI
tools can help you read a large diff faster or spot patterns worth a closer
look, but **the review must reflect that a person actually engaged with the
code**, the same standard we hold contributions to.

A review reads as unverified AI output — and undermines the trust reviewers
are supposed to add — when it does things like:

- restate the PR description back as findings, instead of responding to the
  diff itself,
- praise implementation choices in generic terms without pointing at specific
  lines or behavior,
- ask questions that reading the code would answer directly (e.g., "is this
  tested or mocked?" when the test file is in the diff),
- approve or say "LGTM" while leaving its own open question unresolved,
- reuse the same template (e.g., a numbered "Observations" list) across
  otherwise unrelated PRs.

None of this means AI-assisted review is unwelcome — it means the reviewer
stands behind it the same way an author stands behind AI-assisted code. If
you used AI to help draft a review, disclose it and make sure the content
reflects your own read of the change, not the PR description reflected back.

**Maintainers may hide or minimize a review that shows clear signs of
unverified AI-generated content**, and will leave a comment explaining why
(see the [`ai-slop-review`](review/replies/ai-slop-review.md) canned reply).
Reviews don't gate a PR — only a core maintainer's approval does (see the
[Triage Policy](review/TRIAGE_POLICY.md#4-pull-request-flow)) — but a
low-effort AI review can still mislead less experienced contributors into
thinking a change has been vetted when it hasn't. An account that repeats
this pattern across multiple PRs may have its ability to comment or review on
this repository restricted.

## Non-native English speakers

We understand that AI is useful when communicating as a non-native English
speaker. If you are using AI to edit your comments for this purpose, please
take the time to ensure it reflects your own voice and ideas. If using AI for
translation, we recommend writing in your native language and including the
AI translation in a quote block.

## Maintainer discretion

Maintainers may close or de-prioritize PRs where the contributor cannot
explain the changes, has not tested them, or cannot engage with reviewer
questions. The bar is the same regardless of how the contribution was
produced — we just need to be able to trust that a human stands behind it.
