<!--
Use when: a feature request or PR expands Core's surface (new integration, third-party SDK dependency) and belongs as an Extension instead.
Action: explain the Core/Extension split; for PRs, request the change be restructured under ag2/extensions/.
-->

Thanks @{{author}}! This is useful functionality — and it's a textbook fit for an **Extension** rather than Core.

AG2 v1.0 keeps Core intentionally small: reliance on a third-party SDK, or an integration best owned by someone closer to it, is exactly what our [Contribution Policy](https://docs.ag2.ai/latest/docs/user-guide/contribution_policy/) routes to Extensions. Extensions are first-class: same quality bar, same docs standards — the difference is that a **named maintainer** (you, or your team) commits to keeping it working.

What an Extension contribution needs:

- code under `ag2/extensions/`, with third-party packages declared as additional dependencies (guarded via `missing_additional_dependency`, not pyproject extras),
- tests and documentation (module docstring with a `Maintainer: <github-handle>` line, plus a page under the Extensions docs section),
- your commitment to respond to issues and keep it compatible with Core releases.

If you're up for maintaining it, we'll gladly review — the policy page above has the full checklist.
