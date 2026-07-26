<!--
Use when: a PR has merge conflicts with main.
Action: set `status:changes-requested` (remove `status:needs-review`). Stale window: 30 days, then closed 7 days later.
-->

Hi @{{author}} — this PR now has **merge conflicts with `main`** and has been moved to `status:changes-requested`.

Please rebase onto the current `main` and resolve the conflicts. Once you push the updated branch, the PR will return to the review queue automatically.

⏳ If there's no update within **30 days**, this PR will be marked stale and closed **7 days** after that.
