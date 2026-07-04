<!--
Use when: a bug report can't be reproduced or is missing key information.
Action: set `status:awaiting-response`. Stale window: 10 days, then closed 7 days later.
-->

Hi @{{author}}, thanks for the report — we couldn't reproduce this from the current description.

To move forward, please share:

- the `ag2` version (`python -c "import ag2; print(ag2.__version__)"`) and Python version,
- a **minimal, self-contained script** that triggers the problem,
- the full traceback or unexpected output,
- the LLM provider/model in use, if relevant.

Once you reply, the issue goes back into the triage queue automatically.

⏳ If there's no response within **10 days**, this issue will be marked stale and closed **7 days** after that — reopening with the details is always welcome.
