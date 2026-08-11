<!--
Use when: a gated item passed its gate's grace period with no author activity and is being marked stale.
Action: add `status:stale` on top of the gate label (the gate label stays). Posted by the stale automation.
-->

Hi @{{author}} — this is still waiting on your action (`{{gate_label}}`) and has passed its grace period, so it's now marked **stale**.

⏳ Without activity it will be **closed in 7 days** as *not planned*.

Any response resets this: resolve what the gate label asks for — or comment if you need help or more time — and the stale mark is removed.
