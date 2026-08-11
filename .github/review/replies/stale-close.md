<!--
Use when: a gated item (needs-template / needs-cla / awaiting-response / changes-requested) passed its stale window plus 7 days with no author activity.
Action: close as *not planned*. Both the gate label and `status:stale` stay on the closed item.
-->

Closing this as **not planned**: it has been waiting on author action (`{{gate_label}}`) past the grace period with no activity.

This is routine queue hygiene, not a judgment on the idea itself. If you'd like to pick it up again:

- resolve what the gate label asked for, and
- comment here or open a fresh issue/PR referencing this one — we'll gladly reopen and continue from where it stopped.
