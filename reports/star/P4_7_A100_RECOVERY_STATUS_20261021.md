# P4.7 recovery status — 2026-10-21

The two PointButton cells that stopped on 4080 were resumed on the existing A100 MPS servers from verified full checkpoints, using the frozen training source `62ca7399357c10645a955dcbead7ed2153a82c69`, the original formal configurations, and the original 10,000,000-step budgets.

At the latest direct observation (2026-10-21 18:47 local server time):

- `point-button-2028`: latest recovery segment logged approximately 4.08M environment steps.
- `point-button-2030`: latest recovery segment logged approximately 6.33M environment steps.
- A100 GPU0/GPU1 utilization was 19%/28%; both recovery processes were alive and their progress files were updating.
- 4080 retained six active cells and 5080 retained seven active cells. No task was terminated and no scientific configuration was changed.

The complete checkpoint inputs and the original 4080 records remain preserved. The two recovery cells continue toward their full budgets; this file records an intermediate operational state and is not a final result.
