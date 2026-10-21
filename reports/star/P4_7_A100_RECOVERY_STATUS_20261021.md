# P4.7 recovery status — 2026-10-21

The two PointButton cells that stopped on 4080 were resumed on the existing A100 MPS servers from verified full checkpoints, using the frozen training source `62ca7399357c10645a955dcbead7ed2153a82c69`, the original formal configurations, and the original 10,000,000-step budgets.

At the latest direct observation (2026-10-21 18:47 local server time):

- `point-button-2028`: latest recovery segment logged approximately 4.08M environment steps.
- `point-button-2030`: latest recovery segment logged approximately 6.33M environment steps.
- A100 GPU0/GPU1 utilization was 19%/28%; both recovery processes were alive and their progress files were updating.
- 4080 retained six active cells and 5080 retained seven active cells. No task was terminated and no scientific configuration was changed.

The complete checkpoint inputs and the original 4080 records remain preserved. The two recovery cells continue toward their full budgets; this file records an intermediate operational state and is not a final result.

To use the available A100 capacity, the existing 4080 CPU cells `ant-2028`, `ant-2029`, `walker2d-2028`, `walker2d-2029`, and `ant-2031` were stopped at verified full checkpoints and resumed on the A100 MPS servers. Their original 5,000,000-step budgets and configurations were retained. A later direct observation showed fresh A100 progress files for all five migrated cells, no recovery log errors, and A100 GPU utilization of approximately 42%/41%. At that observation, `point-button-2028` had reached about 4.17M steps, `point-button-2030` about 6.42M, `ant-2028` about 2.585M, and `walker2d-2028` about 2.685M. The next verified write advanced the two recovery PointButton cells to approximately 4.175M and 6.425M steps, with A100 utilization approximately 41%/40%. A further write advanced them to approximately 4.180M and 6.430M steps, with reported training-only FPS of about 5.78 and 5.80. The next write advanced them to approximately 4.185M and 6.435M steps, with training-only FPS about 5.78 and 5.74. The latest write advanced them to approximately 4.190M and 6.440M steps, with training-only FPS about 5.73 and 5.73.
