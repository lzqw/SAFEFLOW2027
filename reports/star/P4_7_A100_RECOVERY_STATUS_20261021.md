# P4.7 recovery status — 2026-10-21

The two PointButton cells that stopped on 4080 were resumed on the existing A100 MPS servers from verified full checkpoints, using the frozen training source `62ca7399357c10645a955dcbead7ed2153a82c69`, the original formal configurations, and the original 10,000,000-step budgets.

At the latest direct observation (2026-10-21 18:47 local server time):

- `point-button-2028`: latest recovery segment logged approximately 4.08M environment steps.
- `point-button-2030`: latest recovery segment logged approximately 6.33M environment steps.
- A100 GPU0/GPU1 utilization was 19%/28%; both recovery processes were alive and their progress files were updating.
- 4080 retained six active cells and 5080 retained seven active cells. No task was terminated and no scientific configuration was changed.

The complete checkpoint inputs and the original 4080 records remain preserved. The two recovery cells continue toward their full budgets; this file records an intermediate operational state and is not a final result.

To use the available A100 capacity, the existing 4080 CPU cells `ant-2028`, `ant-2029`, `walker2d-2028`, `walker2d-2029`, and `ant-2031` were stopped at verified full checkpoints and resumed on the A100 MPS servers. Their original 5,000,000-step budgets and configurations were retained. A later direct observation showed fresh A100 progress files for all five migrated cells, no recovery log errors, and A100 GPU utilization of approximately 42%/41%. At that observation, `point-button-2028` had reached about 4.17M steps, `point-button-2030` about 6.42M, `ant-2028` about 2.585M, and `walker2d-2028` about 2.685M. The next verified write advanced the two recovery PointButton cells to approximately 4.175M and 6.425M steps, with A100 utilization approximately 41%/40%. A further write advanced them to approximately 4.180M and 6.430M steps, with reported training-only FPS of about 5.78 and 5.80. The next write advanced them to approximately 4.185M and 6.435M steps, with training-only FPS about 5.78 and 5.74. A later write advanced them to approximately 4.190M and 6.440M steps, with training-only FPS about 5.73 and 5.73. The latest write advanced them to approximately 4.195M and 6.445M steps, with training-only FPS about 5.63 and 5.64. A further write advanced them to approximately 4.200M and 6.450M steps, with training-only FPS about 5.79 and 5.70.

The next verified write advanced `point-button-2028` to approximately 4.205M steps (training-only FPS about 5.63); `point-button-2030` remained at approximately 6.450M steps (training-only FPS about 5.70) at the 2026-10-22 00:42 local observation. Both processes remained alive, with A100 GPU utilization approximately 43%/42%.

The following verified write advanced `point-button-2030` to approximately 6.455M steps (training-only FPS about 5.64); `point-button-2028` remained at approximately 4.205M steps (training-only FPS about 5.63) at the 2026-10-22 00:44 local observation. Both processes remained alive, with A100 GPU utilization approximately 42%/42%.

The next verified write advanced `point-button-2028` to approximately 4.210M steps (training-only FPS about 5.73) and `point-button-2030` to approximately 6.460M steps (training-only FPS about 5.70) at the 2026-10-22 00:57 local observation. Both processes remained alive, with A100 GPU utilization approximately 52%/50%.

The following verified write advanced `point-button-2028` to approximately 4.215M steps (training-only FPS about 5.82); `point-button-2030` remained at approximately 6.460M steps (training-only FPS about 5.70) at the 2026-10-22 01:11 local observation. Both processes remained alive, with A100 GPU utilization approximately 42%/43%.

The next verified write advanced `point-button-2030` to approximately 6.465M steps (training-only FPS about 5.75); `point-button-2028` remained at approximately 4.215M steps (training-only FPS about 5.82) at the 2026-10-22 01:13 local observation. Both processes remained alive, with A100 GPU utilization approximately 42%/41%.

The following verified write advanced `point-button-2028` to approximately 4.220M steps (training-only FPS about 5.61) and `point-button-2030` to approximately 6.470M steps (training-only FPS about 5.61) at the 2026-10-22 01:27 local observation. Both processes remained alive, with A100 GPU utilization approximately 44%/44%.

The next verified write advanced `point-button-2028` to approximately 4.225M steps (training-only FPS about 5.69) and `point-button-2030` to approximately 6.475M steps (training-only FPS about 5.66) at the 2026-10-22 01:53 local observation. Both processes remained alive, with A100 GPU utilization approximately 49%/48%.

The following verified write advanced `point-button-2028` to approximately 4.230M steps (training-only FPS about 5.71) and `point-button-2030` to approximately 6.480M steps (training-only FPS about 5.76) at the 2026-10-22 01:56 local observation. Both processes remained alive, with A100 GPU utilization approximately 43%/44%.

The next verified write advanced `point-button-2028` to approximately 4.235M steps (training-only FPS about 6.31); `point-button-2030` remained at approximately 6.480M steps (training-only FPS about 5.76) at the 2026-10-22 02:08 local observation. Both processes remained alive, with A100 GPU utilization approximately 50%/52%.

At the 2026-10-22 02:28 local observation, the next verified writes advanced `point-button-2028` to approximately 4.240M steps (training-only FPS about 6.32) and `point-button-2030` to approximately 6.490M steps (training-only FPS about 6.27). Both processes remained alive, with A100 GPU utilization approximately 44%/51%.

At the 2026-10-22 02:56 local observation, verified writes advanced `point-button-2028` to approximately 4.250M steps (training-only FPS about 5.73) and `point-button-2030` to approximately 6.500M steps (training-only FPS about 5.79). Both processes remained alive, with A100 GPU utilization approximately 45%/52%.

At the 2026-10-22 03:07 local observation, `point-button-2028` advanced to approximately 4.255M steps (training-only FPS about 5.90); `point-button-2030` remained at approximately 6.500M steps (training-only FPS about 5.79). Both processes remained alive, with A100 GPU utilization approximately 35%/44%.
