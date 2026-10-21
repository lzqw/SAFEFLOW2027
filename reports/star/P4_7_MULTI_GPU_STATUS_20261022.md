# P4.7 Multi-GPU Training Status

This is an operational snapshot of the ongoing P4.7 training lanes. It is not a completion claim; all listed jobs remain within their configured training budgets.

Snapshot time: **2026-10-22 02:02 CST**

| Host | Active lane | Latest verified progress | GPU utilization |
|---|---|---:|---:|
| A100 | PointButton seed 2028 | 4.230M / 10M steps (FPS about 5.71) | 50% |
| A100 | PointButton seed 2030 | 6.480M / 10M steps (FPS about 5.76) | 51% |
| 4080 | PointButton seed 2029 | 4.065M / 10M steps (FPS about 23.52) | 31% |
| 5080 | 7 active Point lanes | Running; latest aggregate GPU observation | 76% |

All observed training processes were alive at the snapshot. The A100 recovery history is maintained in [P4.7 A100 recovery status](P4_7_A100_RECOVERY_STATUS_20261021.md).

At the **2026-10-22 02:07 CST** follow-up, all three hosts remained reachable with the same active-lane counts. A100 remained at 4.230M/6.480M for the two tracked PointButton jobs (GPU utilization about 40%/43%); 4080 advanced to **4.070M / 10M steps** (FPS about 24.56, GPU utilization about 27%); 5080 remained at seven active lanes (GPU utilization about 82%).

At the **2026-10-22 02:08 CST** follow-up, A100 PointButton 2028 advanced to **4.235M / 10M steps** (FPS about 6.31), while PointButton 2030 remained at 6.480M; A100 GPU utilization was about 50%/52%. The 4080 lane remained at 4.070M and the 5080 lane count remained seven, with GPU utilization about 29% and 82%, respectively.
