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

At the **2026-10-22 02:28 CST** follow-up, A100 PointButton 2028/2030 advanced to **4.240M/6.490M / 10M steps** (FPS about 6.32/6.27), with GPU utilization about 44%/51%. The 4080 PointButton 2029 lane advanced to **4.100M / 10M steps** (FPS about 24.56, GPU utilization about 28%). The 5080 retained seven active lanes with GPU utilization about 79%; all observed training processes remained alive.

At the **2026-10-22 02:30 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.105M / 10M steps** (FPS about 24.43, GPU utilization about 31%). A100 remained at 4.240M/6.490M for the two tracked lanes (GPU utilization about 35%/41%), while 5080 retained seven active lanes (GPU utilization about 85%); all observed training processes remained alive.

At the **2026-10-22 02:32 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.110M / 10M steps** (FPS about 24.80, GPU utilization about 31%). A100 remained at 4.240M/6.490M (GPU utilization about 36%/41%), and 5080 retained seven active lanes (GPU utilization about 80%); all observed training processes remained alive.

At the **2026-10-22 02:36 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.115M / 10M steps** (FPS about 25.12, GPU utilization about 29%). A100 remained at 4.240M/6.490M (GPU utilization about 36%/42%), and 5080 retained seven active lanes (GPU utilization about 81%); all observed training processes remained alive.
At the **2026-10-22 02:56 CST** follow-up, A100 PointButton 2028/2030 advanced to **4.250M/6.500M / 10M steps** (FPS about 5.73/5.79, GPU utilization about 45%/52%). The 4080 PointButton 2029 lane advanced to **4.140M / 10M steps** (FPS about 24.87, GPU utilization about 29%). The 5080 retained seven active lanes and its GPU utilization was about 86%; all observed training processes remained alive.

At the **2026-10-22 02:57 CST** follow-up, A100 remained at **4.250M/6.500M / 10M steps** (GPU utilization about 46%/50%). The 4080 PointButton 2029 lane advanced to **4.145M / 10M steps** (FPS about 24.32, GPU utilization about 33%). The 5080 retained seven active lanes with GPU utilization about 83%; all observed training processes remained alive.

At the **2026-10-22 03:00 CST** follow-up, A100 remained at **4.250M/6.500M / 10M steps** (GPU utilization about 36%/42%). The 4080 PointButton 2029 lane advanced to **4.150M / 10M steps** (FPS about 24.71, GPU utilization about 31%). The 5080 retained seven active lanes with GPU utilization about 80%; all observed training processes remained alive.

At the **2026-10-22 03:01 CST** follow-up, A100 remained at **4.250M/6.500M / 10M steps** (GPU utilization about 35%/42%) and 4080 remained at **4.150M / 10M steps** (GPU utilization about 33%). The 5080 retained seven active lanes and advanced its latest observed lanes to approximately **7.020M, 7.590M, 7.820M, 6.970M, 5.610M, 6.460M, and 4.840M / 10M steps**, with GPU utilization about 86%; all observed training processes remained alive.

At the **2026-10-22 03:02 CST** follow-up, the 5080 retained seven active lanes and advanced three observed lanes to approximately **5.615M, 6.465M, and 4.845M / 10M steps**; the other four remained at approximately 7.020M, 7.590M, 7.820M, and 6.970M. GPU utilization was about 79%, and all observed training processes remained alive.

At the **2026-10-22 03:03 CST** follow-up, the 5080 PointButton 2031 lane advanced to approximately **6.975M / 10M steps**; the other six observed lanes remained unchanged. GPU utilization was about 80%, and all seven training processes remained alive.

At the **2026-10-22 03:03 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.155M / 10M steps** (FPS about 24.16, GPU utilization about 32%). A100 remained at **4.250M/6.500M / 10M steps** (GPU utilization about 36%/42%), while 5080 retained seven active lanes with GPU utilization about 82%; all observed training processes remained alive.

At the **2026-10-22 03:04 CST** follow-up, the 5080 PointGoal lanes advanced to approximately **7.595M and 7.825M / 10M steps**; the other five observed lanes remained at approximately 6.975M, 5.615M, 6.465M, 4.845M, and 7.020M. GPU utilization was about 83%, and all seven training processes remained alive. A100 and 4080 remained active.

At the **2026-10-22 03:05 CST** follow-up, a further 5080 lane advanced to approximately **7.025M / 10M steps**; the other six observed lanes remained unchanged. The 5080 retained seven active processes with GPU utilization about 78%; A100 and 4080 remained active as well.

At the **2026-10-22 03:07 CST** follow-up, A100 PointButton 2028 advanced to **4.255M / 10M steps** while PointButton 2030 remained at **6.500M / 10M steps** (GPU utilization about 35%/44%). A 5080 lane also advanced to approximately **4.850M / 10M steps**; its other six observed lanes remained at approximately 7.025M, 7.595M, 7.825M, 6.975M, 5.615M, and 6.465M, with GPU utilization about 81%. The 4080 remained at 4.155M; all observed training processes remained alive.

At the **2026-10-22 03:07:35 CST** follow-up, A100 PointButton 2030 advanced to **6.505M / 10M steps** (PointButton 2028 remained at 4.255M; GPU utilization about 33%/43%). The 4080 PointButton 2029 lane advanced to **4.160M / 10M steps** (FPS about 24.81, GPU utilization about 25%). On 5080, three lanes advanced to approximately **6.980M, 5.620M, and 6.470M / 10M steps**; the other four were approximately 4.850M, 7.025M, 7.595M, and 7.825M. GPU utilization was about 83%; all observed training processes remained alive.

At the **2026-10-22 03:08:46 CST** follow-up, a 5080 lane advanced to approximately **7.830M / 10M steps**; the other six observed lanes remained unchanged. GPU utilization was about 85%, and all seven training processes remained alive.

At the **2026-10-22 03:09 CST** follow-up, two 5080 lanes advanced to approximately **7.030M and 7.600M / 10M steps**; the remaining observed lanes were approximately 7.830M, 6.980M, 5.620M, 6.470M, and 4.850M. GPU utilization was about 80%, and all seven training processes remained alive. A100 and 4080 remained active.

At the **2026-10-22 03:10 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.165M / 10M steps** (FPS about 24.07, GPU utilization about 30%). A 5080 lane advanced to approximately **4.855M / 10M steps**; its other six observed lanes remained at approximately 7.030M, 7.600M, 7.830M, 6.980M, 5.620M, and 6.470M. GPU utilization was about 83%, and all observed training processes remained alive.

At the **2026-10-22 03:10:42 CST** follow-up, a 5080 lane advanced to approximately **6.475M / 10M steps**; the other six observed lanes remained unchanged. GPU utilization was about 85%, and all seven training processes remained alive.

At the **2026-10-22 03:11 CST** follow-up, two 5080 lanes advanced to approximately **6.985M and 5.625M / 10M steps**; the other five observed lanes remained at approximately 6.475M, 4.855M, 7.030M, 7.600M, and 7.830M. GPU utilization was about 84%, and all seven training processes remained alive.

At the **2026-10-22 03:12:54 CST** follow-up, a 5080 lane advanced to approximately **7.835M / 10M steps**; the other six observed lanes remained unchanged. GPU utilization was about 80%, and all seven training processes remained alive.

At the **2026-10-22 03:13 CST** follow-up, two 5080 lanes advanced to approximately **7.035M and 7.605M / 10M steps**; the other five observed lanes remained at approximately 7.835M, 6.985M, 5.625M, 6.475M, and 4.855M. GPU utilization was about 81%, and all seven training processes remained alive.

At the **2026-10-22 03:14 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.170M / 10M steps** (FPS about 24.40, GPU utilization about 31%). A100 remained active at 4.255M/6.505M, and 5080 retained seven active lanes with GPU utilization about 81%; all observed training processes remained alive.

At the **2026-10-22 03:14:34 CST** follow-up, two 5080 lanes advanced to approximately **6.480M and 4.860M / 10M steps**; the other five observed lanes remained at approximately 7.035M, 7.605M, 7.835M, 6.985M, and 5.625M. GPU utilization was about 83%, and all seven training processes remained alive.

At the **2026-10-22 03:15 CST** follow-up, two 5080 lanes advanced to approximately **5.630M and 6.480M / 10M steps**; the other five observed lanes remained at approximately 7.035M, 7.605M, 7.835M, 6.985M, and 4.860M. GPU utilization was about 79%, and all seven training processes remained alive.

At the **2026-10-22 03:15:46 CST** follow-up, two 5080 lanes advanced to approximately **6.990M and 5.630M / 10M steps**; the other five observed lanes remained at approximately 6.480M, 4.860M, 7.035M, 7.605M, and 7.835M. GPU utilization was about 80%, and all seven training processes remained alive.

At the **2026-10-22 03:16:55 CST** follow-up, a 5080 lane advanced to approximately **7.840M / 10M steps**; the other six observed lanes remained unchanged. GPU utilization was about 86%, and all seven training processes remained alive.
