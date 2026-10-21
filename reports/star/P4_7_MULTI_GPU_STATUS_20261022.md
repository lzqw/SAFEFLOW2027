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

At the **2026-10-22 03:17 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.175M / 10M steps** (FPS about 24.28, GPU utilization about 27%). Two 5080 lanes advanced to approximately **7.040M and 7.610M / 10M steps**; its other five observed lanes remained at approximately 7.840M, 6.990M, 5.630M, 6.480M, and 4.860M. GPU utilization was about 84%, and all observed training processes remained alive.

At the **2026-10-22 03:18:43 CST** follow-up, two 5080 lanes advanced to approximately **6.485M and 4.865M / 10M steps**; the other five observed lanes remained at approximately 7.040M, 7.610M, 7.840M, 6.990M, and 5.630M. GPU utilization was about 82%, and all seven training processes remained alive.

At the **2026-10-22 03:19 CST** follow-up, a 5080 lane advanced to approximately **5.635M / 10M steps**; the other six observed lanes remained unchanged. GPU utilization was about 81%, and all seven training processes remained alive.

At the **2026-10-22 03:19:52 CST** follow-up, A100 PointButton 2028 advanced to **4.260M / 10M steps** while PointButton 2030 remained at **6.505M / 10M steps** (GPU utilization about 41%/49%). A 5080 lane advanced to approximately **6.995M / 10M steps**; its other six observed lanes remained at approximately 5.635M, 6.485M, 4.865M, 7.040M, 7.610M, and 7.840M. GPU utilization was about 84%; all observed training processes remained alive.

At the **2026-10-22 03:20 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.180M / 10M steps** (FPS about 24.75, GPU utilization about 34%). A100 remained active at 4.260M/6.505M, and 5080 retained seven active lanes with GPU utilization about 87%; all observed training processes remained alive.

At the **2026-10-22 03:21:06 CST** follow-up, A100 PointButton 2030 advanced to **6.510M / 10M steps** while PointButton 2028 remained at **4.260M / 10M steps** (GPU utilization about 36%/42%). A 5080 lane advanced to approximately **7.845M / 10M steps**; its other six observed lanes remained at approximately 6.995M, 5.635M, 6.485M, 4.865M, 7.040M, and 7.610M. GPU utilization was about 83%; all observed training processes remained alive.

At the **2026-10-22 03:21:42 CST** follow-up, two 5080 lanes advanced to approximately **7.045M and 7.615M / 10M steps**; the other five observed lanes remained at approximately 7.845M, 6.995M, 5.635M, 6.485M, and 4.865M. GPU utilization was about 85%, and all seven training processes remained alive.

At the **2026-10-22 03:22:42 CST** follow-up, two 5080 lanes advanced to approximately **4.870M and 6.490M / 10M steps**; the other five observed lanes remained at approximately 7.045M, 7.615M, 7.845M, 6.995M, and 5.635M. GPU utilization was about 81%, and all seven training processes remained alive.

At the **2026-10-22 03:23:40 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.185M / 10M steps** (FPS about 24.80, GPU utilization about 29%). Three 5080 lanes advanced to approximately **7.000M, 5.640M, and 4.870M / 10M steps**; the other four remained at approximately 6.490M, 7.045M, 7.615M, and 7.845M. GPU utilization was about 85%, and all observed training processes remained alive.

At the **2026-10-22 03:25:19 CST** follow-up, five 5080 lanes advanced to approximately **7.850M, 7.000M, 5.640M, 4.870M, and 6.490M / 10M steps**; the other two remained at approximately 7.045M and 7.615M. GPU utilization was about 79%, and all seven training processes remained alive. A100 and 4080 remained active.

At the **2026-10-22 03:25:57 CST** follow-up, two 5080 lanes advanced to approximately **7.050M and 7.620M / 10M steps**; the other five observed lanes remained at approximately 7.850M, 7.000M, 5.640M, 4.870M, and 6.490M. GPU utilization was about 81%, and all seven training processes remained alive.

At the **2026-10-22 03:26:54 CST** follow-up, two 5080 lanes advanced to approximately **4.875M and 6.495M / 10M steps**; the other five observed lanes remained at approximately 7.050M, 7.620M, 7.850M, 7.000M, and 5.640M. GPU utilization was about 80%, and all seven training processes remained alive.

At the **2026-10-22 03:27:30 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.190M / 10M steps** (FPS about 24.82, GPU utilization about 31%). A100 remained active at 4.260M/6.510M, and 5080 retained seven active lanes with GPU utilization about 79%; all observed training processes remained alive.

At the **2026-10-22 03:28:11 CST** follow-up, three 5080 lanes advanced to approximately **7.005M, 5.645M, and 4.875M / 10M steps**; the other four observed lanes remained at approximately 6.495M, 7.050M, 7.620M, and 7.850M. GPU utilization was about 82%, and all seven training processes remained alive.

At the **2026-10-22 03:30:11 CST** follow-up, a 5080 lane advanced to approximately **7.855M / 10M steps**; the other six observed lanes remained unchanged. GPU utilization was about 86%, and all seven training processes remained alive.

At the **2026-10-22 03:30:46 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.195M / 10M steps** (FPS about 24.93, GPU utilization about 34%). Two 5080 lanes advanced to approximately **7.055M and 7.625M / 10M steps**; the other five remained at approximately 7.855M, 7.005M, 5.645M, 4.875M, and 6.495M. GPU utilization was about 82%, and all observed training processes remained alive.

At the **2026-10-22 03:31:57 CST** follow-up, two 5080 lanes advanced to approximately **5.650M and 4.880M / 10M steps**; the other five observed lanes remained at approximately 7.055M, 7.625M, 7.855M, 7.005M, and 6.500M. GPU utilization was about 79%, and all seven training processes remained alive.

At the **2026-10-22 03:32:32 CST** follow-up, three 5080 lanes advanced to approximately **7.010M, 5.650M, and 4.880M / 10M steps**; the other four observed lanes remained at approximately 6.500M, 7.625M, 7.855M, and 7.055M. GPU utilization was about 79%, and all seven training processes remained alive.

At the **2026-10-22 03:33:36 CST** follow-up, a 5080 lane advanced to approximately **7.860M / 10M steps**; the other six observed lanes remained unchanged. GPU utilization was about 82%, and all seven training processes remained alive.

At the **2026-10-22 03:34:12 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.200M / 10M steps** (FPS about 24.97, GPU utilization about 35%). A 5080 lane advanced to approximately **7.060M / 10M steps**; its other six observed lanes remained at approximately 7.860M, 7.005M, 5.650M, 4.880M, 6.500M, and 7.625M. GPU utilization was about 86%, and all observed training processes remained alive.

At the **2026-10-22 03:34:48 CST** follow-up, A100 PointButton 2028 advanced to **4.265M / 10M steps** while PointButton 2030 remained at **6.510M / 10M steps** (GPU utilization about 37%/44%). A 5080 lane advanced to approximately **7.630M / 10M steps**; its other six observed lanes remained at approximately 7.060M, 7.860M, 7.005M, 5.650M, 4.880M, and 6.500M. GPU utilization was about 84%; all observed training processes remained alive.

At the **2026-10-22 03:35:28 CST** follow-up, A100 PointButton 2030 advanced to **6.515M / 10M steps** while PointButton 2028 remained at **4.265M / 10M steps** (GPU utilization about 35%/41%). Three 5080 lanes advanced to approximately **4.885M, 6.505M, and 7.630M / 10M steps**; the other four remained at approximately 7.060M, 7.860M, 7.005M, and 5.650M. GPU utilization was about 87%; all observed training processes remained alive.

At the **2026-10-22 03:36:07 CST** follow-up, three 5080 lanes advanced to approximately **5.655M, 4.885M, and 6.505M / 10M steps**; the other four observed lanes remained at approximately 7.060M, 7.860M, 7.005M, and 7.630M. GPU utilization was about 82%, and all seven training processes remained alive.

At the **2026-10-22 03:36:41 CST** follow-up, a 5080 lane advanced to approximately **7.015M / 10M steps**; the other six observed lanes remained unchanged. GPU utilization was about 83%, and all seven training processes remained alive.

At the **2026-10-22 03:37:18 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.205M / 10M steps** (FPS about 24.87, GPU utilization about 29%). A100 remained active at 4.265M/6.515M, and 5080 retained seven active lanes with GPU utilization about 83%; all observed training processes remained alive.

At the **2026-10-22 03:37:49 CST** follow-up, a 5080 lane advanced to approximately **7.865M / 10M steps**; the other six observed lanes remained unchanged. GPU utilization was about 84%, and all seven training processes remained alive.

At the **2026-10-22 03:38:21 CST** follow-up, a 5080 lane advanced to approximately **7.065M / 10M steps**; the other six observed lanes remained unchanged. GPU utilization was about 86%, and all seven training processes remained alive.

At the **2026-10-22 03:38:51 CST** follow-up, a 5080 lane advanced to approximately **7.635M / 10M steps**; the other six observed lanes remained unchanged. GPU utilization was about 85%, and all seven training processes remained alive.

At the **2026-10-22 03:39:26 CST** follow-up, two 5080 lanes advanced to approximately **6.510M and 7.635M / 10M steps**; the other five observed lanes remained at approximately 7.065M, 7.865M, 7.015M, 5.655M, and 4.885M. GPU utilization was about 81%, and all seven training processes remained alive.

At the **2026-10-22 03:40:02 CST** follow-up, a 5080 lane advanced to approximately **4.890M / 10M steps**; the other six observed lanes remained unchanged. GPU utilization was about 79%, and all seven training processes remained alive.

At the **2026-10-22 03:40:37 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.210M / 10M steps** (FPS about 24.44, GPU utilization about 29%). Three 5080 lanes advanced to approximately **7.020M, 5.660M, and 6.510M / 10M steps**; the other four remained at approximately 7.635M, 7.065M, 7.865M, and 4.890M. GPU utilization was about 86%, and all observed training processes remained alive.

At the **2026-10-22 03:41:11 CST** follow-up, two 5080 lanes advanced to approximately **7.020M and 5.660M / 10M steps**; the other five observed lanes remained at approximately 6.510M, 7.635M, 7.065M, 7.865M, and 4.890M. GPU utilization was about 82%, and all seven training processes remained alive.

At the **2026-10-22 03:42:09 CST** follow-up, a 5080 lane advanced to approximately **7.870M / 10M steps**; the other six observed lanes remained unchanged. GPU utilization was about 83%, and all seven training processes remained alive.

At the **2026-10-22 03:42:51 CST** follow-up, three 5080 lanes advanced to approximately **7.070M, 7.870M, and 7.640M / 10M steps**; the other four observed lanes remained at approximately 6.510M, 5.660M, 4.890M, and 7.065M. GPU utilization was about 86%, and all seven training processes remained alive. A100 and 4080 remained active.

At the **2026-10-22 03:43:25 CST** follow-up, two 5080 lanes advanced to approximately **7.070M and 7.640M / 10M steps**; the other five observed lanes remained at approximately 6.510M, 7.870M, 5.660M, 4.890M, and 7.065M. GPU utilization was about 84%, and all seven training processes remained alive.

At the **2026-10-22 03:43:59 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.215M / 10M steps** (FPS about 24.24, GPU utilization about 30%). Three 5080 lanes advanced to approximately **4.895M, 6.515M, and 7.640M / 10M steps**; the other four remained at approximately 7.070M, 7.870M, 5.660M, and 7.065M. GPU utilization was about 83%, and all observed training processes remained alive.

At the **2026-10-22 03:44:33 CST** follow-up, a 5080 lane advanced to approximately **5.665M / 10M steps**; the other six observed lanes remained unchanged. GPU utilization was about 83%, and all seven training processes remained alive.

At the **2026-10-22 03:45:06 CST** follow-up, three 5080 lanes advanced to approximately **7.025M, 6.515M, and 7.640M / 10M steps**; the other four observed lanes remained at approximately 5.665M, 4.895M, 7.070M, and 7.870M. GPU utilization was about 84%, and all seven training processes remained alive.

At the **2026-10-22 03:45:38 CST** follow-up, two 5080 lanes advanced to approximately **7.025M and 6.515M / 10M steps**; the other five observed lanes remained at approximately 7.640M, 5.665M, 4.895M, 7.070M, and 7.870M. GPU utilization was about 83%, and all seven training processes remained alive.

At the **2026-10-22 03:46:12 CST** follow-up, a 5080 lane advanced to approximately **7.875M / 10M steps**; the other six observed lanes remained unchanged. GPU utilization was about 85%, and all seven training processes remained alive.

At the **2026-10-22 03:46:44 CST** follow-up, two 5080 lanes advanced to approximately **7.075M and 7.875M / 10M steps**; the other five observed lanes remained at approximately 6.515M, 5.665M, 4.895M, 7.640M, and 7.870M. GPU utilization was about 85%, and all seven training processes remained alive.

At the **2026-10-22 03:47:18 CST** follow-up, two 5080 lanes advanced to approximately **7.075M and 7.645M / 10M steps**; the other five observed lanes remained at approximately 6.515M, 7.875M, 5.665M, 4.895M, and 7.070M. GPU utilization was about 83%, and all seven training processes remained alive.

At the **2026-10-22 03:31:21 CST** follow-up, two 5080 lanes advanced to approximately **4.880M and 6.500M / 10M steps**; the other five observed lanes remained at approximately 7.055M, 7.625M, 7.855M, 7.005M, and 5.645M. GPU utilization was about 83%, and all seven training processes remained alive.

At the **2026-10-22 03:47:55 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.220M / 10M steps** (FPS about 24.32, GPU utilization about 35%). Three 5080 lanes advanced to approximately **6.520M, 7.645M, and 7.075M / 10M steps**; the other four remained at approximately 7.870M, 7.025M, 5.665M, and 4.895M. GPU utilization was about 85%, and all observed training processes remained alive.

At the **2026-10-22 03:48:52 CST** follow-up, A100 PointButton 2028 advanced to **4.270M / 10M steps** while PointButton 2030 remained at **6.515M / 10M steps** (GPU utilization about 35%/41%). Four 5080 lanes advanced to approximately **5.670M, 4.900M, 6.520M, and 7.875M / 10M steps**; the other three remained at approximately 7.075M, 7.645M, and 7.025M. GPU utilization was about 84%; all observed training processes remained alive.

At the **2026-10-22 03:49:31 CST** follow-up, A100 PointButton 2028 advanced to **4.275M / 10M steps** while PointButton 2030 remained at **6.515M / 10M steps** (GPU utilization about 34%/41%). Three 5080 lanes advanced to approximately **7.030M, 7.645M, and 7.875M / 10M steps**; the other four remained at approximately 6.520M, 5.670M, 4.900M, and 7.075M. GPU utilization was about 86%; all observed training processes remained alive.

At the **2026-10-22 03:50:10 CST** follow-up, A100 PointButton 2030 advanced to **6.520M / 10M steps** while PointButton 2028 remained at **4.275M / 10M steps** (GPU utilization about 37%/44%). Five 5080 lanes advanced to approximately **7.880M, 7.030M, 5.670M, 4.900M, and 6.520M / 10M steps**; the other two remained at approximately 7.645M and 7.075M. GPU utilization was about 83%; all observed training processes remained alive.

At the **2026-10-22 03:51:12 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.225M / 10M steps** (FPS about 24.60, GPU utilization about 30%). Three 5080 lanes advanced to approximately **7.080M, 7.880M, and 7.030M / 10M steps**; the other four remained at approximately 5.670M, 4.900M, 6.520M, and 7.645M. GPU utilization was about 83%, and all observed training processes remained alive.

At the **2026-10-22 03:51:54 CST** follow-up, two 5080 lanes advanced to approximately **7.080M and 7.650M / 10M steps**; the other five observed lanes remained at approximately 7.880M, 7.030M, 5.670M, 4.900M, and 6.520M. GPU utilization was about 83%, and all seven training processes remained alive.

At the **2026-10-22 03:52:32 CST** follow-up, three 5080 lanes advanced to approximately **4.905M, 6.525M, and 7.650M / 10M steps**; the other four observed lanes remained at approximately 7.880M, 7.030M, 5.670M, and 7.080M. GPU utilization was about 82%, and all seven training processes remained alive.

At the **2026-10-22 03:53:09 CST** follow-up, three 5080 lanes advanced to approximately **5.675M, 4.905M, and 6.525M / 10M steps**; the other four observed lanes remained at approximately 7.650M, 7.080M, 7.880M, and 7.030M. GPU utilization was about 82%, and all seven training processes remained alive.

At the **2026-10-22 03:53:42 CST** follow-up, a 5080 lane advanced to approximately **7.035M / 10M steps**; the other six observed lanes remained unchanged. GPU utilization was about 83%, and all seven training processes remained alive.

At the **2026-10-22 03:54:14 CST** follow-up, the 4080 PointButton 2029 lane advanced to **4.230M / 10M steps** (FPS about 24.84, GPU utilization about 31%). Two 5080 lanes advanced to approximately **7.035M and 6.525M / 10M steps**; the other five remained at approximately 7.650M, 7.080M, 7.880M, 5.675M, and 4.905M. GPU utilization was about 84%, and all observed training processes remained alive.
