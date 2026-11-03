Success with 95% Wilson intervals. With 20 episodes per arm and ~90% success, the smallest drop detectable at 80% power is about 36 points, so differences between configs below that are noise.

| run | suite | config | initial states | success | rate % | 95% CI | calls/episode | steps/success | rollout min | client p50 ms | server p50 ms | peak VRAM GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| j1/base_t0 | libero_10 | 10 steps, replan 5 | 0-1 | 19/20 | 95 | 76-99 | 51.2 | 240 | 41.5 | 1860 | 1858 | 9 |
| j1/base_t2 | libero_10 | 10 steps, replan 5 | 2-3 | 17/20 | 85 | 64-95 | 57.8 | 246 | 45.3 | 1865 | 1863 | 9.4 |
| j2/steps1 | libero_10 | 1 step, replan 5 | 0-1 | 19/20 | 95 | 76-99 | 50.5 | 237 | 33.8 | 1586 | 1584 | 9.3 |
| j2/steps3 | libero_10 | 3 steps, replan 5 | 0-1 | 20/20 | 100 | 84-100 | 50.4 | 250 | 37.3 | 1633 | 1631 | 9 |
| j3/replan10 | libero_10 | 10 steps, replan 10 | 0-1 | 19/20 | 95 | 76-99 | 27.6 | 258 | 23.2 | 1913 | 1911 | 9 |
| j3/steps3_replan10 | libero_10 | 3 steps, replan 10 | 0-1 | 20/20 | 100 | 84-100 | 25.9 | 255 | 21.7 | 1685 | 1684 | 9.1 |
| smoke/base | libero_spatial | 10 steps, replan 5 | 0-0 | 9/10 | 90 | 60-98 | 24.3 | 108 | 11.3 | 2118 | 2116 | 8.7 |
