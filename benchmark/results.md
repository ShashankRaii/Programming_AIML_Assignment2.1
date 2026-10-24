# Benchmark results

Best of repeated runs (seconds per call), seed 42.

| Operation | n | Loop (s) | Vectorised (s) | Speed-up |
| --- | ---: | ---: | ---: | ---: |
| StandardScaler | 1,000 | 0.00173 | 0.00022 | 7.9x |
| topk (k=10) | 1,000 | 0.00006 | 0.00004 | 1.4x |
| rank (average) | 1,000 | 0.00029 | 0.00019 | 1.5x |
| Welford mean/var | 1,000 | 0.00020 | 0.00005 | 4.4x |
| euclidean_distances (n x 10) | 1,000 | 0.00736 | 0.00010 | 70.7x |
| StandardScaler | 10,000 | 0.01933 | 0.00436 | 4.4x |
| topk (k=10) | 10,000 | 0.00087 | 0.00009 | 9.6x |
| rank (average) | 10,000 | 0.00429 | 0.00140 | 3.1x |
| Welford mean/var | 10,000 | 0.00216 | 0.00008 | 27.7x |
| euclidean_distances (n x 10) | 10,000 | 0.10822 | 0.00467 | 23.2x |
| StandardScaler | 100,000 | 0.22679 | 0.03964 | 5.7x |
| topk (k=10) | 100,000 | 0.00518 | 0.00155 | 3.3x |
| rank (average) | 100,000 | 0.07662 | 0.01972 | 3.9x |
| Welford mean/var | 100,000 | 0.01053 | 0.00189 | 5.6x |
| euclidean_distances (n x 10) | 100,000 | 0.82591 | 0.03183 | 25.9x |
| grad (central) | 100 | 0.00170 | 0.00023 | 7.4x |
| grad (central) | 500 | 0.01535 | 0.01424 | 1.1x |
| grad (central) | 2,000 | 0.21184 | 0.27045 | 0.8x |

Environment: python 3.14.5, numpy 2.5.4, os Windows 11, machine AMD64, processor Intel64 Family 6 Model 170 Stepping 4, GenuineIntel, cpu_count 18
