# Benchmark results

Best of 5 timed rounds (seconds per call); one call per round, one untimed warmup and an output equality check per implementation. Seed 42.

Inputs and Python-list conversion are prepared before timing. CSV includes mean/std across rounds. Gradient rows compare ordinary vs batched callable evaluation, with the same NumPy objective in both versions. Smaller gradient dimensions bound the quadratic perturbation storage; distances use 10 reference points.

| Operation | n | Loop (s) | Vectorised (s) | Speed-up |
| --- | ---: | ---: | ---: | ---: |
| StandardScaler | 1,000 | 0.00116 | 0.00016 | 7.5x |
| topk (k=10) | 1,000 | 0.00004 | 0.00001 | 3.1x |
| rank (average) | 1,000 | 0.00017 | 0.00005 | 3.5x |
| Welford mean/var | 1,000 | 0.00007 | 0.00001 | 5.9x |
| euclidean_distances (n x 10) | 1,000 | 0.00338 | 0.00049 | 7.0x |
| StandardScaler | 10,000 | 0.01186 | 0.00107 | 11.1x |
| topk (k=10) | 10,000 | 0.00038 | 0.00012 | 3.1x |
| rank (average) | 10,000 | 0.00199 | 0.00067 | 3.0x |
| Welford mean/var | 10,000 | 0.00069 | 0.00002 | 28.6x |
| euclidean_distances (n x 10) | 10,000 | 0.03548 | 0.00498 | 7.1x |
| StandardScaler | 100,000 | 0.13700 | 0.01071 | 12.8x |
| topk (k=10) | 100,000 | 0.00376 | 0.00102 | 3.7x |
| rank (average) | 100,000 | 0.02721 | 0.00858 | 3.2x |
| Welford mean/var | 100,000 | 0.00694 | 0.00016 | 43.4x |
| euclidean_distances (n x 10) | 100,000 | 0.37234 | 0.04781 | 7.8x |
| grad (central) | 100 | 0.00060 | 0.00010 | 6.2x |
| grad (central) | 500 | 0.00463 | 0.00217 | 2.1x |
| grad (central) | 2,000 | 0.04742 | 0.04205 | 1.1x |

Environment: python 3.11.13, numpy 2.4.6, os Darwin 27.0.0, machine arm64, processor arm, cpu_count 8
