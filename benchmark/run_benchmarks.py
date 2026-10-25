"""Loop vs vectorised benchmarks for NumCompute.

Loop implementations are compared with NumCompute on the same data. For
gradients, the objective uses NumPy in both cases; only the callable dispatch
changes from per-coordinate evaluations to one batch. Outputs are checked for
agreement before timing. Input/list preparation is outside the timed calls.
"""

import argparse
import bisect
import csv
import math
from pathlib import Path

import numpy as np

from numcompute.benchmarking import SEED, compare, environment_info
from numcompute.optim import grad
from numcompute.preprocessing import StandardScaler
from numcompute.rank import rank
from numcompute.sort_search import topk
from numcompute.stats import RunningStats
from numcompute.utils import euclidean_distances

HERE = Path(__file__).resolve().parent
N_FEATURES = 5  # columns for the scaler benchmark
N_CENTROIDS = 10  # rows of B for the distance benchmark
K = 10  # k for top-k


# Pure-Python loop baselines (deliberately loop-based for comparison)

def loop_standard_scaler(rows):
    n, d = len(rows), len(rows[0])
    means = [0.0] * d
    for row in rows:
        for j in range(d):
            means[j] += row[j]
    means = [m / n for m in means]
    variances = [0.0] * d
    for row in rows:
        for j in range(d):
            variances[j] += (row[j] - means[j]) ** 2
    scales = [math.sqrt(v / n) or 1.0 for v in variances]
    return [[(row[j] - means[j]) / scales[j] for j in range(d)] for row in rows]


def loop_topk(values, k):
    best = []  # ascending list of the k largest values seen so far
    for v in values:
        if len(best) < k:
            bisect.insort(best, v)
        elif v > best[0]:
            best.pop(0)
            bisect.insort(best, v)
    return best[::-1]


def loop_rank_average(values):
    n = len(values)
    order = sorted(range(n), key=values.__getitem__)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2 + 1  # ranks start at 1
        for idx in order[i : j + 1]:
            ranks[idx] = average
        i = j + 1
    return ranks


def loop_welford(values):
    count, mean, m2 = 0, 0.0, 0.0
    for x in values:
        count += 1
        delta = x - mean
        mean += delta / count
        m2 += delta * (x - mean)
    return [mean, m2 / count]


def loop_distances(A, B):
    out = []
    for a in A:
        row = []
        for b in B:
            total = 0.0
            for p, q in zip(a, b):
                total += (p - q) ** 2
            row.append(math.sqrt(total))
        out.append(row)
    return out


def objective(v):
    """f(x) = sum(x^2 + sin x); works on shape (n,) or a stack (k, n)."""
    return np.sum(v**2 + np.sin(v), axis=-1)

# Benchmark cases

def make_cases(n, rng):
    """Return {operation: (loop_callable, vectorised_callable)} for size n."""
    X = rng.normal(size=(n, N_FEATURES))
    x = rng.normal(size=n)
    ties = rng.integers(0, max(n // 10, 1), size=n).astype(float)  # many ties
    A = rng.normal(size=(n, 3))
    B = rng.normal(size=(N_CENTROIDS, 3))

    # Python lists for the loop versions, prepared outside the timed call.
    X_list, x_list, ties_list = X.tolist(), x.tolist(), ties.tolist()
    A_list, B_list = A.tolist(), B.tolist()

    return {
        "StandardScaler": (
            lambda: loop_standard_scaler(X_list),
            lambda: StandardScaler().fit_transform(X),
        ),
        f"topk (k={K})": (
            lambda: loop_topk(x_list, K),
            lambda: topk(x, K, return_indices=False),
        ),
        "rank (average)": (
            lambda: loop_rank_average(ties_list),
            lambda: rank(ties),
        ),
        "Welford mean/var": (
            lambda: loop_welford(x_list),
            lambda: (lambda s: [float(s.mean), float(s.variance)])(RunningStats().update(x)),
        ),
        f"euclidean_distances (n x {N_CENTROIDS})": (
            lambda: loop_distances(A_list, B_list),
            lambda: euclidean_distances(A, B),
        ),
    }


def gradient_case(dim, rng):
    """Gradient in `dim` dimensions: one call per coordinate vs one stacked call."""
    point = rng.normal(size=dim)
    return (
        lambda: grad(objective, point),
        lambda: grad(objective, point, vectorized=True),
    )


def run(sizes, grad_sizes, repeats):
    rng = np.random.default_rng(SEED)
    results = []

    def record(operation, n, loop_fn, vec_fn):
        rows = compare({"loop": loop_fn, "vectorised": vec_fn}, repeats=repeats, number=1)
        by_name = {row["name"]: row for row in rows}
        loop_s, vec_s = by_name["loop"]["min"], by_name["vectorised"]["min"]
        results.append(
            {
                "operation": operation,
                "n": n,
                "loop_s": loop_s,
                "vectorised_s": vec_s,
                "speedup": loop_s / vec_s if vec_s > 0 else float("inf"),
                "loop_mean_s": by_name["loop"]["mean"],
                "loop_std_s": by_name["loop"]["std"],
                "vectorised_mean_s": by_name["vectorised"]["mean"],
                "vectorised_std_s": by_name["vectorised"]["std"],
            }
        )
        print(
            f"{operation:<34} n={n:>7,}  loop={loop_s:9.5f}s  "
            f"vec={vec_s:9.5f}s  x{results[-1]['speedup']:,.1f}"
        )

    for n in sizes:
        for operation, (loop_fn, vec_fn) in make_cases(n, rng).items():
            record(operation, n, loop_fn, vec_fn)
    for dim in grad_sizes:
        record("grad (central)", dim, *gradient_case(dim, rng))
    return results


def write_outputs(results, out_dir, repeats):
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["operation", "n", "loop_s", "vectorised_s", "speedup",
              "loop_mean_s", "loop_std_s", "vectorised_mean_s", "vectorised_std_s"]
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in results:
            writer.writerow(
                {k: (f"{row[k]:.6g}" if isinstance(row[k], float) else row[k]) for k in fields}
            )

    env = environment_info()
    lines = [
        "# Benchmark results",
        "",
        f"Best of {repeats} timed rounds (seconds per call); one call per round, "
        "one untimed warmup and an output equality check per implementation. Seed 42.",
        "",
        "Inputs and Python-list conversion are prepared before timing. CSV includes "
        "mean/std across rounds. Gradient rows compare ordinary vs batched callable "
        "evaluation, with the same NumPy objective in both versions. Smaller gradient "
        "dimensions bound the quadratic perturbation storage; distances use 10 reference points.",
        "",
        "| Operation | n | Loop (s) | Vectorised (s) | Speed-up |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            f"| {row['operation']} | {row['n']:,} | {row['loop_s']:.5f} | "
            f"{row['vectorised_s']:.5f} | {row['speedup']:,.1f}x |"
        )
    lines += ["", "Environment: " + ", ".join(f"{k} {v}" for k, v in env.items()), ""]
    (out_dir / "results.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="only n = 1,000 and 10,000")
    parser.add_argument("--repeats", type=int, default=3, help="timed rounds per implementation")
    parser.add_argument("--out", type=Path, default=HERE, help="output folder")
    args = parser.parse_args()

    sizes = [1_000, 10_000] if args.quick else [1_000, 10_000, 100_000]
    # grad builds a (2n x n) perturbation matrix, so dimensions stay smaller.
    grad_sizes = [100, 500] if args.quick else [100, 500, 2_000]
    results = run(sizes, grad_sizes, args.repeats)
    write_outputs(results, args.out, args.repeats)
    print(f"\nWrote {args.out / 'results.csv'} and {args.out / 'results.md'}")


if __name__ == "__main__":
    main()
