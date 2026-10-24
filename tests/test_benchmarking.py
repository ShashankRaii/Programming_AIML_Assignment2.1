"""Tests for numcompute.benchmarking."""

import time

import numpy as np
import pytest

from numcompute.benchmarking import SEED, benchmark, compare, environment_info


def test_benchmark_returns_per_call_statistics():
    result = benchmark(sum, [1, 2, 3], repeats=4, number=3)
    assert set(result) == {"mean", "std", "min", "repeats", "number"}
    assert result["repeats"] == 4 and result["number"] == 3
    assert 0 <= result["min"] <= result["mean"]
    assert result["std"] >= 0


def test_benchmark_call_count_includes_warmup():
    calls = []
    benchmark(lambda: calls.append(1), repeats=3, number=4, warmup=2)
    assert len(calls) == 2 + 3 * 4


def test_benchmark_measures_real_time():
    result = benchmark(time.sleep, 0.002, repeats=2, number=2, warmup=0)
    assert result["min"] >= 0.0015


def test_benchmark_validates_arguments():
    with pytest.raises(TypeError, match="callable"):
        benchmark("not a function")
    with pytest.raises(ValueError, match="repeats must be at least 1"):
        benchmark(sum, [1], repeats=0)
    with pytest.raises(ValueError, match="warmup must be at least 0"):
        benchmark(sum, [1], warmup=-1)
    with pytest.raises(TypeError, match="number must be an integer"):
        benchmark(sum, [1], number=2.5)


def test_environment_info_has_versions():
    info = environment_info()
    assert info["numpy"] == np.__version__
    assert info["python"].count(".") == 2
    assert all(isinstance(value, str) for value in info.values())


def test_fixed_seed_is_reproducible():
    a = np.random.default_rng(SEED).normal(size=5)
    b = np.random.default_rng(SEED).normal(size=5)
    np.testing.assert_array_equal(a, b)


def _loop_sum_squares(x):
    total = 0.0
    for value in x:
        total += value * value
    return total


def _vector_sum_squares(x):
    return float(np.dot(x, x))


def test_compare_sorts_fastest_first_with_speedup():
    x = np.random.default_rng(SEED).random(20_000)
    funcs = {"loop": _loop_sum_squares, "vectorised": _vector_sum_squares}
    rows = compare(funcs, x, repeats=3, number=2)
    assert [row["name"] for row in rows] == ["vectorised", "loop"]
    assert rows[-1]["speedup"] == 1.0
    assert rows[0]["speedup"] > 1.0
    assert set(rows[0]) == {"name", "mean", "std", "min", "speedup"}


def test_compare_detects_different_results():
    with pytest.raises(ValueError, match="different result"):
        compare({"a": lambda: 1.0, "b": lambda: 2.0}, repeats=1, number=1)
    rows = compare({"a": lambda: 1.0, "b": lambda: 2.0}, repeats=1, number=1, check_equal=False)
    assert len(rows) == 2


def test_compare_validates_input():
    with pytest.raises(TypeError, match="dict"):
        compare([sum])
    with pytest.raises(ValueError, match="at least one"):
        compare({})
    with pytest.raises(TypeError, match="callable"):
        compare({"bad": 3})
