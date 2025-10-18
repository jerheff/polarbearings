"""Performance benchmarks for polarbear metrics."""

import time
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score, log_loss as sklearn_log_loss, brier_score_loss

from polarbear import roc_auc, log_loss, brier_score


def benchmark_roc_auc(n_samples: int, n_iterations: int = 10):
    """Benchmark ROC AUC calculation."""
    np.random.seed(42)
    labels = np.random.randint(0, 2, n_samples)
    scores = labels * 0.6 + np.random.randn(n_samples) * 0.3

    df = pl.DataFrame({"label": labels, "score": scores})

    # Benchmark polarbear
    start = time.perf_counter()
    for _ in range(n_iterations):
        result = df.select(roc_auc("label", "score")).to_series()[0]
    polarbear_time = (time.perf_counter() - start) / n_iterations

    # Benchmark sklearn
    start = time.perf_counter()
    for _ in range(n_iterations):
        sklearn_result = roc_auc_score(labels, scores)
    sklearn_time = (time.perf_counter() - start) / n_iterations

    return polarbear_time, sklearn_time, result, sklearn_result


def benchmark_log_loss(n_samples: int, n_iterations: int = 10):
    """Benchmark log loss calculation."""
    np.random.seed(42)
    labels = np.random.randint(0, 2, n_samples)
    probs = np.random.rand(n_samples)

    df = pl.DataFrame({"label": labels, "prob": probs})

    # Benchmark polarbear
    start = time.perf_counter()
    for _ in range(n_iterations):
        result = df.select(log_loss("label", "prob")).to_series()[0]
    polarbear_time = (time.perf_counter() - start) / n_iterations

    # Benchmark sklearn
    start = time.perf_counter()
    for _ in range(n_iterations):
        sklearn_result = sklearn_log_loss(labels, probs)
    sklearn_time = (time.perf_counter() - start) / n_iterations

    return polarbear_time, sklearn_time, result, sklearn_result


def benchmark_brier_score(n_samples: int, n_iterations: int = 10):
    """Benchmark Brier score calculation."""
    np.random.seed(42)
    labels = np.random.randint(0, 2, n_samples)
    probs = np.random.rand(n_samples)

    df = pl.DataFrame({"label": labels, "prob": probs})

    # Benchmark polarbear
    start = time.perf_counter()
    for _ in range(n_iterations):
        result = df.select(brier_score("label", "prob")).to_series()[0]
    polarbear_time = (time.perf_counter() - start) / n_iterations

    # Benchmark sklearn
    start = time.perf_counter()
    for _ in range(n_iterations):
        sklearn_result = brier_score_loss(labels, probs)
    sklearn_time = (time.perf_counter() - start) / n_iterations

    return polarbear_time, sklearn_time, result, sklearn_result


def benchmark_grouped_metrics(n_groups: int, samples_per_group: int, n_iterations: int = 10):
    """Benchmark grouped metric calculations."""
    np.random.seed(42)

    groups = []
    labels = []
    scores = []
    probs = []

    for i in range(n_groups):
        group_labels = np.random.randint(0, 2, samples_per_group)
        group_scores = group_labels * 0.6 + np.random.randn(samples_per_group) * 0.3
        group_probs = np.random.rand(samples_per_group)

        groups.extend([f"group_{i}"] * samples_per_group)
        labels.extend(group_labels)
        scores.extend(group_scores)
        probs.extend(group_probs)

    df = pl.DataFrame({
        "group": groups,
        "label": labels,
        "score": scores,
        "prob": probs,
    })

    # Benchmark polarbear grouped operations
    start = time.perf_counter()
    for _ in range(n_iterations):
        result = df.group_by("group").agg(
            roc_auc("label", "score"),
            log_loss("label", "prob"),
            brier_score("label", "prob"),
        )
    polarbear_time = (time.perf_counter() - start) / n_iterations

    return polarbear_time, n_groups * samples_per_group


def main():
    """Run all benchmarks."""
    print("=" * 80)
    print("Polarbear Performance Benchmarks")
    print("=" * 80)
    print()

    # ROC AUC benchmarks
    print("ROC AUC Performance:")
    print("-" * 80)
    for n in [100, 1_000, 10_000, 100_000]:
        iterations = 100 if n <= 10_000 else 10
        pb_time, sk_time, pb_result, sk_result = benchmark_roc_auc(n, iterations)
        speedup = sk_time / pb_time
        print(f"n={n:>7,}: polarbear={pb_time*1000:>7.3f}ms  sklearn={sk_time*1000:>7.3f}ms  speedup={speedup:>5.2f}x")
        # Verify correctness
        assert abs(pb_result - sk_result) < 1e-5, f"Mismatch: {pb_result} vs {sk_result}"
    print()

    # Log Loss benchmarks
    print("Log Loss Performance:")
    print("-" * 80)
    for n in [100, 1_000, 10_000, 100_000]:
        iterations = 100 if n <= 10_000 else 10
        pb_time, sk_time, pb_result, sk_result = benchmark_log_loss(n, iterations)
        speedup = sk_time / pb_time
        print(f"n={n:>7,}: polarbear={pb_time*1000:>7.3f}ms  sklearn={sk_time*1000:>7.3f}ms  speedup={speedup:>5.2f}x")
        # Verify correctness
        assert abs(pb_result - sk_result) < 1e-5, f"Mismatch: {pb_result} vs {sk_result}"
    print()

    # Brier Score benchmarks
    print("Brier Score Performance:")
    print("-" * 80)
    for n in [100, 1_000, 10_000, 100_000]:
        iterations = 100 if n <= 10_000 else 10
        pb_time, sk_time, pb_result, sk_result = benchmark_brier_score(n, iterations)
        speedup = sk_time / pb_time
        print(f"n={n:>7,}: polarbear={pb_time*1000:>7.3f}ms  sklearn={sk_time*1000:>7.3f}ms  speedup={speedup:>5.2f}x")
        # Verify correctness
        assert abs(pb_result - sk_result) < 1e-5, f"Mismatch: {pb_result} vs {sk_result}"
    print()

    # Grouped operations benchmarks
    print("Grouped Operations Performance:")
    print("-" * 80)
    for n_groups, samples_per_group in [(10, 1000), (100, 1000), (1000, 100), (100, 10000)]:
        iterations = 10
        pb_time, total_samples = benchmark_grouped_metrics(n_groups, samples_per_group, iterations)
        print(f"groups={n_groups:>4}  samples/group={samples_per_group:>5}  total={total_samples:>7,}  time={pb_time*1000:>7.2f}ms")
    print()

    print("=" * 80)
    print("All benchmarks completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
