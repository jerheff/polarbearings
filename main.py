"""Example usage of polarbear metrics."""

import polars as pl

from polarbear import brier_score, log_loss, roc_auc


def main() -> None:
    """Demonstrate polarbear metrics."""
    print("=" * 60)
    print("Polarbear: High-performance metrics for Polars DataFrames")
    print("=" * 60)

    # Create example data with realistic (imperfect) predictions
    df = pl.DataFrame(
        {
            "actual": [0, 0, 0, 1, 1, 1, 1, 0],
            "score": [0.2, 0.3, 0.6, 0.55, 0.7, 0.8, 0.9, 0.4],
            "probability": [0.2, 0.3, 0.6, 0.55, 0.7, 0.8, 0.9, 0.4],
            "segment": ["A", "A", "B", "B", "A", "A", "B", "B"],
        }
    )

    print("\nExample DataFrame:")
    print(df)

    # Calculate all metrics
    print("\n" + "=" * 60)
    print("Overall Metrics")
    print("=" * 60)
    result = df.select(
        roc_auc("actual", "score"),
        log_loss("actual", "probability"),
        brier_score("actual", "probability"),
    )
    print(result)

    # Group-wise metrics
    print("\n" + "=" * 60)
    print("Metrics by Segment")
    print("=" * 60)
    grouped_result = df.group_by("segment").agg(
        roc_auc("actual", "score"),
        log_loss("actual", "probability"),
        brier_score("actual", "probability"),
    )
    print(grouped_result)

    # Perfect classification example
    print("\n" + "=" * 60)
    print("Perfect Classification Example")
    print("=" * 60)
    perfect_df = pl.DataFrame(
        {"label": [0, 0, 1, 1], "score": [0.1, 0.2, 0.8, 0.9]}
    )
    print(perfect_df)
    perfect_result = perfect_df.select(
        roc_auc("label", "score"),
        log_loss("label", "score"),
        brier_score("label", "score"),
    )
    print("\nMetrics for perfect classification:")
    print(perfect_result)


if __name__ == "__main__":
    main()
