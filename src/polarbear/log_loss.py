"""Log loss metric implemented as a Polars expression."""

import polars as pl


def log_loss(target: str, prob: str, eps: float = 1e-15, weight: str | None = None) -> pl.Expr:
    """Compute log loss (binary cross-entropy) for binary classification.

    Args:
        target: Name of the column containing binary labels (0 or 1).
        prob: Name of the column containing predicted probabilities [0, 1].
        eps: Small constant to clip probabilities for numerical stability.
        weight: Optional name of the column containing sample weights.

    Returns:
        A Polars expression that computes the log loss.

    Notes:
        - Lower is better (0 is perfect).
        - Heavily penalizes confident wrong predictions.
        - Probabilities are clipped to [eps, 1-eps] for numerical stability.
    """
    target_float = pl.col(target).cast(pl.Float64)

    prob_clipped = pl.col(prob).clip(eps, 1 - eps)
    log_prob = prob_clipped.log()
    log_1_minus_prob = (1 - prob_clipped).log()

    per_sample = -(target_float * log_prob + (1 - target_float) * log_1_minus_prob)

    if weight is not None:
        w = pl.col(weight).cast(pl.Float64)
        loss = (per_sample * w).sum() / w.sum()
    else:
        loss = per_sample.mean()

    alias = f"log_loss_{target}_{prob}"
    if weight is not None:
        alias += f"_{weight}"
    return loss.alias(alias)
