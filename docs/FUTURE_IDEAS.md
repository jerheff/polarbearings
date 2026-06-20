# Future Ideas

A parking lot for metrics and model-evaluation utilities that fit the
"native Polars expression" design but are not yet implemented. Each entry notes
the scikit-learn analog (where one exists), why it fits, and the main
implementation wrinkle.

## Cross-validation fold assignment (high interest)

Assign deterministic CV fold ids to a frame entirely in-engine — no pandas
round-trip, scales to 100M+ rows. Reuses the row-hash machinery already built for
the Bayesian bootstrap (`bootstrap.py`).

- **`kfold_assign(n_splits, *, seed) -> pl.Expr`** — returns a fold id in
  `0..n_splits-1` per row. Shuffled assignment via `hash(row_index, seed) % n_splits`
  (or rank-based for exactly balanced fold sizes). Mirrors
  `sklearn.model_selection.KFold(shuffle=True)` semantics (fold *membership*, not
  the train/test index pairs).
- **`stratified_kfold_assign(target, n_splits, *, seed) -> pl.Expr`** — balances
  class proportions within each fold. Implementable as a per-class rank modulo
  `n_splits` (`rank().over(target) % n_splits`), optionally hashed for shuffling.
  Mirrors `StratifiedKFold`.
- **`group_kfold_assign(group, n_splits) -> pl.Expr`** — keeps all rows of a group
  in the same fold (`hash(group) % n_splits`). Mirrors `GroupKFold`.

Wrinkles: exact fold-size balancing under hashing needs a rank-based fallback;
shuffled-vs-sorted assignment should be a documented choice. A `train_test_split`
flavor (`hash(row) < frac`) is a trivial special case worth shipping alongside.

## Deferred metrics

- **`class_likelihood_ratios`** (LR+ = TPR/FPR, LR− = FNR/TNR) — free from the
  boolean confusion components; standard in diagnostic/medical evaluation.
  Mirrors `sklearn.metrics.class_likelihood_ratios`. Deferred only on priority.
- **`top_k_accuracy_score`** — fraction of rows whose true label is among the
  top-k scores. Trivial for binary; the multiclass form needs wide (one-column-
  per-class) score input, which is a different data shape than the rest of the
  library. Mirrors `sklearn.metrics.top_k_accuracy_score`.
- **`det_curve`** — Detection Error Tradeoff (FNR vs FPR across thresholds).
  Cheap to derive from the existing `threshold_sweep` components. Mirrors
  `sklearn.metrics.det_curve`.
- **`hinge_loss`** — `mean(max(0, 1 - margin))`. Trivial elementwise, but niche
  (SVM-specific). Mirrors `sklearn.metrics.hinge_loss`.
- **`d2_log_loss_score`** — D² with a log-loss deviance and the class-prior
  baseline; rounds out the D² family. Mirrors `sklearn.metrics.d2_log_loss_score`.

## Multilabel ranking (lower priority)

`coverage_error`, `label_ranking_average_precision_score`, `label_ranking_loss` —
operate on the wide multilabel shape; defer until there is demand for multilabel
support generally.

## Cost curves / threshold economics

Already expressible today by applying per-cell costs to `confusion_matrix` across
a `threshold_sweep` (see the diagnostics notebook). If a recurring pattern
emerges, a thin `expected_cost(costs, thresholds)` helper could wrap it.
