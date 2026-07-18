# Future Ideas

A parking lot for metrics and model-evaluation utilities that fit the
"native Polars expression" design but are not yet implemented. Each entry notes
the scikit-learn analog (where one exists), why it fits, and the main
implementation wrinkle. Items that ship graduate to the [Shipped since this list was
written](#shipped-since-this-list-was-written) section at the bottom.

## Cross-validation fold assignment (partly shipped)

The id-keyed building blocks now exist in `split.py`: `hash_uniform`, `hash_split`
(boolean holdout / `train_test_split`), `hash_splits` (named multi-way), and
`hash_fold` (`KFold(shuffle=True)` membership). Exact stratification is documented
as the rank-within-stratum pattern (`u.rank().over(class)`) rather than a parameter.

Still parked, if demand appears:

- **`stratified_kfold_assign(target, n_splits, *, seed)`** — a dedicated helper that
  bakes in the per-class rank-modulo so the user doesn't hand-write the `.over`.
- **`group_kfold_assign(group, n_splits)`** — keep all rows of a group in the same
  fold by hashing the *group* key instead of the row id (`hash_fold("group", k)`
  already does exactly this; a named alias would document the intent).

## Deferred metrics

- **`class_likelihood_ratios`** (LR+ = TPR/FPR, LR− = FNR/TNR) — free from the
  boolean confusion components; standard in diagnostic/medical evaluation.
  Mirrors `sklearn.metrics.class_likelihood_ratios`. Deferred only on priority.
- **`top_k_accuracy_score`** — fraction of rows whose true label is among the
  top-k scores. Trivial for binary; the multiclass form needs wide (one-column-
  per-class) score input, which is a different data shape than the rest of the
  library. Mirrors `sklearn.metrics.top_k_accuracy_score`.
- **`hinge_loss`** — `mean(max(0, 1 - margin))`. Trivial elementwise, but niche
  (SVM-specific). Mirrors `sklearn.metrics.hinge_loss`.
- **`d2_log_loss_score`** — D² with a log-loss deviance and the class-prior
  baseline; rounds out the D² family. Mirrors `sklearn.metrics.d2_log_loss_score`.

## Multilabel ranking (lower priority)

`coverage_error`, `label_ranking_average_precision_score`, `label_ranking_loss` —
operate on the wide multilabel shape; defer until there is demand for multilabel
support generally.

## Shipped since this list was written

`det_curve`, `expected_cost` (cost curves), `roc_curve`/`pr_curve`, the
`confusion_curve` primitive with a `thresholds=` grid, `calibration_curve` (incl.
`by=`), and the bootstrap helpers (`bootstrap_ci`, `bootstrap_weight`) are all
implemented — removed from the lists above.
