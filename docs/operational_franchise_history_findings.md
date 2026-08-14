# Time-Aware Franchise History

## Hypothesis

Earlier releases in the same collection may provide information about brand
familiarity and the historical scale of a project. The experiment added
history features to the A+B feature set without changing the sample, target, or
outer-fold partitions.

## Four history features

- `collection_prior_movie_count`;
- smoothed `collection_prior_success_rate`;
- `collection_prior_mean_log_budget`;
- `collection_years_since_previous`.

The history builder is fit separately within each training partition. Only
movies released before the query movie are eligible; validation and test rows
do not update the history state. The target movie's `revenue` is not used as a
predictor.

## Pooled outer-OOF result

| Configuration | Macro-F1 | F1, class 0 | Recall, class 0 | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: |
| A+B fixed | 0.710977 | 0.597194 | **0.624738** | 0.716988 |
| A+B + franchise history | **0.719483** | **0.605128** | 0.618449 | **0.722398** |

Franchise-history features increased Macro-F1 by 0.008505 and class-0 F1 by
0.007934, while class-0 recall decreased by 0.006289. The improvement is
modest but occurs in the primary metric and balanced accuracy; therefore, this
configuration was retained as the reference model. The observed difference
should not be interpreted as causal evidence of a franchise effect.

The public evidence is available in
[`operational_franchise_comparison.csv`](../reports/tables/operational_franchise_comparison.csv)
and [`operational_franchise_metrics.csv`](../reports/tables/operational_franchise_metrics.csv).
Per-movie predictions and outer-fold assignments remain local.
