# Experiment Registry

This registry records evaluated directions so that rejected approaches are not
repeated. The current reference is XGBoost A+B with time-aware franchise
history within the defined pre-release operational scope: pooled outer-OOF
Macro-F1 **0.719483** on 1,646 movies.

## Required protocol

- Target: `revenue >= 2 × budget`.
- `StratifiedKFold`: five outer folds with seed 42 and four inner folds with seed 43.
- Primary metric: Macro-F1.
- Preprocessing, feature state, tuning, and threshold selection are fit within training or inner-validation data.
- Outer-fold results are not used to select a configuration.
- `revenue`, popularity, ratings, votes, profit, ROI, and post-release variables are prohibited predictors.

## Reference chronology

| Milestone | Scope | Macro-F1 | Conclusion |
| --- | --- | ---: | --- |
| Earlier extended XGBoost | Included post-release signals | 0.7597 | Not valid for the final research question |
| Earlier pre-release XGBoost | Earlier scope, not fully verified | 0.7144 | Historical milestone, not the current reference |
| A+B fixed | Operational pre-release scope | 0.710977 | Official control configuration |
| Operational Budget Context V2 | Operational pre-release scope | 0.7087 | Did not exceed the control |
| A+B with franchise history | Operational pre-release scope | **0.719483** | Current reference |

## Rejected directions

### Post-release signals

Popularity, ratings, and vote counts improved some historical scores but are not
reliably available before release. They remain excluded from the final model.

### Hyperparameters and threshold

Grid search and Optuna evaluated learning rate, depth, child weight,
subsampling, regularization, class weighting, boosting rounds, and the
classification threshold. Broader search did not produce a stable improvement
over the locked A+B configuration. The same search space should not be repeated
without new features or data.

### Content and metadata representations

TF-IDF/SVD overview features, keyword identity, budget/genre interactions,
alternative categorical encodings, and expanded metadata groups did not improve
inner-validation performance consistently enough to justify their complexity.

### Operational Budget Context V1

The seven budget-context features improved mean inner Macro-F1 by approximately
**0.001261**, below the +0.003 screening criterion. They are not part of the
reference configuration.

### Entity History V1

Time-aware history from the available snapshot did not pass the inner-validation
screening gate:

| Entity group | Inner Macro-F1 difference vs. control |
| --- | ---: |
| Top-billed cast | -0.001900 |
| Director | -0.002840 |
| Production company | -0.002957 |

The distributor branch was blocked because no source with reliable provenance
and identifiers was available. TMDb `production_companies` is not treated as a
distributor. Reassessment requires broader historical coverage and a clearer
hypothesis.

### Out-of-scope models

k-NN, Logistic Regression, Random Forest, and CatBoost were surveyed but removed
when the project scope was fixed to XGBoost. They are not part of the final
workflow.

## Conditions for new experiments

A new experiment must state a distinct hypothesis, new data or features, leakage
controls, inner-validation screening criterion, computational cost, and separate
results. It must not overwrite the reference model or aggregate tables. Outer
evaluation is performed once only after the feature set is locked through inner
validation.
