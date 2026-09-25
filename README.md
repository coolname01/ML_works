# Hidden pairs: predicting shared latent identity with tree ensembles

Each row of the competition data describes a pair of hidden observations through
512 anonymous numerical features (`f0000` … `f0511`); `target = 1` means the two
observations share the same latent identity. Submissions are scored with
**ROC-AUC** (see `baseline_student-ru.ipynb`), so the model outputs the
probability of the positive class rather than a hard label.

## Data

| file | rows | columns | positives |
|---|---|---|---|
| `train.csv` | 2 520 | `row_id`, 512 features, `target` | 50.0 % |
| `validation.csv` | 560 | `row_id`, 512 features, `target` | 50.0 % |
| `test.csv` | 1 116 | `row_id`, 512 features | – |
| `sample_submission.csv` | 1 116 | `row_id`, `target` | – |

There are no missing values, all features are `float64`, `row_id`s do not
overlap between files, and the `row_id` order of `test.csv` matches
`sample_submission.csv`. The validation split is separated from the training
data at the level of the hidden entities, so it is used as the only local
checkpoint; no random re-splitting or merging of the provided files is done.

What exploration revealed about the features:

* Every feature is non-negative and **all 512 features have a univariate
  AUC below 0.5** (0.30–0.42): larger values always point towards "different
  identity". Each column therefore behaves like a per-dimension distance
  between the two hidden observations. There is no `a_i` / `b_i` half
  structure (the correlation between `f_j` and `f_{j+256}` is no higher than
  between arbitrary columns), so classical pairwise features (differences,
  products, cosine similarity of two halves) do not apply.
* The plain L1 distance (row sum) alone reaches validation AUC 0.833, versus
  0.747 for the notebook's depth-3 decision tree on raw columns.
* Dimensions are far from equally informative and are not conditionally
  independent: naive-Bayes-style sums of per-dimension evidence stay at
  ≈ 0.84 while discriminative models reach > 0.91. Imposing "monotone
  decreasing" constraints on every feature hurt all boosting libraries by
  ≈ 0.03 AUC, i.e. conditional on the overall distance some dimensions act as
  *nuisance* dimensions (a large difference there makes a same-identity pair
  more likely). Trees learn this on their own.
* Per-row normalised profiles (`f_i / L1`) and within-row ranks added nothing
  beyond noise, so they were left out.

## Feature engineering

Implemented in `PairDistanceFeatures` (`train_model.py`). Everything that is
learned is fitted on `train.csv` only and then applied to validation and test.

1. **Raw features** – the 512 distance columns, unchanged.
2. **Unsupervised row aggregates** (16 columns) describing the distance
   profile of the pair: L1 and L2 norms, standard deviation, maximum,
   quartiles and the 90th percentile, counts of dimensions below 0.1 / 0.5
   and above 1 / 2, sum of logs, sum of square roots, mean of the 16 largest
   and the 16 smallest values.
3. **Train-fitted discriminability groups** (14 columns). Dimensions are
   ranked by their univariate ROC-AUC on the training rows. For
   `k ∈ {32, 64, 128, 256}` the sums over the `k` most and `k` least
   discriminative dimensions and their ratio are added, plus weighted sums
   with weights `(0.5 − AUC_i)^p`, `p ∈ {2, 4}`. The top-64 sum alone scores
   0.870 on validation (L1: 0.833), and these columns lift LightGBM from
   0.923 to 0.929 and HistGradientBoosting from 0.921 to 0.930 (same
   hyper-parameters), because shallow trees cannot approximate a good
   weighted distance with few splits on their own.

Final matrix: 542 columns.

## Models and validation results

All models are fitted on `train.csv` only and scored by ROC-AUC on
`validation.csv`. Hyper-parameter variation was kept to a small hand-picked
grid per family to avoid over-fitting the 560-row validation set. Blends are
plain probability averages of the best configuration of each family. Logistic
regression is listed only as the reference asked for by the notebook's
task 8; it was never a candidate for selection.

| model | family | features | validation AUC | fit time |
|---|---|---|---|---|
| **Blend(LightGBM + HistGradientBoosting + CatBoost + ExtraTrees)** | Blend | engineered | **0.9338** | 32.6 s |
| Blend(LightGBM + HistGradientBoosting + CatBoost) | Blend | engineered | 0.9332 | 27.8 s |
| Blend(LightGBM + HistGradientBoosting + XGBoost + CatBoost + ExtraTrees + RandomForest) | Blend | engineered | 0.9321 | 70.8 s |
| Blend(LightGBM + HistGradientBoosting) | Blend | engineered | 0.9320 | 10.3 s |
| LightGBM lr=0.05 n=400 leaves=31 colsample=0.5 | LightGBM | engineered | 0.9314 | 2.7 s |
| LightGBM lr=0.02 n=1000 leaves=15 colsample=0.3 | LightGBM | engineered | 0.9305 | 2.5 s |
| HistGradientBoosting lr=0.05 iter=400 leaves=31 | HistGradientBoosting | engineered | 0.9300 | 7.6 s |
| LightGBM lr=0.03 n=800 leaves=15 colsample=0.2 | LightGBM | engineered | 0.9299 | 1.4 s |
| CatBoost depth=4 lr=0.03 iter=1500 | CatBoost | engineered | 0.9297 | 17.5 s |
| ExtraTrees 800 trees max_features=0.2 | ExtraTrees | engineered | 0.9292 | 4.8 s |
| LightGBM lr=0.03 n=600 leaves=15 colsample=0.5 | LightGBM | engineered | 0.9291 | 2.1 s |
| XGBoost depth=5 lr=0.03 n=600 | XGBoost | engineered | 0.9290 | 4.9 s |
| XGBoost depth=3 lr=0.03 n=600 | XGBoost | engineered | 0.9272 | 3.3 s |
| HistGradientBoosting lr=0.03 iter=800 leaves=15 | HistGradientBoosting | engineered | 0.9262 | 8.6 s |
| CatBoost depth=6 iter=1000 (defaults) | CatBoost | engineered | 0.9248 | 27.1 s |
| LightGBM lr=0.03 n=600 leaves=7 colsample=0.3 | LightGBM | engineered | 0.9230 | 0.9 s |
| RandomForest 800 trees max_features=0.2 | RandomForest | engineered | 0.9217 | 33.2 s |
| ExtraTrees 800 trees max_features=sqrt | ExtraTrees | engineered | 0.9214 | 1.9 s |
| HistGradientBoosting lr=0.05 iter=300 leaves=7 | HistGradientBoosting | engineered | 0.9200 | 1.8 s |
| RandomForest 800 trees max_features=sqrt | RandomForest | engineered | 0.9178 | 6.4 s |
| LogisticRegression C=0.01 (reference only) | LogisticRegression | engineered | 0.9175 | 0.4 s |
| DecisionTree depth=3 leaf=20 | DecisionTree | engineered | 0.8645 | 0.4 s |
| DecisionTree depth=6 leaf=20 | DecisionTree | engineered | 0.8513 | 0.6 s |
| DecisionTree depth=3 leaf=20 (notebook baseline) | DecisionTree | raw | 0.7473 | 0.3 s |

The same table is written to `validation_results.csv` by the script.

### Chosen model

**Probability average of the best LightGBM, HistGradientBoosting, CatBoost and
ExtraTrees configurations, all fitted on `train.csv` only** — validation
ROC-AUC **0.9338** (baseline tree: 0.7473).

Why this one:

* It has the highest validation AUC, and blending four differently built
  ensembles (leaf-wise boosting, histogram boosting, symmetric-tree ordered
  boosting, randomised bagging) reduces the variance of any single
  configuration, which matters with only 2 520 training rows.
* Every strong single model sits between 0.929 and 0.931. A bootstrap of the
  validation set gives a standard error of ≈ 0.011 AUC, so these models are
  statistically indistinguishable; the blend is preferred for robustness
  rather than for the 0.002–0.004 nominal gain. Picking the best of ~20
  configurations on 560 rows also carries a small optimistic bias, which is
  another reason to favour an average over a single winner.
* Runtime is modest: the whole comparison plus the final prediction takes
  about two minutes on 4 CPU cores.

`submission.csv` was produced by this blend fitted on `train.csv` only, in
line with the notebook's rule not to merge the provided splits. Refitting the
selected configuration on train + validation before predicting test is
available through `--refit-on-train-val` but is off by default; when used,
the validation score above still refers to the train-only fit and can no
longer be re-measured for the refitted models.

## How to run

```bash
python3 -m pip install -r requirements.txt
python3 train_model.py            # from the directory containing the four CSV files
```

Options: `--data-dir DIR` (default `.`), `--submission PATH` (default
`submission.csv`), `--results PATH` (default `validation_results.csv`),
`--refit-on-train-val` (see above). All seeds are fixed (`SEED = 20260916`);
repeated runs produce a byte-identical `submission.csv`.

The script prints the shapes and class balance, the validation AUC of every
candidate, the selected configuration, and validates the submission (two
columns, 1 116 rows, `row_id` order identical to `sample_submission.csv`,
unique ids, no missing values, probabilities in `[0, 1]`).

## Files

| file | purpose |
|---|---|
| `train_model.py` | full pipeline: load → features → validation comparison → final fit → `submission.csv` |
| `requirements.txt` | pinned library versions |
| `submission.csv` | test predictions of the selected model (1 116 rows) |
| `validation_results.csv` | results table produced by the last run |
| `baseline_student-ru.ipynb` | original baseline notebook (unchanged) |
| `train.csv`, `validation.csv`, `test.csv`, `sample_submission.csv` | competition data (unchanged) |
