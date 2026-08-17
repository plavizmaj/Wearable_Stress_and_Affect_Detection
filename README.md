# WESAD: Wearable Stress and Affect Detection

This project implements a complete, reproducible machine learning pipeline that classifies a person's affective state (baseline, stress, amusement) from raw physiological signals recorded by two wearable devices: a RespiBAN chest belt and an Empatica E4 wristband. It works on the public WESAD dataset (15 subjects), converts the raw multi-rate sensor streams into 242 hand-crafted physiological features over 60-second windows, and evaluates several classifiers under a strict Leave-One-Subject-Out (LOSO) protocol.

The problem it addresses is the one that actually blocks deployment of wearable stress detection: physiological baselines differ enormously between people, so a model validated with a random train/test split learns to recognise the *subject* rather than the *state* and collapses on new users. Every design decision here (subject-grouped cross-validation, relative rather than absolute band powers, per-fold refitting of the scaler and imputer, discarding of label-impure transition windows) exists to produce an honest estimate of subject-independent performance. The notebook also quantifies the remaining gap: per-subject accuracy varies from 0.599 to well above 0.90 for the best model.

It is intended for researchers, data scientists, and students working on affective computing, digital health, or biosignal processing (ECG, EDA, respiration, EMG, BVP, accelerometry), and for anyone who needs a worked reference on evaluating physiological models without subject leakage.

## Results

Measured with Leave-One-Subject-Out on all 15 subjects (2038 windows, 242 features):

| Task | Best model | Accuracy | Balanced accuracy | Macro F1 | ROC-AUC |
|---|---|---|---|---|---|
| 3-class (baseline / stress / amusement) | LogisticRegression | 0.867 | 0.850 | 0.831 | - |
| Binary (stress vs non-stress) | XGBoost | 0.940 | 0.939 | - | 0.985 |

The notable finding is the model ranking: regularised logistic regression beats both XGBoost (accuracy 0.855, balanced accuracy 0.769) and Random Forest (0.815 / 0.693) on the 3-class task. With only 15 subjects and hundreds of correlated features, the high-bias linear model has the better inductive bias, while the tree ensembles fit subject-specific patterns and drift towards predicting the majority class. Per-subject accuracy of the best model is 0.867 +/- 0.093, worst case 0.599 (S16).

## Features

**Signal loading and preprocessing**
* Reads the WESAD Python 2 pickles (`S*.pkl`) with `encoding="latin1"` and normalises byte keys to strings so that re-pickled dataset mirrors also load.
* Anti-aliased decimation via `scipy.signal.resample_poly` (an FIR low-pass before decimation, unlike plain slicing).
* Per-channel target sampling rates chosen from the physiology of each signal: ECG and EMG stay at 700 Hz, respiration and accelerometry drop to 70 Hz, EDA and temperature drop to 7 Hz. This reduces feature extraction time and memory by roughly 10x.
* Wrist channels are used at their native E4 rates (ACC 32 Hz, BVP 64 Hz, EDA 4 Hz, TEMP 4 Hz).
* Triaxial accelerometer data is split into x, y, z channels plus an orientation-invariant magnitude channel.
* Subjects are processed one at a time and their raw arrays freed with `gc.collect()`, so peak RAM stays at roughly one subject instead of 15.

**Windowing**
* 60-second windows with a 15-second step (75% overlap), which multiplies the number of training samples by 4.
* A window is kept only if at least 98% of its 700 Hz label samples carry a single class, which discards transition windows straddling two protocol conditions.
* Only labels 1, 2, and 3 (baseline, stress, amusement) are modelled; transient, meditation, and the unused codes are dropped.

**Feature extraction (242 features per window)**
* `stat_features`: mean, std, min, max, range, median, IQR, RMS, MAD, normalised least-squares slope, mean absolute difference, RMS of the first difference, skewness, kurtosis, zero-crossing rate.
* `freq_features`: Welch PSD relative band powers per signal type, peak frequency, spectral centroid, normalised spectral entropy, log total power. Relative rather than absolute powers make the features invariant to sensor gain and subject-specific amplitude.
* `detect_r_peaks`: Pan-Tompkins-style QRS detection (5-20 Hz band-pass, derivative, squaring, 120 ms moving-window integration, adaptive 98th-percentile threshold, 300 ms refractory period).
* `hrv_features`: HR mean and std, RR mean/min/max, SDNN, RMSSD, pNN50, CVNN, beat count, plus frequency-domain HRV (LF 0.04-0.15 Hz, HF 0.15-0.40 Hz, LF/HF ratio) computed from the RR tachogram interpolated onto a 4 Hz grid.
* `eda_features`: tonic skin conductance level statistics plus phasic features after a 0.05 Hz high-pass (SCR count, mean and summed SCR amplitude, phasic std and max). The peak-prominence threshold is floored at twice the median sample-to-sample change, which prevents dozens of spurious SCRs on the quantisation-noisy 4 Hz E4 channel.
* `resp_features`: breathing rate, cycle-length variability, inhalation and exhalation durations, I/E ratio, cycle count, after a 0.6 Hz low-pass that removes cardiac artefact.
* `bvp_features`: pulse rate mean and std, IBI RMSSD, pulse count from a 0.7-3.5 Hz band-passed wrist PPG.
* `_clean`: median-imputes NaN and Inf once at feature level, so the feature table is guaranteed numeric.

**Exploration and diagnostics**
* Per-subject duration of every protocol condition as a sanity-check table and stacked bar chart.
* Full-session multi-channel signal plot with the protocol conditions shaded behind the traces, plus a 20-second ECG zoom per condition.
* Detection of NaN-bearing and constant features, with constant features dropped from the feature list.
* One-way ANOVA F-value ranking of individual features and boxplots of the top 6 per class.
* Two-panel PCA scatter (coloured by condition, then by subject) that shows directly whether subject identity dominates the variance.

**Modelling and evaluation**
* Four pipelines, each `SimpleImputer(median) -> StandardScaler -> classifier`: Logistic Regression, Random Forest, XGBoost, MLP. Putting preprocessing inside the pipeline guarantees it is fitted on training subjects only.
* `class_weight="balanced"` (and `scale_pos_weight` for binary XGBoost) because baseline is roughly twice as frequent as the other conditions.
* `GridSearchCV` with `GroupKFold` (groups = subjects) so no subject is split across folds, scored with macro F1 rather than accuracy.
* `run_loso`: `LeaveOneGroupOut` evaluation returning out-of-fold predictions, class probabilities mapped onto a global class order, and a per-subject accuracy and F1 table.
* Metrics: accuracy, balanced accuracy, macro precision/recall/F1, one-vs-rest macro ROC-AUC.
* Row-normalised confusion matrices, one-vs-rest ROC curves, grouped metric bar charts, per-subject accuracy boxplot and bar chart.
* Feature importance for the tree models, both per feature (top 25) and aggregated per sensor to show which device carries the signal.
* A binary stress vs non-stress task with its own tuning, LOSO run, confusion matrix, and ROC comparison.
* `nested_loso`: optional fully nested cross-validation (outer LOSO, inner GroupKFold grid search on training subjects only) that removes the optimism of tuning on all subjects, at roughly 15x the cost.

## Project Structure

The project is a single self-contained Jupyter notebook. Its 18 numbered sections form the module structure:

```
WESAD__Wearable_Stress_and_Affect_Detection_.ipynb
├── 1.  Install and Import Libraries
├── 2.  Configuration ................ WINDOW_SEC, STEP_SEC, LABEL_PURITY,
│                                      KEEP_LABELS, CHEST_TARGET_FS, WRIST_FS,
│                                      USE_CHEST/USE_WRIST, RUN_NESTED_CV
├── 3.  Download Dataset with kagglehub ... resolves DATA_ROOT, collects and
│                                      numerically sorts PKL_FILES (S2 .. S17)
├── 4.  Inspect Raw Dataset Structure ... peek_pickle()
├── 5.  Signal Loading and Resampling ... _normalise_keys(), downsample(),
│                                         load_subject(), window_label()
├── 6.  Data Exploration ............. label duration table, shade_conditions(),
│                                      full-session and per-condition ECG plots
├── 7.  Feature Extraction Functions .. _clean(), stat_features(), psd(),
│                                       freq_features(), detect_r_peaks(),
│                                       hrv_features(), eda_features(),
│                                       resp_features(), bvp_features(),
│                                       BANDS, extract_window_features(),
│                                       subject_feature_table()
├── 8.  Build the Feature Dataset .... loops over all subjects, writes df
├── 9.  Feature Dataset Exploration .. NaN/constant checks, ANOVA, PCA
├── 10. Protocol and Model Definitions  make_pipeline(), MODELS, tune(),
│                                       run_loso(), score_predictions()
├── 11. Hyperparameter Tuning ........ grouped grid search per model
├── 12. LOSO Evaluation of Tuned Models
├── 13. Model Comparison Tables and Plots ... selects BEST_MODEL
├── 14. Confusion Matrices and ROC ... plot_confusion()
├── 15. Feature Importance ........... per feature and aggregated per sensor
├── 16. Binary Task: Stress vs Non-Stress
├── 17. Optional: Nested Cross-Validation ... nested_loso()
└── 18. Conclusions and Suggestions

wesad_features.csv   generated by section 8, the extracted feature table
                     (windows x features plus label, subject, t_start)
```

### Key functions

| Function | Section | Purpose |
|---|---|---|
| `peek_pickle(pkl_path)` | 4 | Prints top-level keys, label durations, and channel shapes for one subject pickle |
| `downsample(x, fs_in, fs_out)` | 5 | Anti-aliased integer-factor decimation, works on 1-D and (N, 3) arrays |
| `load_subject(pkl_path)` | 5 | Returns `(channels, labels_700hz, subject_id)` where `channels` maps a name to `(array, fs)` |
| `window_label(labels, t0, t1)` | 5 | Returns the class name of a window, or `None` if impure or out of scope |
| `extract_window_features(win)` | 7 | Turns one window of all channels into a flat feature dict |
| `subject_feature_table(...)` | 7 | Slides the window across one subject and returns its feature rows |
| `make_pipeline(model)` | 10 | Wraps a classifier in impute + scale |
| `tune(...)` | 10 | `GroupKFold` grid search, returns best estimator, params, and score |
| `run_loso(...)` | 10 | LOSO evaluation with out-of-fold predictions and per-subject metrics |
| `score_predictions(...)` | 10 | Aggregates accuracy, balanced accuracy, macro P/R/F1, ROC-AUC |
| `plot_confusion(...)` | 14 | Row-normalised confusion matrix with counts annotated |
| `nested_loso(...)` | 17 | Nested LOSO with an inner grouped grid search |

## Installation

Python 3.9 or newer is required.

```bash
git clone <your-repository-url>
cd wesad-stress-detection

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install numpy pandas scipy scikit-learn matplotlib xgboost kagglehub jupyter
```

Inside the notebook, the first cell installs the two non-standard dependencies directly (currently commented out):

```python
!pip install -q kagglehub xgboost
```

The dataset is downloaded programmatically by `kagglehub`, which needs Kaggle credentials. Create an API token in your Kaggle account settings and place `kaggle.json` at `~/.kaggle/kaggle.json`, or export the variables shown in the Configuration section below.

## Usage

### Running the whole pipeline

```bash
jupyter notebook WESAD__Wearable_Stress_and_Affect_Detection_.ipynb
```

Then run all cells in order (Kernel > Restart & Run All). Sections 1 to 8 build the feature table, sections 9 to 18 explore it and evaluate the models. A GPU is not required; XGBoost uses `tree_method="hist"` on CPU.

To run headlessly:

```bash
jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=-1 \
  WESAD__Wearable_Stress_and_Affect_Detection_.ipynb \
  --output WESAD_executed.ipynb
```

### Downloading the dataset

```python
import kagglehub

DATA_ROOT = kagglehub.dataset_download(
    "orvile/wesad-wearable-stress-affect-detection-dataset"
)
print("Dataset root:", DATA_ROOT)
```

### Loading one subject and extracting its features

```python
channels, labels, subject = load_subject(PKL_FILES[0])

for name, (arr, fs) in channels.items():
    print(f"{name:<14} n={arr.size:>9}  fs={fs:>6.1f} Hz  "
          f"duration={arr.size / fs / 60:6.2f} min")

rows = subject_feature_table(channels, labels, subject)
print(f"{subject}: {len(rows)} windows, {len(rows[0]) - 3} features")
```

### Building the full feature table

```python
all_rows = []
for pkl in PKL_FILES:
    channels, labels, subject = load_subject(pkl)
    all_rows.extend(subject_feature_table(channels, labels, subject))
    del channels, labels
    gc.collect()

df = pd.DataFrame(all_rows)
df.to_csv("wesad_features.csv", index=False)
```

Feature extraction is the expensive step. Because the table is persisted to `wesad_features.csv`, later experiments can skip sections 3 to 8 entirely:

```python
df = pd.read_csv("wesad_features.csv")
META_COLS = ["label", "subject", "t_start"]
FEATURE_COLS = [c for c in df.columns if c not in META_COLS]
```

### Tuning and evaluating a single model

```python
X = df[FEATURE_COLS].to_numpy(dtype=np.float64)
y = LabelEncoder().fit(CLASS_ORDER).transform(df["label"].to_numpy())
groups_subj = df["subject"].to_numpy()

estimator, grid = MODELS["LogisticRegression"]
best_est, best_params, cv_f1 = tune("LogisticRegression", estimator, grid,
                                    X, y, groups_subj)

y_pred, y_proba, per_subject = run_loso(best_est, X, y, groups_subj, n_classes=3)
print(score_predictions(y, y_pred, y_proba, CLASS_ORDER))
print(per_subject.round(3))
```

### Running the binary stress task only

```python
y_bin = (df["label"].to_numpy() == "stress").astype(int)

estimator, grid = bin_models["XGBoost"]
best_est, _, _ = tune("[bin] XGBoost", estimator, grid, X, y_bin,
                      groups_subj, scoring="f1")
y_pred_b, y_proba_b, _ = run_loso(best_est, X, y_bin, groups_subj, n_classes=2)
print(score_predictions(y_bin, y_pred_b, y_proba_b, ["non_stress", "stress"]))
```

### Producing the unbiased estimate

Set `RUN_NESTED_CV = True` in section 2 and re-run sections 2 and 17. The outer loop is LOSO and the inner loop is a `GroupKFold` grid search restricted to the training subjects, which removes the optimism introduced by tuning on all subjects. On WESAD the gap is around 1 to 2 points, and the run costs roughly 15 times more.

## Configuration

All tunable constants live in section 2 of the notebook.

| Constant | Default | Meaning |
|---|---|---|
| `WINDOW_SEC` | `60.0` | Window length in seconds. Long enough for reliable HRV and respiration rate (about 60 to 100 beats and 15 breaths), short enough for the affective state to be stationary. |
| `STEP_SEC` | `15.0` | Window step, giving 75% overlap. |
| `LABEL_PURITY` | `0.98` | Minimum fraction of a window's label samples belonging to one class. |
| `KEEP_LABELS` | `{1: baseline, 2: stress, 3: amusement}` | Protocol conditions that are modelled. |
| `CLASS_ORDER` | `["baseline", "stress", "amusement"]` | Fixed class ordering for encoding, plots, and reports. |
| `CHEST_TARGET_FS` | ECG 700, EMG 700, Resp 70, ACC 70, EDA 7, Temp 7 (Hz) | Target rate per chest channel. |
| `WRIST_FS` | ACC 32, BVP 64, EDA 4, TEMP 4 (Hz) | Native E4 rates, used as-is. |
| `USE_CHEST` / `USE_WRIST` | `True` / `True` | Enable or disable a whole device, which allows chest-only or wrist-only ablations. |
| `TUNING_FOLDS` | `5` | Number of `GroupKFold` splits during grid search. |
| `N_JOBS` | `-1` | Parallel jobs for scikit-learn and XGBoost. |
| `RANDOM_STATE` | `42` | Seed for models and PCA. |
| `RUN_NESTED_CV` | `False` | Enables the nested cross-validation in section 17. |
| `EPS` | `1e-12` | Numerical floor used in ratios, entropies, and slope guards. |
| `BANDS` | see section 7 | Frequency bands per signal type (EMG, ACC, RESP, BVP). |

`kagglehub` reads its credentials from the environment, so the dataset download can also be configured without a credentials file:

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

## Technologies Used

* **Language**: Python 3
* **Numerical and data handling**: NumPy, pandas
* **Signal processing**: SciPy (`butter`, `sosfiltfilt`, `resample_poly`, `welch`, `find_peaks`, `scipy.stats`)
* **Machine learning**: scikit-learn (`Pipeline`, `SimpleImputer`, `StandardScaler`, `LabelEncoder`, `PCA`, `LogisticRegression`, `RandomForestClassifier`, `MLPClassifier`, `GroupKFold`, `GridSearchCV`, `LeaveOneGroupOut`, metrics)
* **Gradient boosting**: XGBoost (`XGBClassifier`)
* **Visualisation**: Matplotlib
* **Data access**: kagglehub
* **Standard library**: `os`, `gc`, `glob`, `time`, `pickle`, `warnings`
* **Environment**: Jupyter notebook

## Dataset

WESAD (Wearable Stress and Affect Detection), 15 subjects (S2 to S17, S1 and S12 are absent from the public release), each stored as one pickle containing a 700 Hz label track and two signal groups:

* **chest** (RespiBAN, all channels at 700 Hz): ACC (3 axes), ECG, EMG, EDA, Temp, Resp
* **wrist** (Empatica E4): ACC at 32 Hz, BVP at 64 Hz, EDA at 4 Hz, TEMP at 4 Hz

Label codes: `0` transient, `1` baseline, `2` stress, `3` amusement, `4` meditation, `5` to `7` unused. Roughly 36 minutes of usable labelled signal per subject for the three modelled classes.

The dataset is released for academic research; consult its original licence and cite the WESAD paper by Schmidt et al. (2018) if you use it. Note that citation details should be verified against the original publication.

## Limitations and Next Steps

* Sections 11 and 12 tune on all subjects and then evaluate with LOSO on the same subjects, so those figures are mildly optimistic. Section 17 provides the unbiased alternative.
* With 15 subjects, confidence intervals on all reported metrics are wide.
* Between-subject spread is the main obstacle to deployment (worst subject at 0.599 accuracy). Subject-wise baseline normalisation, for example expressing each feature relative to that person's resting window, or light personalisation with a short calibration recording, is the most promising next step.
* Feature importances come from a model refitted on all subjects and describe what the model uses, not causality. Many features are correlated and therefore dilute each other's importance.
* The amusement class is the hardest to separate, which suggests the difficulty lies in distinguishing mild positive arousal from rest rather than in detecting stress.

## License

Released under the MIT License. The WESAD dataset itself is covered by its own separate licence terms.

## Author

*M. E. Petra Bošković*
