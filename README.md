# Obesity Level Classification — ML Assignment 2

M.Tech (AIML), Work Integrated Learning Programmes Division, BITS Pilani
Machine Learning — Assignment 2

---

## a. Problem statement

Given a person's demographic profile, eating habits and physical-activity
routine, predict which of seven obesity / weight categories they fall into.

This is a **multi-class classification** problem with 7 mutually exclusive
target classes. The practical motivation is screening: a short lifestyle
questionnaire is far cheaper to administer at scale than a clinical
assessment, so a model that can flag likely obesity levels from questionnaire
answers is useful as a first-pass triage tool for public-health programmes.

Five supervised classifiers are trained on the same train/test split, scored
on six metrics, and served through a Streamlit web app where any test CSV can
be uploaded and any of the five models selected.

---

## b. Dataset description

**Name:** Estimation of Obesity Levels Based On Eating Habits and Physical Condition
**Source:** UCI Machine Learning Repository (also mirrored on Kaggle)
**URL:** https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition

| Property | Value |
|---|---|
| Instances (raw) | 2,111 |
| Instances after removing 24 exact duplicates | 2,087 |
| Features | 16 (8 numeric, 8 categorical) |
| Target column | `NObeyesdad` |
| Target classes | 7 |
| Missing values | None |
| Train / Test split | 1,565 / 522 (75 / 25, stratified, `random_state=5376`) |

Both assignment thresholds are met: 16 features (minimum 12) and 2,087
instances (minimum 500).

**Origin of the data.** The records were collected from respondents in Mexico,
Peru and Colombia through a web questionnaire. Roughly 23% of the rows are
directly collected responses; the remaining 77% were generated synthetically
with SMOTE in Weka to balance the seven classes. This matters for
interpretation and is discussed in the observations below.

### Feature dictionary

| # | Feature | Type | Meaning |
|---|---|---|---|
| 1 | `Gender` | categorical | Male / Female |
| 2 | `Age` | numeric | Age in years |
| 3 | `Height` | numeric | Height in metres |
| 4 | `Weight` | numeric | Weight in kilograms |
| 5 | `family_history_with_overweight` | categorical | Family history of overweight (yes / no) |
| 6 | `FAVC` | categorical | Frequent consumption of high-calorie food (yes / no) |
| 7 | `FCVC` | numeric | Frequency of vegetable consumption |
| 8 | `NCP` | numeric | Number of main meals per day |
| 9 | `CAEC` | categorical | Eating between meals (no / Sometimes / Frequently / Always) |
| 10 | `SMOKE` | categorical | Smokes (yes / no) |
| 11 | `CH2O` | numeric | Daily water intake |
| 12 | `SCC` | categorical | Monitors calorie intake (yes / no) |
| 13 | `FAF` | numeric | Physical activity frequency per week |
| 14 | `TUE` | numeric | Time spent on technology devices |
| 15 | `CALC` | categorical | Alcohol consumption (no / Sometimes / Frequently / Always) |
| 16 | `MTRANS` | categorical | Usual mode of transport |

### Target class distribution (after de-duplication)

| Class | Count |
|---|---|
| Obesity_Type_I | 351 |
| Obesity_Type_III | 324 |
| Obesity_Type_II | 297 |
| Overweight_Level_II | 290 |
| Normal_Weight | 282 |
| Overweight_Level_I | 276 |
| Insufficient_Weight | 267 |

The classes are close to balanced, so plain accuracy is a reasonable headline
metric here — but MCC and macro-F1 are still reported because they are far less
forgiving of a model that only does well on the easy classes.

### Preprocessing

A single `ColumnTransformer` is used inside every model pipeline, so the exact
same transformation is applied at training time and at prediction time in the
app:

- numeric columns → `StandardScaler`
- categorical columns → `OneHotEncoder(handle_unknown="ignore")`

Each classifier is wrapped in a `Pipeline(preprocessor, estimator)` and saved
with `joblib`, which means the Streamlit app can feed it a raw CSV with no
manual encoding step.

---

## c. GitHub Repository Link

**Repository:** https://github.com/2025ac05376-commits/obesity-level-classifier-ml-assignment2

**Live Streamlit app:** https://obesity-classifier-2025ac05376.streamlit.app/

Repository contents:

```
obesity-risk-classifier/
├── app.py                     Streamlit application
├── requirements.txt           pinned dependencies
├── README.md                  this file
├── test_data.csv              522-row stratified hold-out split
├── data/
│   └── obesity_levels.csv     full source dataset (2,111 rows)
└── model/
    ├── train_models.py        trains all 5 models and writes the artefacts
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── random_forest_ensemble.joblib
    ├── metrics.csv            the comparison table, machine readable
    ├── results_tables.md      the comparison table, markdown
    └── dataset_meta.json      class order and feature order used by the app
```

Reproduce everything with:

```bash
pip install -r requirements.txt
python model/train_models.py
streamlit run app.py
```

---

## d. Models used

All five models are trained on the identical stratified 75/25 split of the same
dataset, with the identical preprocessing pipeline. Metrics are computed on the
522-row hold-out set only.

Because the problem is multi-class, AUC is computed one-vs-rest with macro
averaging, and Precision / Recall / F1 are macro-averaged across the seven
classes. MCC is computed directly on the multi-class confusion matrix.

### Hyperparameters

| Model | Settings |
|---|---|
| Logistic Regression | `max_iter=3000`, `C=1.0`, multinomial (default) |
| Decision Tree | `max_depth=12`, `min_samples_leaf=3` |
| kNN | `n_neighbors=7`, `weights="distance"` |
| Naive Bayes | `GaussianNB`, `var_smoothing=1e-8` |
| Random Forest | `n_estimators=300`, unrestricted depth |

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8793 | 0.9869 | 0.8802 | 0.8756 | 0.8710 | 0.8607 |
| Decision Tree | 0.9234 | 0.9713 | 0.9224 | 0.9222 | 0.9209 | 0.9110 |
| kNN | 0.8295 | 0.9575 | 0.8252 | 0.8220 | 0.8012 | 0.8056 |
| Naive Bayes | 0.5268 | 0.8901 | 0.5747 | 0.5180 | 0.4639 | 0.4629 |
| Random Forest (Ensemble) | 0.9521 | 0.9967 | 0.9531 | 0.9504 | 0.9511 | 0.9442 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Solid but not the best: 87.9% accuracy with the second-highest AUC (0.9869). The high AUC alongside the lower accuracy is the informative part — the model ranks the correct class near the top almost always, but its linear decision boundaries cannot cleanly separate adjacent categories such as `Overweight_Level_I` vs `Overweight_Level_II`, so it loses points on the final argmax. It is the most interpretable model here and the one whose coefficients could actually be reported to a health team. |
| Decision Tree | Strong second place at 92.3% accuracy and MCC 0.9110. A single tree suits this dataset because the target is essentially a set of thresholds on BMI, and axis-aligned splits on `Weight` and `Height` reproduce those thresholds almost exactly. Its AUC (0.9713) is the second-lowest despite the good accuracy, because a depth-limited tree emits coarse, near-binary leaf probabilities and therefore ranks poorly even when it classifies correctly. Depth was capped at 12 with `min_samples_leaf=3` to stop it memorising the training set. |
| kNN | Weakest of the four serious models at 83.0% accuracy, and its macro-F1 (0.8012) sits noticeably below its accuracy, meaning the errors are concentrated in particular classes rather than spread evenly. The cause is dimensionality: one-hot encoding expands 16 features into roughly 30 columns, most of them sparse binary, so Euclidean distance gets diluted by the dummy variables and neighbourhoods stop being meaningful. Distance weighting and k=7 helped but could not fix the underlying geometry. |
| Naive Bayes | Clearly the worst model — 52.7% accuracy, macro-F1 0.4639, MCC 0.4629 — and it is worth being precise about why. Gaussian Naive Bayes assumes the features are conditionally independent given the class and that each is normally distributed. Both assumptions fail badly here: `Height` and `Weight` are strongly dependent (the target is derived from their ratio), and the one-hot dummy columns are Bernoulli, not Gaussian. Notably its AUC is still 0.8901, far better than its accuracy suggests — the class ranking is broadly right, but the calibration is so distorted that the argmax lands on the wrong class about half the time. This is the clearest illustration in the whole experiment of why AUC and accuracy must be read together. |
| Random Forest (Ensemble) | The winner on every single metric: 95.2% accuracy, AUC 0.9967, macro-F1 0.9511, MCC 0.9442. Averaging 300 de-correlated trees keeps the axis-aligned splitting that suits this data while removing the variance that limits the single tree, and bagging over feature subsets means the sparse one-hot columns cost it nothing. The near-perfect AUC also shows the averaged probabilities are well calibrated, not just correctly ordered. Cost: it is the slowest to train and the least interpretable. |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble).** It is best on all six metrics simultaneously, which removes any ambiguity about the choice. The MCC gap is the most telling: 0.9442 versus 0.9110 for the next best model, and MCC is the metric least likely to be flattered by a balanced test set. If interpretability were a hard requirement, Logistic Regression would be the sensible fallback — it gives up about 7 accuracy points but produces coefficients a domain expert can read and challenge. |

### Additional note on the results

`Height` and `Weight` are strong predictors here because BMI — computed from
exactly those two columns — is what defines the seven target classes in the
first place. The models are therefore partly recovering a known formula rather
than discovering a novel relationship, which is why the top scores are so high.
This is a genuine property of the dataset rather than a bug in the pipeline,
but it is worth stating explicitly: a fairer test of the *lifestyle* features
would be to drop `Height` and `Weight` and re-run, and the gap between the
models would then narrow considerably.

---

## Streamlit app features

| Requirement | Implementation |
|---|---|
| Dataset upload option (CSV) | Sidebar file uploader; falls back to the bundled `test_data.csv` if nothing is uploaded. Uploads are capped at 10 MB. |
| Model selection dropdown | Sidebar `selectbox` with all five trained models. |
| Display of evaluation metrics | All six metrics shown as metric cards for the selected model, plus an optional leaderboard scoring all five models on the uploaded data with the best value per column highlighted. |
| Confusion matrix / classification report | Annotated confusion-matrix heatmap plus a per-class classification report side by side. |

Extras: data preview, predicted-class distribution, row-level predictions with
per-row model confidence, and a CSV download of the predictions.

---

## Uploading your own CSV

The uploaded file must contain the 16 feature columns listed in the feature
dictionary above **and** the `NObeyesdad` label column (the label is needed to
compute metrics). Column order does not matter. `test_data.csv` in the
repository root is a valid example.
