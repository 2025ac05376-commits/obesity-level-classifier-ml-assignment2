import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

SEED = 5376
TARGET_COLUMN = "NObeyesdad"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SOURCE_CSV = os.path.join(ROOT, "data", "obesity_levels.csv")
TEST_CSV = os.path.join(ROOT, "test_data.csv")


def load_frame():
    frame = pd.read_csv(SOURCE_CSV)
    frame = frame.drop_duplicates().reset_index(drop=True)
    return frame


def dense_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(features):
    numeric_columns = features.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = [c for c in features.columns if c not in numeric_columns]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_columns),
            ("cat", dense_encoder(), categorical_columns),
        ],
        remainder="drop",
    )


def model_zoo():
    return {
        "Logistic Regression": LogisticRegression(max_iter=3000, C=1.0, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, min_samples_leaf=3, random_state=SEED),
        "kNN": KNeighborsClassifier(n_neighbors=7, weights="distance"),
        "Naive Bayes": GaussianNB(var_smoothing=1e-8),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=1, n_jobs=-1, random_state=SEED
        ),
    }


def score_model(fitted, x_eval, y_eval, class_order):
    predicted = fitted.predict(x_eval)
    probabilities = fitted.predict_proba(x_eval)
    if len(class_order) == 2:
        auc = roc_auc_score(y_eval, probabilities[:, 1])
        averaging = "binary"
    else:
        auc = roc_auc_score(y_eval, probabilities, multi_class="ovr", average="macro", labels=class_order)
        averaging = "macro"
    return {
        "Accuracy": accuracy_score(y_eval, predicted),
        "AUC": auc,
        "Precision": precision_score(y_eval, predicted, average=averaging, zero_division=0),
        "Recall": recall_score(y_eval, predicted, average=averaging, zero_division=0),
        "F1": f1_score(y_eval, predicted, average=averaging, zero_division=0),
        "MCC": matthews_corrcoef(y_eval, predicted),
    }


def freeze_requirements():
    import sklearn
    lines = [
        "streamlit",
        "scikit-learn==" + sklearn.__version__,
        "pandas",
        "numpy",
        "matplotlib",
        "joblib",
    ]
    with open(os.path.join(ROOT, "requirements.txt"), "w") as handle:
        handle.write("\n".join(lines) + "\n")


def markdown_tables(results):
    header = "| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |\n"
    header += "|---|---|---|---|---|---|---|\n"
    rows = ""
    for name, scores in results.items():
        rows += "| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |\n".format(
            name, scores["Accuracy"], scores["AUC"], scores["Precision"],
            scores["Recall"], scores["F1"], scores["MCC"]
        )
    return header + rows


def main():
    frame = load_frame()
    labels = frame[TARGET_COLUMN]
    features = frame.drop(columns=[TARGET_COLUMN])

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.25, stratify=labels, random_state=SEED
    )

    holdout = x_test.copy()
    holdout[TARGET_COLUMN] = y_test.values
    holdout.to_csv(TEST_CSV, index=False)

    class_order = sorted(labels.unique().tolist())
    results = {}

    for name, estimator in model_zoo().items():
        pipeline = Pipeline(
            steps=[("prep", build_preprocessor(features)), ("clf", estimator)]
        )
        pipeline.fit(x_train, y_train)
        results[name] = score_model(pipeline, x_test, y_test, class_order)
        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(pipeline, os.path.join(HERE, slug + ".joblib"), compress=3)
        print(name, {k: round(v, 4) for k, v in results[name].items()})

    table = pd.DataFrame(results).T
    table.index.name = "ML Model Name"
    table.to_csv(os.path.join(HERE, "metrics.csv"))

    with open(os.path.join(HERE, "results_tables.md"), "w") as handle:
        handle.write(markdown_tables(results))

    metadata = {
        "target": TARGET_COLUMN,
        "classes": class_order,
        "n_rows": int(frame.shape[0]),
        "n_features": int(features.shape[1]),
        "train_rows": int(x_train.shape[0]),
        "test_rows": int(x_test.shape[0]),
        "random_state": SEED,
        "feature_order": features.columns.tolist(),
    }
    with open(os.path.join(HERE, "dataset_meta.json"), "w") as handle:
        json.dump(metadata, handle, indent=2)

    freeze_requirements()
    print("\nbest by F1:", table["F1"].idxmax())
    print("artefacts written to", HERE)


if __name__ == "__main__":
    main()
