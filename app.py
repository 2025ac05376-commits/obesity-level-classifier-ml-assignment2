import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")
DEFAULT_TEST = os.path.join(HERE, "test_data.csv")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest_ensemble.joblib",
}

INK = "#0B4F6C"
AMBER = "#F6AE2D"
SLATE = "#33475B"

st.set_page_config(
    page_title="Obesity Level Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2.2rem; max-width: 1250px;}
    h1, h2, h3 {color: #0B4F6C; letter-spacing: -0.02em;}
    .banner {
        border-left: 6px solid #F6AE2D;
        background: linear-gradient(90deg, #EAF3F7 0%, #FFFFFF 100%);
        padding: 1.1rem 1.4rem; border-radius: 4px; margin-bottom: 1.4rem;
    }
    .banner h1 {margin: 0 0 .25rem 0; font-size: 1.85rem;}
    .banner p {margin: 0; color: #33475B; font-size: .95rem;}
    .tag {
        display: inline-block; background: #0B4F6C; color: #fff; font-size: .72rem;
        padding: .15rem .55rem; border-radius: 999px; margin-right: .35rem;
        letter-spacing: .04em; text-transform: uppercase;
    }
    div[data-testid="stMetric"] {
        background: #F7F9FA; border: 1px solid #E1E8EC;
        border-radius: 6px; padding: .75rem .9rem;
    }
    div[data-testid="stMetricValue"] {color: #0B4F6C; font-size: 1.55rem;}
    section[data-testid="stSidebar"] {background: #F4F7F9;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_metadata():
    path = os.path.join(MODEL_DIR, "dataset_meta.json")
    with open(path) as handle:
        return json.load(handle)


@st.cache_resource
def load_estimator(filename):
    return joblib.load(os.path.join(MODEL_DIR, filename))


@st.cache_data
def read_csv(source):
    return pd.read_csv(source)


def evaluate(truth, predicted, probabilities, class_order):
    binary = len(class_order) == 2
    averaging = "binary" if binary else "macro"
    if binary:
        auc = roc_auc_score(truth, probabilities[:, 1])
    else:
        auc = roc_auc_score(truth, probabilities, multi_class="ovr",
                            average="macro", labels=class_order)
    return {
        "Accuracy": accuracy_score(truth, predicted),
        "AUC": auc,
        "Precision": precision_score(truth, predicted, average=averaging, zero_division=0),
        "Recall": recall_score(truth, predicted, average=averaging, zero_division=0),
        "F1": f1_score(truth, predicted, average=averaging, zero_division=0),
        "MCC": matthews_corrcoef(truth, predicted),
    }


def confusion_figure(truth, predicted, class_order):
    matrix = confusion_matrix(truth, predicted, labels=class_order)
    shades = LinearSegmentedColormap.from_list("marine", ["#FFFFFF", "#7FB2C6", INK])
    height = 0.55 * len(class_order) + 2.2
    figure, axis = plt.subplots(figsize=(height + 1.6, height))
    axis.imshow(matrix, cmap=shades)
    axis.set_xticks(range(len(class_order)))
    axis.set_yticks(range(len(class_order)))
    axis.set_xticklabels(class_order, rotation=45, ha="right", fontsize=8)
    axis.set_yticklabels(class_order, fontsize=8)
    axis.set_xlabel("Predicted", fontsize=9, color=SLATE)
    axis.set_ylabel("Actual", fontsize=9, color=SLATE)
    ceiling = matrix.max() if matrix.max() else 1
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            tone = "#FFFFFF" if value > 0.55 * ceiling else SLATE
            axis.text(column, row, str(value), ha="center", va="center",
                      fontsize=9, color=tone)
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.tight_layout()
    return figure


meta = load_metadata()
class_order = meta["classes"]
target_column = meta["target"]

st.markdown(
    "<div class='banner'><h1>Obesity Level Classifier</h1>"
    "<p><span class='tag'>UCI dataset</span>"
    "<span class='tag'>{} classes</span>"
    "<span class='tag'>{} features</span>"
    "Five supervised classifiers trained on eating-habit and physical-condition survey data, "
    "scored side by side on an unseen hold-out set.</p></div>".format(
        len(class_order), meta["n_features"]),
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("1. Test data")
    uploaded = st.file_uploader("Upload a test CSV", type=["csv"])
    st.caption("Leave empty to score the bundled hold-out split (test_data.csv).")

    st.subheader("2. Model")
    chosen_model = st.selectbox("Classifier", list(MODEL_FILES.keys()), index=4)

    st.subheader("3. View")
    show_leaderboard = st.checkbox("Compare all five models", value=True)
    show_report = st.checkbox("Show classification report", value=True)
    show_predictions = st.checkbox("Show row-level predictions", value=False)

source = uploaded if uploaded is not None else DEFAULT_TEST
frame = read_csv(source)

if uploaded is None:
    st.info("Scoring the bundled hold-out split. Upload your own CSV in the sidebar to replace it.")

if target_column not in frame.columns:
    st.error("Column '{}' is missing. The uploaded CSV needs the true label to compute metrics.".format(target_column))
    st.stop()

expected = meta["feature_order"]
missing = [column for column in expected if column not in frame.columns]
if missing:
    st.error("These feature columns are missing from the CSV: {}".format(", ".join(missing)))
    st.stop()

features = frame[expected]
truth = frame[target_column]

left, middle, right = st.columns(3)
left.metric("Rows scored", "{:,}".format(len(frame)))
middle.metric("Features", len(expected))
right.metric("Target classes", truth.nunique())

with st.expander("Preview the data"):
    st.dataframe(frame.head(15))

pipeline = load_estimator(MODEL_FILES[chosen_model])
predicted = pipeline.predict(features)
probabilities = pipeline.predict_proba(features)
scores = evaluate(truth, predicted, probabilities, class_order)

st.subheader("Evaluation metrics — {}".format(chosen_model))
columns = st.columns(6)
for column, (name, value) in zip(columns, scores.items()):
    column.metric(name, "{:.4f}".format(value))
st.caption("Multi-class AUC uses one-vs-rest with macro averaging. "
           "Precision, recall and F1 are macro-averaged across the {} classes.".format(len(class_order)))

st.subheader("Confusion matrix")
matrix_column, report_column = st.columns([1.15, 1])
with matrix_column:
    st.pyplot(confusion_figure(truth, predicted, class_order))

with report_column:
    if show_report:
        st.markdown("**Classification report**")
        report = classification_report(truth, predicted, labels=class_order,
                                       output_dict=True, zero_division=0)
        report_frame = pd.DataFrame(report).T.round(4)
        st.dataframe(report_frame, height=380)
    else:
        st.markdown("**Predicted class distribution**")
        st.bar_chart(pd.Series(predicted).value_counts())

if show_leaderboard:
    st.subheader("All models on this test set")
    board = {}
    progress = st.progress(0.0)
    for position, (name, filename) in enumerate(MODEL_FILES.items(), start=1):
        estimator = load_estimator(filename)
        board[name] = evaluate(truth, estimator.predict(features),
                               estimator.predict_proba(features), class_order)
        progress.progress(position / len(MODEL_FILES))
    progress.empty()
    board_frame = pd.DataFrame(board).T.round(4)
    board_frame.index.name = "ML Model Name"
    st.dataframe(board_frame.style.highlight_max(axis=0, props="background-color:#FDF0D2;font-weight:700;"))
    winner = board_frame["F1"].idxmax()
    st.success("Best macro F1 on this test set: **{}** ({:.4f})".format(winner, board_frame.loc[winner, "F1"]))

if show_predictions:
    st.subheader("Row-level predictions")
    output = frame.copy()
    output["predicted_" + target_column] = predicted
    output["confidence"] = probabilities.max(axis=1).round(4)
    output["correct"] = output[target_column].values == predicted
    st.dataframe(output.head(200))
    st.download_button(
        "Download predictions as CSV",
        output.to_csv(index=False).encode("utf-8"),
        file_name="predictions_{}.csv".format(chosen_model.split()[0].lower()),
        mime="text/csv",
    )

st.divider()
st.caption("M.Tech (AIML) — Machine Learning Assignment 2 · models trained offline in model/train_models.py "
           "and loaded here as scikit-learn pipelines.")
