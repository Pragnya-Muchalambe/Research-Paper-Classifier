"""Research Paper Acceptance Classifier.

Offline-friendly ML module and CLI script.
No NLTK or external downloads are required.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import matplotlib


# Use a non-GUI backend so plotting works in terminals/servers.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

FEATURE_COLUMNS = [
    "word_count",
    "avg_sentence_length",
    "tech_keyword_count",
    "positive_word_count",
    "readability_score",
    "author_count",
    "unique_word_ratio",
    "title_length",
    "has_results_mention",
    "citation_density",
]

TECH_KEYWORDS = [
    "novel",
    "framework",
    "algorithm",
    "transformer",
    "neural",
    "deep",
    "learning",
    "optimization",
    "benchmark",
    "architecture",
    "attention",
    "gradient",
    "convergence",
    "representation",
    "contrastive",
    "reinforcement",
]

POSITIVE_WORDS = [
    "novel",
    "innovative",
    "superior",
    "outperform",
    "significant",
    "improve",
    "state-of-the-art",
    "remarkable",
    "efficient",
    "robust",
    "promising",
]


@dataclass
class TrainingBundle:
    models: Dict[str, Pipeline]
    metrics: Dict[str, Dict[str, Any]]
    feature_columns: List[str]
    train_df: pd.DataFrame
    test_df: pd.DataFrame


def word_tokenize(text: str) -> List[str]:
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())


def sent_tokenize(text: str) -> List[str]:
    return re.split(r"(?<=[.!?])\s+", text.strip())


def extract_features(text: str) -> Dict[str, float]:
    text_lower = text.lower()
    words = word_tokenize(text_lower)
    sentences = sent_tokenize(text)

    word_count = len(words)
    sentence_count = max(len(sentences), 1)
    avg_sent_length = word_count / sentence_count
    unique_ratio = len(set(words)) / max(word_count, 1)
    tech_count = sum(1 for kw in TECH_KEYWORDS if kw in text_lower)
    positive_count = sum(1 for pw in POSITIVE_WORDS if pw in text_lower)

    avg_word_len = np.mean([len(w) for w in words]) if words else 0.0
    readability = max(0.0, 100 - (avg_word_len * 8) - (avg_sent_length * 0.5))
    has_results = int(
        any(w in text_lower for w in ["result", "achieve", "demonstrate", "show", "prove"])
    )

    return {
        "word_count": float(word_count),
        "avg_sentence_length": float(round(avg_sent_length, 2)),
        "tech_keyword_count": float(tech_count),
        "positive_word_count": float(positive_count),
        "readability_score": float(round(readability, 2)),
        "unique_word_ratio": float(round(unique_ratio, 3)),
        "has_results_mention": float(has_results),
    }


def _sample_abstract(label: int, rng: np.random.Generator) -> str:
    strong = [
        "We propose a novel framework with strong empirical gains on benchmark datasets.",
        "Our optimization strategy demonstrates consistent improvements across tasks.",
        "Extensive experiments show robust and promising performance.",
    ]
    weak = [
        "This work studies a basic approach and reports moderate results.",
        "We compare standard methods with a simple variation.",
        "Experiments show mixed outcomes and limited improvements.",
    ]

    # Add overlap so classes are not perfectly separable.
    if label == 1:
        picks = rng.choice(strong, size=2, replace=True).tolist()
        if rng.random() < 0.35:
            picks.append(rng.choice(weak))
    else:
        picks = rng.choice(weak, size=2, replace=True).tolist()
        if rng.random() < 0.35:
            picks.append(rng.choice(strong))
    return " ".join(picks)


def generate_dataset(n_samples: int = 600, seed: int = 42, noise_rate: float = 0.08) -> pd.DataFrame:
    """Generate a synthetic dataset with realistic overlap and light label noise."""
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []

    for _ in range(n_samples):
        true_label = int(rng.random() > 0.45)
        abstract = _sample_abstract(true_label, rng)
        text_feats = extract_features(abstract)

        if true_label == 1:
            word_count = rng.normal(240, 55)
            avg_sentence_length = rng.normal(22, 4)
            tech_keyword_count = rng.normal(11, 4)
            positive_word_count = rng.normal(6, 2.5)
            readability = rng.normal(46, 10)
            author_count = rng.normal(4, 1.5)
            unique_ratio = rng.normal(0.65, 0.1)
            title_length = rng.normal(11, 3)
            has_results = rng.binomial(1, 0.75)
            citation_density = rng.normal(0.08, 0.03)
        else:
            word_count = rng.normal(190, 60)
            avg_sentence_length = rng.normal(18, 4)
            tech_keyword_count = rng.normal(7, 4)
            positive_word_count = rng.normal(3, 2)
            readability = rng.normal(56, 11)
            author_count = rng.normal(2.8, 1.3)
            unique_ratio = rng.normal(0.56, 0.1)
            title_length = rng.normal(8, 3)
            has_results = rng.binomial(1, 0.45)
            citation_density = rng.normal(0.045, 0.025)

        row = {
            "abstract": abstract,
            "word_count": float(np.clip(0.65 * word_count + 0.35 * text_feats["word_count"], 70, 450)),
            "avg_sentence_length": float(np.clip(0.7 * avg_sentence_length + 0.3 * text_feats["avg_sentence_length"], 7, 40)),
            "tech_keyword_count": float(np.clip(0.7 * tech_keyword_count + 0.3 * text_feats["tech_keyword_count"], 0, 30)),
            "positive_word_count": float(np.clip(0.7 * positive_word_count + 0.3 * text_feats["positive_word_count"], 0, 20)),
            "readability_score": float(np.clip(0.8 * readability + 0.2 * text_feats["readability_score"], 5, 95)),
            "author_count": float(np.clip(author_count, 1, 10)),
            "unique_word_ratio": float(np.clip(0.75 * unique_ratio + 0.25 * text_feats["unique_word_ratio"], 0.2, 0.95)),
            "title_length": float(np.clip(title_length, 3, 20)),
            "has_results_mention": float(max(has_results, int(text_feats["has_results_mention"]))),
            "citation_density": float(np.clip(citation_density, 0.0, 0.2)),
            "label": true_label,
        }

        # Flip a small percentage of labels to avoid unrealistically perfect accuracy.
        if rng.random() < noise_rate:
            row["label"] = 1 - row["label"]

        rows.append(row)

    return pd.DataFrame(rows)


def _build_models(seed: int = 42) -> Dict[str, Pipeline]:
    return {
        "Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GaussianNB()),
        ]),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]),
        # Trees do not need scaling.
        "Decision Tree": Pipeline([
            ("clf", DecisionTreeClassifier(max_depth=5, min_samples_leaf=8, random_state=seed)),
        ]),
    }


def train_models(df: pd.DataFrame, seed: int = 42) -> TrainingBundle:
    X = df[FEATURE_COLUMNS].copy()
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    models = _build_models(seed=seed)
    metrics: Dict[str, Dict[str, Any]] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics[name] = {
            "Accuracy": float(accuracy_score(y_test, y_pred)),
            "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "F1 Score": float(f1_score(y_test, y_pred, zero_division=0)),
            "Classification Report": classification_report(y_test, y_pred, digits=3, zero_division=0),
            "Confusion Matrix": confusion_matrix(y_test, y_pred),
            "y_test": y_test,
            "y_pred": y_pred,
        }

    train_df = X_train.copy()
    train_df["label"] = y_train.values
    test_df = X_test.copy()
    test_df["label"] = y_test.values

    return TrainingBundle(
        models=models,
        metrics=metrics,
        feature_columns=FEATURE_COLUMNS,
        train_df=train_df,
        test_df=test_df,
    )


def _safe_predict_proba(model: Pipeline, input_df: pd.DataFrame) -> Tuple[int, float]:
    # Get raw prediction value (may be numeric or a label string)
    raw_pred = model.predict(input_df)[0]
    confidence = 50.0

    # Try to map raw_pred to a class index (0 or 1)
    pred_idx = None
    try:
        if isinstance(raw_pred, (int, np.integer)):
            pred_idx = int(raw_pred)
        else:
            # If estimator exposes classes_, locate the index
            clf = getattr(model, "named_steps", {}).get("clf", model)
            classes = getattr(clf, "classes_", None)
            if classes is not None:
                # numpy comparison will work for both numeric and string labels
                try:
                    idxs = np.where(classes == raw_pred)[0]
                    if len(idxs) > 0:
                        pred_idx = int(idxs[0])
                except Exception:
                    pred_idx = None

            # Fallback string mapping
            if pred_idx is None:
                sval = str(raw_pred).strip().lower()
                if sval in ("accepted", "accept", "yes", "true", "1"):
                    pred_idx = 1
                elif sval in ("rejected", "reject", "no", "false", "0"):
                    pred_idx = 0
    except Exception:
        pred_idx = None

    # If still unknown, default to 1 if truthy, else 0
    if pred_idx is None:
        try:
            pred_idx = int(bool(raw_pred))
        except Exception:
            pred_idx = 1

    # If the model supports predict_proba, use it for a meaningful confidence
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(input_df)[0]
            # If probs is a dict-like (rare), try to handle it; otherwise index by pred_idx
            if isinstance(probs, dict):
                # Try mapping using classes_ if available
                clf = getattr(model, "named_steps", {}).get("clf", model)
                classes = getattr(clf, "classes_", None)
                if classes is not None and raw_pred in classes:
                    confidence = float(probs.get(raw_pred, 0.5) * 100)
                else:
                    confidence = float(next(iter(probs.values())) * 100)
            else:
                confidence = float(probs[pred_idx] * 100)
        except Exception:
            confidence = 50.0
    else:
        # Try to use decision_function -> convert to probability via sigmoid
        try:
            if hasattr(model, "decision_function"):
                df = model.decision_function(input_df)
                # df may be scalar or array
                score = float(df[0]) if hasattr(df, "__len__") else float(df)
                prob = 1.0 / (1.0 + float(np.exp(-score)))
                # If classes_ exists and positive class index is not 1, invert
                clf = getattr(model, "named_steps", {}).get("clf", model)
                classes = getattr(clf, "classes_", None)
                if classes is not None and len(classes) == 2:
                    # assume classes[1] corresponds to positive class for sigmoid
                    confidence = float(prob * 100)
                else:
                    confidence = float(prob * 100)
            else:
                # No probability info; fallback neutral value
                confidence = 50.0
        except Exception:
            confidence = 50.0

    return int(pred_idx), float(confidence)


def build_input_row(
    abstract: str,
    author_count: int,
    title_length: int,
    citation_density: float,
) -> pd.DataFrame:
    feats = extract_features(abstract)
    feats["author_count"] = float(author_count)
    feats["title_length"] = float(title_length)
    feats["citation_density"] = float(citation_density)
    return pd.DataFrame([feats])[FEATURE_COLUMNS]


def predict_with_models(
    abstract: str,
    author_count: int,
    title_length: int,
    citation_density: float,
    models: Dict[str, Pipeline],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    input_df = build_input_row(abstract, author_count, title_length, citation_density)

    pred_rows = []
    for name, model in models.items():
        pred, confidence = _safe_predict_proba(model, input_df)
        pred_rows.append(
            {
                "Model": name,
                "Prediction": "Accepted" if pred == 1 else "Rejected",
                "Confidence (%)": round(confidence, 2),
            }
        )

    return pd.DataFrame(pred_rows), input_df


def generate_visualizations(df: pd.DataFrame, bundle: TrainingBundle, output_dir: str = "outputs") -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Figure 1: Dataset distribution.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    counts = df["label"].map({1: "Accepted", 0: "Rejected"}).value_counts()
    axes[0].bar(counts.index, counts.values, color=["#2ecc71", "#e74c3c"])
    axes[0].set_title("Dataset Distribution")
    axes[0].set_ylabel("Count")

    axes[1].hist(
        [df[df["label"] == 1]["word_count"], df[df["label"] == 0]["word_count"]],
        bins=24,
        label=["Accepted", "Rejected"],
        color=["#2ecc71", "#e74c3c"],
        alpha=0.8,
    )
    axes[1].set_title("Word Count Distribution")
    axes[1].set_xlabel("Word Count")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig1_dataset_overview.png"), dpi=150)
    plt.close(fig)

    # Figure 2: Model metric comparison.
    model_names = list(bundle.metrics.keys())
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    x = np.arange(len(model_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(11, 5))
    for idx, metric in enumerate(metric_names):
        values = [bundle.metrics[m][metric] * 100 for m in model_names]
        ax.bar(x + idx * width, values, width=width, label=metric)

    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig2_model_comparison.png"), dpi=150)
    plt.close(fig)

    # Figure 3: Confusion matrices.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, (name, m) in zip(axes, bundle.metrics.items()):
        cm = m["Confusion Matrix"]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Rejected", "Accepted"])
        ax.set_yticklabels(["Rejected", "Accepted"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=axes, fraction=0.025)
    fig.savefig(os.path.join(output_dir, "fig3_confusion_matrices.png"), dpi=150)
    plt.close(fig)

    # Figure 4: Correlation matrix.
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = df[FEATURE_COLUMNS + ["label"]].corr().values
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    labels = FEATURE_COLUMNS + ["label"]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.025)
    ax.set_title("Feature Correlation Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig4_correlation_heatmap.png"), dpi=150)
    plt.close(fig)

    # Figure 5: Decision tree importance.
    dt = bundle.models["Decision Tree"].named_steps["clf"]
    importances = dt.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(np.arange(len(FEATURE_COLUMNS)), importances[sorted_idx], color="#3498db")
    ax.set_xticks(np.arange(len(FEATURE_COLUMNS)))
    ax.set_xticklabels([FEATURE_COLUMNS[i] for i in sorted_idx], rotation=45, ha="right", fontsize=8)
    ax.set_title("Decision Tree Feature Importance")
    ax.set_ylabel("Importance")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig5_feature_importance.png"), dpi=150)
    plt.close(fig)


def print_training_summary(bundle: TrainingBundle) -> None:
    print("\n" + "=" * 72)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 72)
    for name, m in bundle.metrics.items():
        print(f"\n{name}")
        print(f"  Accuracy : {m['Accuracy'] * 100:.2f}%")
        print(f"  Precision: {m['Precision'] * 100:.2f}%")
        print(f"  Recall   : {m['Recall'] * 100:.2f}%")
        print(f"  F1 Score : {m['F1 Score'] * 100:.2f}%")
        print("  Classification Report:")
        print(m["Classification Report"])


def main() -> None:
    print("\n" + "=" * 72)
    print("RESEARCH PAPER ACCEPTANCE CLASSIFIER")
    print("=" * 72)

    df = generate_dataset(n_samples=600, seed=42)
    bundle = train_models(df, seed=42)

    print_training_summary(bundle)
    generate_visualizations(df, bundle, output_dir="outputs")
    print("\nSaved visualizations in outputs/")

    demo_abstract = (
        "We propose a novel learning framework with consistent improvements. "
        "Extensive experiments demonstrate robust performance across benchmarks."
    )
    predictions, features = predict_with_models(
        abstract=demo_abstract,
        author_count=4,
        title_length=11,
        citation_density=0.08,
        models=bundle.models,
    )

    print("\nDemo prediction:")
    print(predictions.to_string(index=False))
    print("\nExtracted input features:")
    print(features.to_string(index=False))


if __name__ == "__main__":
    main()
