
"""Train complaint-category and sentiment classification models.

The category model can use a labelled CSV containing:
- narrative: complaint text
- product: complaint category

The bundled sentiment examples are synthetic demonstration data.
They must not be used as evidence of real-world sentiment performance.
"""

from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


SENTIMENT_EXAMPLES = {
    "Negative": [
        "I am extremely disappointed with this terrible service",
        "Nobody has helped me and the issue remains unresolved",
        "This is frustrating and unacceptable",
        "I have been waiting for weeks with no response",
        "The service was very poor and I want a solution",
        "I was charged incorrectly and support ignored me",
        "Repeated failures have caused serious inconvenience",
        "Very unhappy with the careless handling of my complaint",
        "The problem is still happening and I am dissatisfied",
        "This delay is unacceptable and stressful"
    ],
    "Neutral": [
        "Please provide an update on my application",
        "I submitted a request yesterday for review",
        "The issue occurred on Monday afternoon",
        "I would like to know the current status",
        "Please check the attached transaction reference",
        "The service request number is 12345",
        "Kindly inform me about the next steps",
        "I am reporting an issue with my account details",
        "The complaint was registered last week",
        "Please confirm whether the request was received"
    ],
    "Positive": [
        "Thank you for resolving my issue quickly",
        "I appreciate the helpful support from your team",
        "The problem has been fixed and service is working well",
        "Excellent response and very helpful staff",
        "I am satisfied with the resolution provided",
        "Thanks for the quick update and assistance",
        "The refund arrived and everything is correct now",
        "Good service and a smooth resolution process",
        "Your team handled my request professionally",
        "I am happy with the support I received"
    ]
}


def flatten(mapping):
    texts, labels = [], []

    for label, samples in mapping.items():
        texts.extend(samples)
        labels.extend([label] * len(samples))

    return texts, labels


def evaluate_models(texts, labels, name, data_note):
    """Compare classifiers and save the best-performing pipeline."""

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels
    )

    candidates = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(
            max_iter=1500,
            class_weight="balanced"
        ),
        "Linear SVM": LinearSVC(
            class_weight="balanced",
            random_state=42
        )
    }

    reports = {}
    best_name = None
    best_score = -1
    best_pipeline = None

    all_labels = sorted(set(labels))

    for model_name, classifier in candidates.items():

        pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    stop_words="english",
                    sublinear_tf=True
                )
            ),
            ("classifier", classifier)
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )

        reports[model_name] = {
            "accuracy": accuracy_score(y_test, predictions),
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": f1,
            "classification_report": classification_report(
                y_test,
                predictions,
                zero_division=0
            ),
            "confusion_matrix": confusion_matrix(
                y_test,
                predictions,
                labels=all_labels
            ).tolist(),
            "labels": all_labels
        }

        if f1 > best_score:
            best_name = model_name
            best_score = f1
            best_pipeline = pipeline

    # Retrain the winning model on all labelled records.
    best_pipeline.fit(texts, labels)

    model_path = MODEL_DIR / f"{name}_model.joblib"
    joblib.dump(best_pipeline, model_path)

    return {
        "selected_model": best_name,
        "models": reports,
        "number_of_records": len(texts),
        "number_of_categories": len(all_labels),
        "data_note": data_note
    }


def train_category_from_csv(csv_path):
    """Train the category model from a labelled complaint CSV."""

    df = pd.read_csv(csv_path)

    required_columns = {"narrative", "product"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "CSV must contain both 'narrative' and 'product' columns."
        )

    df = df[["narrative", "product"]].dropna().copy()

    df["narrative"] = df["narrative"].astype(str).str.strip()
    df["product"] = df["product"].astype(str).str.strip()

    df = df[
        (df["narrative"] != "") &
        (df["product"] != "")
    ]

    # Every category needs at least two examples for stratified splitting.
    counts = df["product"].value_counts()
    valid_categories = counts[counts >= 2].index

    df = df[df["product"].isin(valid_categories)]

    if df["product"].nunique() < 2:
        raise ValueError(
            "The CSV must contain at least two categories, "
            "with at least two records in each category."
        )

    texts = df["narrative"].tolist()
    labels = df["product"].tolist()

    result = evaluate_models(
        texts,
        labels,
        "category",
        "Real labelled CSV dataset. Check its source and licence "
        "before reporting academic results."
    )

    results_path = MODEL_DIR / "category_training_results.json"

    results_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8"
    )

    return result


def train_demo_sentiment_model():
    """Train the separate demonstration sentiment model."""

    texts, labels = flatten(SENTIMENT_EXAMPLES)

    result = evaluate_models(
        texts,
        labels,
        "sentiment",
        "Synthetic demonstration examples; not valid evidence "
        "of real-world performance."
    )

    results_path = MODEL_DIR / "sentiment_training_results.json"

    results_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8"
    )

    return result


if __name__ == "__main__":
    print("Training the demonstration sentiment model only.")

    sentiment_result = train_demo_sentiment_model()

    print(
        "Sentiment model:",
        sentiment_result["selected_model"]
    )

    print(
        "The category model was NOT changed."
    )

    print(
        "Use train_category_from_csv() to train it with your CSV dataset."
    )
