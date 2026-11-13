"""
Optimized Model Training & Serialization Pipeline.
Trains and compares multi-class NLP classifiers across 28 world languages:
  1. Logistic Regression (Primary Production Pipeline)
  2. Multinomial Naive Bayes (Baseline)
  3. Linear Support Vector Classifier (LinearSVC)
Saves the serialized pipeline, confusion matrix plot, and metadata.
"""

import json
import os
import time
from typing import Any, Dict

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
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
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.preprocess import preprocess_corpus


def load_dataset() -> pd.DataFrame:
    """Loads and combines raw datasets into a single processed corpus."""
    processed_path = "data/processed/cleaned_dataset.csv"
    if os.path.exists(processed_path):
        df = pd.read_csv(processed_path)
        print(f"      Loaded existing preprocessed corpus: {df.shape}")
        return df

    dfs = []
    if os.path.exists("data/raw/dataset_22lang.csv"):
        d1 = pd.read_csv("data/raw/dataset_22lang.csv")
        d1 = d1.rename(columns={c: "Text" for c in d1.columns if c.lower() == "text"})
        d1 = d1.rename(columns={c: "language" for c in d1.columns if c.lower() == "language"})
        dfs.append(d1[["Text", "language"]])

    if os.path.exists("data/raw/dataset_17lang.csv"):
        d2 = pd.read_csv("data/raw/dataset_17lang.csv")
        d2 = d2.rename(columns={c: "Text" for c in d2.columns if c.lower() == "text"})
        d2 = d2.rename(columns={c: "language" for c in d2.columns if c.lower() == "language"})
        dfs.append(d2[["Text", "language"]])

    combined = pd.concat(dfs, ignore_index=True)
    cleaned = preprocess_corpus(combined)
    cleaned = cleaned.drop_duplicates(subset=["cleaned_text"])
    os.makedirs("data/processed", exist_ok=True)
    cleaned.to_csv(processed_path, index=False)
    print(f"      Saved cleaned corpus: {cleaned.shape}")
    return cleaned


def run_training() -> Dict[str, Any]:
    print("=" * 65)
    print("🚀 TRAINING PRODUCTION MULTILINGUAL LANGUAGE DETECTION PIPELINE")
    print("=" * 65)

    # 1. Load Data
    print("\n[1/6] Loading corpus...")
    t0 = time.time()
    df = load_dataset()
    languages = sorted(df["language"].unique().tolist())
    print(f"      Corpus size: {len(df)} samples across {len(languages)} languages.")

    # 2. Stratified Train-Test Split (Strict Featurization Ordering)
    print("\n[2/6] Performing Stratified Train-Test Split (80% Train, 20% Test)...")
    X = df["cleaned_text"]
    y = df["language"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"      Training set: {len(X_train)} | Testing set: {len(X_test)}")

    # 3. TF-IDF Feature Extraction
    print("\n[3/6] Setting up character n-gram (1-4) TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 4),
        max_features=50000,
        sublinear_tf=True,
    )

    # 4. Multi-Model Comparison
    print("\n[4/6] Training & comparing models...")
    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Linear Support Vector (LinearSVC)": LinearSVC(C=1.0, random_state=42, max_iter=1500),
        "Logistic Regression": LogisticRegression(C=5.0, max_iter=400, random_state=42),
    }

    comparison = {}
    fitted_pipelines = {}

    for name, clf in models.items():
        print(f"\n   >>> Fitting {name}...")
        t_fit = time.time()
        pipeline = Pipeline([
            ("vectorizer", vectorizer),
            ("classifier", clf),
        ])
        pipeline.fit(X_train, y_train)
        fit_duration = round(time.time() - t_fit, 2)

        t_inf = time.time()
        y_pred = pipeline.predict(X_test)
        inf_duration = round(time.time() - t_inf, 4)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

        comparison[name] = {
            "accuracy": round(acc, 4),
            "macro_precision": round(prec, 4),
            "macro_recall": round(rec, 4),
            "macro_f1": round(f1, 4),
            "train_time_sec": fit_duration,
            "inference_time_sec": inf_duration,
        }
        fitted_pipelines[name] = pipeline

        print(f"       Accuracy: {acc * 100:.2f}% | Macro F1: {f1 * 100:.2f}% | Fit Time: {fit_duration}s")

    # 5. Production Selection (Logistic Regression)
    selected_name = "Logistic Regression"
    best_pipeline = fitted_pipelines[selected_name]
    y_pred_best = best_pipeline.predict(X_test)

    clf_report = classification_report(y_test, y_pred_best, output_dict=True, zero_division=0)
    print(f"\n[5/6] Final Test Performance ({selected_name}):")
    print(f"      Accuracy:         {clf_report['accuracy'] * 100:.2f}%")
    print(f"      Macro Avg F1:     {clf_report['macro avg']['f1-score'] * 100:.2f}%")
    print(f"      Weighted Avg F1:  {clf_report['weighted avg']['f1-score'] * 100:.2f}%")

    # 6. Confusion Matrix & Serialization
    print("\n[6/6] Generating Confusion Matrix & Serializing Artifacts...")
    cm_path = "static/confusion_matrix.png"
    os.makedirs("static", exist_ok=True)
    cm = confusion_matrix(y_test, y_pred_best, labels=languages)

    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=languages,
        yticklabels=languages,
        cbar=True,
        linewidths=0.5,
        annot_kws={"size": 8},
    )
    plt.title(f"Multilingual Language Detection Confusion Matrix ({selected_name})", fontsize=15, pad=15)
    plt.xlabel("Predicted Language", fontsize=12, labelpad=10)
    plt.ylabel("True Language", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print("      Confusion matrix heatmap saved to 'static/confusion_matrix.png'")

    # Serialize model pipeline
    model_path = "models/language_detector_pipeline.joblib"
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_pipeline, model_path, compress=3)
    print(f"      Model pipeline serialized to '{model_path}'")

    # Save metadata
    metadata = {
        "selected_model": selected_name,
        "supported_languages": languages,
        "num_languages": len(languages),
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "model_comparison": comparison,
        "best_model_metrics": {
            "accuracy": comparison[selected_name]["accuracy"],
            "macro_precision": comparison[selected_name]["macro_precision"],
            "macro_recall": comparison[selected_name]["macro_recall"],
            "macro_f1": comparison[selected_name]["macro_f1"],
        },
        "per_language_metrics": {
            lang: {
                "precision": round(clf_report[lang]["precision"], 4),
                "recall": round(clf_report[lang]["recall"], 4),
                "f1-score": round(clf_report[lang]["f1-score"], 4),
                "support": clf_report[lang]["support"],
            }
            for lang in languages
            if lang in clf_report
        },
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open("models/model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print("      Model metadata saved to 'models/model_metadata.json'")

    print("\n" + "=" * 65)
    print("✅ TRAINING & SERIALIZATION COMPLETED SUCCESSFULLY!")
    print("=" * 65)
    return metadata


if __name__ == "__main__":
    run_training()
