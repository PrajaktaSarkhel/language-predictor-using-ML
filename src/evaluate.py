"""
Evaluation script for language detection model.
Computes comprehensive metrics, per-class classification reports,
and error analysis on top confused language pairs.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from src.preprocess import clean_text


def evaluate_model(
    test_csv_path: str = "data/processed/cleaned_dataset.csv",
    model_path: str = "models/language_detector_pipeline.joblib",
    metadata_path: str = "models/model_metadata.json",
):
    print("=" * 60)
    print("📊 LANGUAGE DETECTION MODEL EVALUATION REPORT")
    print("=" * 60)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}.")
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test dataset not found at {test_csv_path}.")

    print(f"\nLoading model from: {model_path}")
    pipeline = joblib.load(model_path)

    print(f"Loading evaluation dataset from: {test_csv_path}")
    df = pd.read_csv(test_csv_path)

    X = df["cleaned_text"]
    y_true = df["language"]
    languages = sorted(y_true.unique().tolist())

    print(f"Total samples for evaluation: {len(df)} across {len(languages)} languages.")

    print("\nRunning inference...")
    y_pred = pipeline.predict(X)

    acc = accuracy_score(y_true, y_pred)
    print(f"\n🎯 Overall Evaluation Accuracy: {acc * 100:.2f}%\n")

    report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    print("Detailed Classification Report:")
    print("-" * 60)
    print(report)

    # Error Analysis: Top confused pairs
    print("\n🔍 Error Analysis (Top Confused Language Pairs):")
    print("-" * 60)
    cm = confusion_matrix(y_true, y_pred, labels=languages)
    confusions = []
    for i, true_lang in enumerate(languages):
        for j, pred_lang in enumerate(languages):
            if i != j and cm[i, j] > 0:
                confusions.append((true_lang, pred_lang, cm[i, j]))

    confusions.sort(key=lambda x: x[2], reverse=True)
    if confusions:
        print(f"{'True Language':<15} {'Predicted As':<15} {'Errors':<10}")
        print("-" * 40)
        for true_l, pred_l, count in confusions[:10]:
            print(f"{true_l:<15} {pred_l:<15} {count:<10}")
    else:
        print("No confusion pairs found! Model made 0 errors.")

    print("\n" + "=" * 60)
    print("✅ EVALUATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_model()
