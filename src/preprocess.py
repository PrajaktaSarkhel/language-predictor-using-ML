"""
Text preprocessing and normalization pipeline for Language Detection.
Supports Unicode letters, combining marks (diacritics, matras, accents),
and multiple writing systems (Latin, Cyrillic, Arabic, Devanagari, Tamil, CJK, etc.).
"""

import re
import unicodedata
from typing import List, Union
import pandas as pd


def normalize_unicode(text: str) -> str:
    """Normalize text using Unicode NFKC (Compatibility Decomposition followed by Canonical Composition)."""
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFKC", text)


def clean_text(text: str) -> str:
    """
    Cleans raw input text:
      - Normalizes Unicode representations.
      - Removes URLs, emails, and citation brackets (e.g., [1], [14]).
      - Replaces common punctuation with spaces to avoid concatenating distinct words.
      - Retains letters (category 'L') and combining marks (category 'M') across all scripts.
      - Folds case to lowercase.
      - Normalizes repeated whitespaces.
    """
    if not isinstance(text, str):
        return ""

    # 1. Unicode normalization
    text = normalize_unicode(text)

    # 2. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 3. Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # 4. Remove citation numbers like [1], [23]
    text = re.sub(r"\[\d+\]", " ", text)

    # 5. Replace common punctuation & symbols with spaces (preserves word boundaries)
    text = re.sub(r"['\"`.,!?;:()\[\]{}<>/\\|@#$%^&*~+=_\-0-9]", " ", text)

    # 6. Retain characters in Unicode Letter ('L') or Mark ('M') categories, plus whitespace
    filtered_chars = [
        char
        for char in text
        if unicodedata.category(char).startswith(("L", "M")) or char.isspace()
    ]
    cleaned = "".join(filtered_chars).lower()

    # 7. Collapse multiple spaces into single space
    return re.sub(r"\s+", " ", cleaned).strip()


def preprocess_corpus(
    df: pd.DataFrame,
    text_col: str = "Text",
    label_col: str = "language",
    min_length: int = 3,
) -> pd.DataFrame:
    """
    Preprocesses a DataFrame containing text and language labels.
    Removes null values, applies clean_text, and drops rows shorter than min_length.
    """
    df = df.copy()

    # Drop nulls
    df = df.dropna(subset=[text_col, label_col])

    # Clean text
    df["cleaned_text"] = df[text_col].apply(clean_text)

    # Filter out empty or trivially short strings
    df = df[df["cleaned_text"].str.len() >= min_length]

    # Standardize label casing and strip spaces
    df[label_col] = df[label_col].astype(str).str.strip().str.title()

    # Common spelling normalizations (e.g. Portugese -> Portuguese, Sweedish -> Swedish)
    spelling_fixes = {
        "Portugese": "Portuguese",
        "Portugeese": "Portuguese",
        "Sweedish": "Swedish",
    }
    df[label_col] = df[label_col].replace(spelling_fixes)

    return df
