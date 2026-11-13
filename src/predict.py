"""
Inference module for Language Detection.
Loads the serialized Scikit-learn Pipeline and provides fast predictions
with confidence scores and linguistic metadata (flags, scripts, language families).
"""

import os
from typing import Any, Dict, List, Optional
import joblib
import numpy as np

from src.preprocess import clean_text

LANGUAGE_METADATA = {
    "Arabic": {
        "code": "ar",
        "native": "العربية",
        "flag": "🇸🇦",
        "family": "Afroasiatic (Semitic)",
        "script": "Arabic",
    },
    "Chinese": {
        "code": "zh",
        "native": "中文",
        "flag": "🇨🇳",
        "family": "Sino-Tibetan",
        "script": "Hanzi (Logographic)",
    },
    "Danish": {
        "code": "da",
        "native": "Dansk",
        "flag": "🇩🇰",
        "family": "Indo-European (North Germanic)",
        "script": "Latin",
    },
    "Dutch": {
        "code": "nl",
        "native": "Nederlands",
        "flag": "🇳🇱",
        "family": "Indo-European (West Germanic)",
        "script": "Latin",
    },
    "English": {
        "code": "en",
        "native": "English",
        "flag": "🇬🇧",
        "family": "Indo-European (West Germanic)",
        "script": "Latin",
    },
    "Estonian": {
        "code": "et",
        "native": "Eesti keel",
        "flag": "🇪🇪",
        "family": "Uralic (Finno-Ugric)",
        "script": "Latin",
    },
    "French": {
        "code": "fr",
        "native": "Français",
        "flag": "🇫🇷",
        "family": "Indo-European (Romance)",
        "script": "Latin",
    },
    "German": {
        "code": "de",
        "native": "Deutsch",
        "flag": "🇩🇪",
        "family": "Indo-European (West Germanic)",
        "script": "Latin",
    },
    "Greek": {
        "code": "el",
        "native": "Ελληνικά",
        "flag": "🇬🇷",
        "family": "Indo-European (Hellenic)",
        "script": "Greek",
    },
    "Hindi": {
        "code": "hi",
        "native": "हिन्दी",
        "flag": "🇮🇳",
        "family": "Indo-European (Indo-Aryan)",
        "script": "Devanagari",
    },
    "Indonesian": {
        "code": "id",
        "native": "Bahasa Indonesia",
        "flag": "🇮🇩",
        "family": "Austronesian (Malayo-Polynesian)",
        "script": "Latin",
    },
    "Italian": {
        "code": "it",
        "native": "Italiano",
        "flag": "🇮🇹",
        "family": "Indo-European (Romance)",
        "script": "Latin",
    },
    "Japanese": {
        "code": "ja",
        "native": "日本語",
        "flag": "🇯🇵",
        "family": "Japonic",
        "script": "Kanji / Hiragana / Katakana",
    },
    "Kannada": {
        "code": "kn",
        "native": "ಕನ್ನಡ",
        "flag": "🇮🇳",
        "family": "Dravidian (Southern)",
        "script": "Kannada",
    },
    "Korean": {
        "code": "ko",
        "native": "한국어",
        "flag": "🇰🇷",
        "family": "Koreanic",
        "script": "Hangul",
    },
    "Latin": {
        "code": "la",
        "native": "Lingua Latina",
        "flag": "🏛️",
        "family": "Indo-European (Italic)",
        "script": "Latin",
    },
    "Malayalam": {
        "code": "ml",
        "native": "മലയാളം",
        "flag": "🇮🇳",
        "family": "Dravidian (Southern)",
        "script": "Malayalam",
    },
    "Persian": {
        "code": "fa",
        "native": "فارسی",
        "flag": "🇮🇷",
        "family": "Indo-European (Iranian)",
        "script": "Perso-Arabic",
    },
    "Portuguese": {
        "code": "pt",
        "native": "Português",
        "flag": "🇵🇹",
        "family": "Indo-European (Romance)",
        "script": "Latin",
    },
    "Pushto": {
        "code": "ps",
        "native": "پښتو",
        "flag": "🇦🇫",
        "family": "Indo-European (Iranian)",
        "script": "Perso-Arabic",
    },
    "Romanian": {
        "code": "ro",
        "native": "Română",
        "flag": "🇷🇴",
        "family": "Indo-European (Romance)",
        "script": "Latin",
    },
    "Russian": {
        "code": "ru",
        "native": "Русский",
        "flag": "🇷🇺",
        "family": "Indo-European (Slavic)",
        "script": "Cyrillic",
    },
    "Spanish": {
        "code": "es",
        "native": "Español",
        "flag": "🇪🇸",
        "family": "Indo-European (Romance)",
        "script": "Latin",
    },
    "Swedish": {
        "code": "sv",
        "native": "Svenska",
        "flag": "🇸🇪",
        "family": "Indo-European (North Germanic)",
        "script": "Latin",
    },
    "Tamil": {
        "code": "ta",
        "native": "தமிழ்",
        "flag": "🇮🇳",
        "family": "Dravidian (Southern)",
        "script": "Tamil",
    },
    "Thai": {
        "code": "th",
        "native": "ภาษาไทย",
        "flag": "🇹🇭",
        "family": "Kra-Dai",
        "script": "Thai",
    },
    "Turkish": {
        "code": "tr",
        "native": "Türkçe",
        "flag": "🇹🇷",
        "family": "Turkic",
        "script": "Latin",
    },
    "Urdu": {
        "code": "ur",
        "native": "اردو",
        "flag": "🇵🇰",
        "family": "Indo-European (Indo-Aryan)",
        "script": "Perso-Arabic (Nastaliq)",
    },
}


class LanguageDetector:
    """Wrapper class for loading serialized model pipeline and running predictions."""

    def __init__(self, model_path: str = "models/language_detector_pipeline.joblib"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at '{model_path}'. Please run 'python -m src.train' first."
            )
        self.model_path = model_path
        self.pipeline = joblib.load(model_path)
        self.classes_ = list(self.pipeline.classes_)

    def predict(self, text: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Predict the language of a single text input.
        Returns predicted language, confidence score, and top-k predictions with metadata.
        """
        if not text or not text.strip():
            return {
                "success": False,
                "error": "Input text is empty. Please enter or paste some text.",
            }

        cleaned = clean_text(text)
        if len(cleaned) < 2:
            return {
                "success": False,
                "error": "Input text contains insufficient characters after cleaning for reliable language detection.",
            }

        # Predict probability distribution
        probs = self.pipeline.predict_proba([cleaned])[0]
        top_indices = np.argsort(probs)[::-1][:top_k]

        top_predictions = []
        for idx in top_indices:
            lang_name = self.classes_[idx]
            score = float(probs[idx])
            meta = LANGUAGE_METADATA.get(
                lang_name,
                {"code": "und", "native": lang_name, "flag": "🌐", "family": "Unknown", "script": "Unknown"},
            )
            top_predictions.append({
                "language": lang_name,
                "probability": round(score, 4),
                "confidence_percent": round(score * 100, 2),
                **meta,
            })

        best = top_predictions[0]
        return {
            "success": True,
            "raw_text": text,
            "cleaned_text": cleaned,
            "predicted_language": best["language"],
            "confidence": best["confidence_percent"],
            "code": best["code"],
            "native": best["native"],
            "flag": best["flag"],
            "family": best["family"],
            "script": best["script"],
            "top_predictions": top_predictions,
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict languages for a batch of strings."""
        return [self.predict(t) for t in texts]


# Singleton instance for app reuse
_detector_instance: Optional[LanguageDetector] = None


def get_detector(model_path: str = "models/language_detector_pipeline.joblib") -> LanguageDetector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = LanguageDetector(model_path)
    return _detector_instance
