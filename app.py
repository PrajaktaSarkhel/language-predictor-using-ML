"""FastAPI service and static frontend for the notebook-trained language model."""

import re
import unicodedata
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "language_detector_pipeline.joblib"
app = FastAPI(title="Language Detection API", version="1.0.0")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
detector = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

LANGUAGE_METADATA = {
    "Arabic": ("ar", "العربية", "🇸🇦"), "Chinese": ("zh", "中文", "🇨🇳"),
    "Danish": ("da", "Dansk", "🇩🇰"), "Dutch": ("nl", "Nederlands", "🇳🇱"),
    "English": ("en", "English", "🇬🇧"), "Estonian": ("et", "Eesti keel", "🇪🇪"),
    "French": ("fr", "Français", "🇫🇷"), "German": ("de", "Deutsch", "🇩🇪"),
    "Greek": ("el", "Ελληνικά", "🇬🇷"), "Hindi": ("hi", "हिन्दी", "🇮🇳"),
    "Indonesian": ("id", "Bahasa Indonesia", "🇮🇩"), "Italian": ("it", "Italiano", "🇮🇹"),
    "Japanese": ("ja", "日本語", "🇯🇵"), "Kannada": ("kn", "ಕನ್ನಡ", "🇮🇳"),
    "Korean": ("ko", "한국어", "🇰🇷"), "Latin": ("la", "Lingua Latina", "🏛️"),
    "Malayalam": ("ml", "മലയാളം", "🇮🇳"), "Persian": ("fa", "فارسی", "🇮🇷"),
    "Portuguese": ("pt", "Português", "🇵🇹"), "Pushto": ("ps", "پښتو", "🇦🇫"),
    "Romanian": ("ro", "Română", "🇷🇴"), "Russian": ("ru", "Русский", "🇷🇺"),
    "Spanish": ("es", "Español", "🇪🇸"), "Swedish": ("sv", "Svenska", "🇸🇪"),
    "Tamil": ("ta", "தமிழ்", "🇮🇳"), "Thai": ("th", "ภาษาไทย", "🇹🇭"),
    "Turkish": ("tr", "Türkçe", "🇹🇷"), "Urdu": ("ur", "اردو", "🇵🇰"),
}


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    top_k: int = Field(default=4, ge=1, le=10)


def clean_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = re.sub(r"https?://\S+|www\.\S+|\S+@\S+|\[\d+\]", " ", value)
    value = re.sub(r"['\"`.,!?;:()\[\]{}<>/\\|@#$%^&*~+=_\-0-9]", " ", value)
    kept = "".join(ch.lower() if unicodedata.category(ch).startswith(("L", "M")) else " " for ch in value)
    return re.sub(r"\s+", " ", kept).strip()


@app.get("/", response_class=HTMLResponse)
def index():
    page = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    # The existing page is static HTML with a few Flask template expressions.
    page = re.sub(r"\{\{\s*url_for\('static',\s*filename='([^']+)'\)\s*\}\}", r"/static/\1", page)
    return page


@app.post("/api/predict")
def predict(payload: PredictionRequest):
    if detector is None:
        raise HTTPException(status_code=503, detail="Model artifact missing. Run the training notebook first.")
    normalized = clean_text(payload.text)
    if len(normalized) < 3:
        raise HTTPException(status_code=400, detail="Enter at least three letters.")
    probs = detector.predict_proba([normalized])[0]
    order = probs.argsort()[::-1][:min(payload.top_k, len(probs))]
    best = int(order[0])
    language = str(detector.classes_[best])
    code, native, flag = LANGUAGE_METADATA.get(language, ("und", language, "🌐"))
    return {"success": True, "predicted_language": language, "confidence": round(float(probs[best]) * 100, 2),
            "code": code, "native": native, "flag": flag,
            "top_predictions": [{"language": str(detector.classes_[i]), "confidence_percent": round(float(probs[i]) * 100, 2),
                                 "code": LANGUAGE_METADATA.get(str(detector.classes_[i]), ("und",))[0],
                                 "native": LANGUAGE_METADATA.get(str(detector.classes_[i]), ("und", str(detector.classes_[i])))[1],
                                 "flag": LANGUAGE_METADATA.get(str(detector.classes_[i]), ("und", "", "🌐"))[2]} for i in order]}


@app.get("/api/languages")
def languages():
    names = sorted(detector.classes_) if detector is not None else []
    return {"total_languages": len(names), "languages": [{"name": n, "code": LANGUAGE_METADATA.get(n, ("und",))[0],
            "native": LANGUAGE_METADATA.get(n, ("und", n))[1], "flag": LANGUAGE_METADATA.get(n, ("und", "", "🌐"))[2]} for n in names]}


@app.get("/health")
def health():
    return {"status": "healthy" if detector is not None else "model_missing", "model_loaded": detector is not None,
            "supported_classes": len(detector.classes_) if detector is not None else 0}
