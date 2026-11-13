"""
Flask Web Application & REST API for Real-Time Multilingual Language Detection.
Serves a responsive UI with real-time prediction, probability breakdown,
model metrics, and interactive sample presets.
"""

import json
import os
from flask import Flask, jsonify, render_template, request

from src.predict import LANGUAGE_METADATA, get_detector

app = Flask(__name__)

# Initialize the detector
detector = get_detector()

# Load model metadata if available
metadata = {}
meta_path = "models/model_metadata.json"
if os.path.exists(meta_path):
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load metadata: {e}")


@app.route("/")
def index():
    """Renders the main web interface."""
    return render_template(
        "index.html",
        languages=detector.classes_,
        metadata=LANGUAGE_METADATA,
        model_info=metadata.get("best_model_metrics", {}),
        comparison=metadata.get("model_comparison", {}),
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    """API endpoint to detect language from input text."""
    try:
        data = request.get_json(force=True, silent=True)
        if not data or "text" not in data:
            return jsonify({"success": False, "error": "Missing 'text' field in JSON payload."}), 400

        text = data.get("text", "")
        top_k = int(data.get("top_k", 4))

        result = detector.predict(text, top_k=top_k)
        if not result["success"]:
            return jsonify(result), 400

        return jsonify(result), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/languages", methods=["GET"])
def get_languages():
    """Returns all supported languages with codes, flags, and linguistic families."""
    lang_list = []
    for lang in detector.classes_:
        meta = LANGUAGE_METADATA.get(lang, {"code": "und", "native": lang, "flag": "🌐", "family": "Unknown", "script": "Unknown"})
        lang_list.append({"name": lang, **meta})
    return jsonify({"total_languages": len(lang_list), "languages": lang_list})


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """Returns model benchmark metrics and comparison data."""
    return jsonify(metadata)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": detector is not None,
        "supported_classes": len(detector.classes_),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Language Detection App running at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
