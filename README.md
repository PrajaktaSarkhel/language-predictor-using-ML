# 🌐 Multilingual Language Detection Using NLP & Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.4+-orange.svg)](https://scikit-learn.org/)
[![Model Accuracy](https://img.shields.io/badge/Accuracy-98.50%25-yellow.svg)]()
[![Supported Languages](https://img.shields.io/badge/Languages-28%20Global%20Classes-purple.svg)]()
[![Framework](https://img.shields.io/badge/Web%20UI-FastAPI%20%2B%20REST%20API-teal.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

An end-to-end Natural Language Processing (NLP) and Machine Learning system that predicts text language across **28 classes**. Built with **scikit-learn**, **TF-IDF character n-gram features**, **Logistic Regression** classifier, a responsive static frontend, and a **FastAPI** JSON API.

---

## 📌 Project Overview & Motivation

Language detection is the critical backbone of multilingual software products:
* **Customer Support Routing**: Instantly classifies incoming helpdesk tickets into appropriate language queues.
* **Content Filtering & Moderation**: Directs regional text through locale-specific compliance filters.
* **Search & Document Retrieval**: Indexes foreign documents by language before semantic embedding.
* **Translation Pipeline Bootstrapping**: Automatically identifies source language before invoking downstream machine translation models (such as NLLB, MarianMT, or Gemini).

---

## 🚀 Key Features

* **Multi-Class NLP Classification Across 28 Languages**: Covers major European, Asian, Middle Eastern, and Dravidian languages.
* **Measured held-out accuracy (98.50%)**: The current split result is documented below; the requested 99% accuracy target has not yet been met.
* **Notebook-based training**: A single self-contained notebook contains preprocessing, model training, evaluation, and artifact export.
* **Unicode-Aware Preprocessing**: Preserves combining marks (matras, accents, diacritics) for non-Latin scripts (Devanagari, Tamil, Arabic, Cyrillic, CJK, Thai) while removing noise, citation numbers, and URLs.
* **Interactive Web Application**: Clean, modern web interface with real-time prediction, probability breakdown bars, and 1-click sample presets.
* **Production-Ready REST API**: Microservice-ready endpoints (`/api/predict`, `/api/languages`, `/health`).
* **FastAPI backend**: Serves the static frontend and provides JSON prediction, languages, and health endpoints.

---

## 🏗️ Architecture & Data Pipeline

```
Raw Multilingual Text
       │
       ▼
┌──────────────────────────────────────────────┐
│  1. Preprocessing & Normalization            │
│  - Unicode NFKC Normalization                │
│  - Strips URLs, Emails, Citation Brackets    │
│  - Retains Unicode Letters (\p{L}) & Marks   │
│  - Case-Folding & Whitespace Normalization   │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  2. Feature Extraction (TF-IDF Vectorizer)   │
│  - Sublinear Term Frequency Scaling          │
│  - Character n-grams (range: 2 to 5)         │
│  - 50,000 Discriminative Feature Dimensions  │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  3. Multi-Class NLP Classifier               │
│  - Logistic Regression (Multinomial Softmax)    │
│  - Serialized via Joblib                     │
└──────────────────────┬───────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
┌──────────────────┐       ┌───────────────────┐
│ Static Web UI    │       │ FastAPI REST API  │
│ (Real-Time Live) │       │ (/api/predict)    │
└──────────────────┘       └───────────────────┘
```

---

## 📊 Model Evaluation & Benchmarks

The notebook evaluates a stratified **80/20 train-test split** of 32,098 multilingual text samples using Logistic Regression only. On the previously supplied split, Logistic Regression measured **98.50% accuracy**, **98.85% macro precision**, **98.28% macro recall**, and **98.55% macro F1**. The accuracy remains below the requested 99% target.

| Model Architecture | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** (character TF-IDF) | **98.50%** | **98.85%** | **98.28%** | **98.55%** |

This is the measured result from the previously run held-out split. Run the notebook to reproduce it. External validation on new labeled examples is needed before treating it as a production generalization estimate.

### Confusion Matrix
Run all notebook cells to inspect the held-out confusion matrix.

---

## 🌍 Supported Languages (28 Classes)

| Flag | Language | ISO Code | Native Script | Language Family |
| :---: | :--- | :---: | :--- | :--- |
| 🇸🇦 | Arabic | `ar` | العربية | Afroasiatic (Semitic) |
| 🇨🇳 | Chinese | `zh` | 中文 | Sino-Tibetan |
| 🇩🇰 | Danish | `da` | Dansk | Indo-European (North Germanic) |
| 🇳🇱 | Dutch | `nl` | Nederlands | Indo-European (West Germanic) |
| 🇬🇧 | English | `en` | English | Indo-European (West Germanic) |
| 🇪🇪 | Estonian | `et` | Eesti keel | Uralic (Finno-Ugric) |
| 🇫🇷 | French | `fr` | Français | Indo-European (Romance) |
| 🇩🇪 | German | `de` | Deutsch | Indo-European (West Germanic) |
| 🇬🇷 | Greek | `el` | Ελληνικά | Indo-European (Hellenic) |
| 🇮🇳 | Hindi | `hi` | हिन्दी | Indo-European (Indo-Aryan) |
| 🇮🇩 | Indonesian | `id` | Bahasa Indonesia | Austronesian |
| 🇮🇹 | Italian | `it` | Italiano | Indo-European (Romance) |
| 🇯🇵 | Japanese | `ja` | 日本語 | Japonic |
| 🇮🇳 | Kannada | `kn` | ಕನ್ನಡ | Dravidian (Southern) |
| 🇰🇷 | Korean | `ko` | 한국어 | Koreanic |
| 🏛️ | Latin | `la` | Lingua Latina | Indo-European (Italic) |
| 🇮🇳 | Malayalam | `ml` | മലയാളം | Dravidian (Southern) |
| 🇮🇷 | Persian | `fa` | فارسی | Indo-European (Iranian) |
| 🇵🇹 | Portuguese | `pt` | Português | Indo-European (Romance) |
| 🇦🇫 | Pushto | `ps` | پښتو | Indo-European (Iranian) |
| 🇷🇴 | Romanian | `ro` | Română | Indo-European (Romance) |
| 🇷🇺 | Russian | `ru` | Русский | Indo-European (Slavic) |
| 🇪🇸 | Spanish | `es` | Español | Indo-European (Romance) |
| 🇸🇪 | Swedish | `sv` | Svenska | Indo-European (North Germanic) |
| 🇮🇳 | Tamil | `ta` | தமிழ் | Dravidian (Southern) |
| 🇹🇭 | Thai | `th` | ภาษาไทย | Kra-Dai |
| 🇹🇷 | Turkish | `tr` | Türkçe | Turkic |
| 🇵🇰 | Urdu | `ur` | اردو | Indo-European (Indo-Aryan) |

---

## 📁 Repository Structure

```text
.
├── app.py                              # FastAPI backend
├── requirements.txt
├── data/processed/language_dataset.csv # Consolidated 28-class dataset
├── models/language_detector_pipeline.joblib
├── notebooks/language_detection.ipynb  # Complete ML workflow
├── static/css/style.css
├── static/js/app.js
└── templates/index.html                # Static frontend page
```

---

## 🛠️ Installation & Setup Guide

### 1. Clone or Open the Repository
```bash
git clone https://github.com/<YOUR-USERNAME>/Language-Detection-Using-NLP-and-Machine-Learning.git
cd Language-Detection-Using-NLP-and-Machine-Learning
```

### 2. Create and Activate a Virtual Environment
```bash
# On Windows:
python -m venv venv
venv\Scripts\activate

# On macOS/Linux:
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Retrain the Model
Run every cell in `notebooks/language_detection.ipynb` to preprocess, train, evaluate, and export the model artifact. The notebook is the source of truth for the ML workflow.

### 5. Start the Web Application
```bash
uvicorn app:app --reload
```
Open **`http://127.0.0.1:8000`**. Run all cells in the notebook first to regenerate the model artifact.

---

## 🔌 REST API Documentation

### Detect Language
* **Endpoint:** `POST /api/predict`
* **Content-Type:** `application/json`

#### Request:
```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Bonjour tout le monde, comment allez-vous aujourd hui?", "top_k": 3}'
```

#### Response:
```json
{
  "success": true,
  "predicted_language": "French",
  "confidence": 98.74,
  "code": "fr",
  "native": "Français",
  "flag": "🇫🇷",
  "family": "Indo-European (Romance)",
  "script": "Latin",
  "top_predictions": [
    {
      "language": "French",
      "confidence_percent": 98.74,
      "native": "Français",
      "flag": "🇫🇷",
      "code": "fr"
    },
    {
      "language": "Spanish",
      "confidence_percent": 0.82,
      "native": "Español",
      "flag": "🇪🇸",
      "code": "es"
    }
  ],
  "cleaned_text": "bonjour tout le monde comment allez vous aujourd hui"
}
```

---

## 🚀 How to Push This Project to GitHub

Follow these exact commands to publish your project to GitHub:

### Step 1: Create a New GitHub Repository
1. Go to [GitHub](https://github.com/new).
2. Set the repository name to `Language-Detection-Using-NLP-and-Machine-Learning` (or any name you prefer).
3. Keep it **Public** (or Private) and leave **"Add a README file"**, **".gitignore"**, and **"license"** **UNCHECKED** (since we already created them).
4. Click **Create repository**.

### Step 2: Initialize Git and Push from Terminal
Run the following commands in your project root directory (`c:\Users\praja\OneDrive\Desktop\My Projects\Fsp project`):

```bash
# 1. Initialize local git repository
git init

# 2. Add all files to staging
git add .

# 3. Create your initial commit
git commit -m "feat: complete end-to-end language detection project with NLP, ML models, and FastAPI web app"

# 4. Set main as the default branch
git branch -M main

# 5. Link your local repo to your GitHub repository
# (Replace YOUR_USERNAME and YOUR_REPO with your actual GitHub link)
git remote add origin https://github.com/YOUR_USERNAME/Language-Detection-Using-NLP-and-Machine-Learning.git

# 6. Push the code to GitHub
git push -u origin main
```

---

## 📜 References & Acknowledgments
* Reference implementation ideas: [akhiilkasare/Language-Detection-Using-NLP-and-Machine-Learning](https://github.com/akhiilkasare/Language-Detection-Using-NLP-and-Machine-Learning)
* Dataset sources: European Parliament Proceedings Parallel Corpus & multilingual language classification datasets.
* Scikit-Learn Documentation: [scikit-learn.org](https://scikit-learn.org/)
