"""
Unit tests for the Language Detection NLP Pipeline & Model.
"""

import unittest
from src.preprocess import clean_text
from src.predict import get_detector, LANGUAGE_METADATA


class TestLanguageDetection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.detector = get_detector("models/language_detector_pipeline.joblib")

    def test_clean_text_basic(self):
        raw = "Hello World! Check out https://example.com [12] 123."
        cleaned = clean_text(raw)
        self.assertEqual(cleaned, "hello world check out")

    def test_clean_text_unicode_accents(self):
        raw = "C'est un été très chaud à Paris!"
        cleaned = clean_text(raw)
        self.assertTrue("été" in cleaned)
        self.assertTrue("très" in cleaned)

    def test_clean_text_indic_scripts(self):
        raw = "नमस्ते भारत! आप कैसे हैं?"
        cleaned = clean_text(raw)
        self.assertIn("नमस्ते", cleaned)
        self.assertIn("भारत", cleaned)

    def test_empty_and_short_inputs(self):
        res1 = self.detector.predict("")
        self.assertFalse(res1["success"])

        res2 = self.detector.predict("   ")
        self.assertFalse(res2["success"])

        res3 = self.detector.predict("a")
        self.assertFalse(res3["success"])

    def test_english_prediction(self):
        text = "Natural language processing is an essential component of artificial intelligence."
        res = self.detector.predict(text)
        self.assertTrue(res["success"])
        self.assertEqual(res["predicted_language"], "English")
        self.assertGreater(res["confidence"], 50.0)

    def test_french_prediction(self):
        text = "Bonjour tout le monde! C'est un réel plaisir de vous rencontrer aujourd'hui."
        res = self.detector.predict(text)
        self.assertTrue(res["success"])
        self.assertEqual(res["predicted_language"], "French")

    def test_spanish_prediction(self):
        text = "Hola amigos, este es un sistema inteligente de procesamiento de texto."
        res = self.detector.predict(text)
        self.assertTrue(res["success"])
        self.assertEqual(res["predicted_language"], "Spanish")

    def test_german_prediction(self):
        text = "Guten Tag, das maschinelle Lernen bietet erstaunliche Möglichkeiten für die Zukunft."
        res = self.detector.predict(text)
        self.assertTrue(res["success"])
        self.assertEqual(res["predicted_language"], "German")

    def test_hindi_prediction(self):
        text = "नमस्ते दोस्तों, प्राकृतिक भाषा प्रसंस्करण तकनीक कंप्यूटर को मानवीय भाषा सिखाती है।"
        res = self.detector.predict(text)
        self.assertTrue(res["success"])
        self.assertEqual(res["predicted_language"], "Hindi")

    def test_russian_prediction(self):
        text = "Привет мир! Машинное обучение и искусственный интеллект меняют современные технологии."
        res = self.detector.predict(text)
        self.assertTrue(res["success"])
        self.assertEqual(res["predicted_language"], "Russian")

    def test_arabic_prediction(self):
        text = "مرحبا بكم في هذا المشروع المتطور للتعرف التلقائي على اللغات الطبيعية."
        res = self.detector.predict(text)
        self.assertTrue(res["success"])
        self.assertEqual(res["predicted_language"], "Arabic")

    def test_japanese_prediction(self):
        text = "こんにちは、機械学習による高精度な言語判定システムをテストしています。"
        res = self.detector.predict(text)
        self.assertTrue(res["success"])
        self.assertEqual(res["predicted_language"], "Japanese")

    def test_top_k_structure(self):
        text = "Artificial intelligence and deep learning"
        res = self.detector.predict(text, top_k=4)
        self.assertTrue(res["success"])
        self.assertEqual(len(res["top_predictions"]), 4)
        for pred in res["top_predictions"]:
            self.assertIn("language", pred)
            self.assertIn("confidence_percent", pred)
            self.assertIn("flag", pred)


if __name__ == "__main__":
    unittest.main()
