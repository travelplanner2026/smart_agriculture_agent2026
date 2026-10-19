"""
Automated Test Suite for Crop Disease Advisory Guardrails
Verifies:
1. Explicit supported-crop and supported-class verification (Rice, Cotton, etc.)
2. Low-confidence warning (< 40.0% preliminary threshold)
3. Elimination of generic substring rust advice
4. Suppression of pesticide/treatment advice for unsupported crops or uncertain predictions
5. Accurate model probability reporting without false guarantees
6. Supported crop inference (>= 40.0%)
7. Flask API endpoints integration
8. Regression check on Crop Recommendation feature
"""

import io
import unittest
import numpy as np
from PIL import Image
from app import app
from ml.disease_advisor import (
    process_leaf_advisory,
    PRELIMINARY_CONFIDENCE_THRESHOLD,
    SUPPORTED_CROPS,
    UNSUPPORTED_CROPS,
    CLASS_SPECIFIC_ADVISORY
)
from ml.predictor import predict_crop


class TestDiseaseGuardrails(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create standard test images in memory
        # 1. Plain green leaf pattern (triggers corn/healthy in PlantVillage)
        cls.green_img_bytes = io.BytesIO()
        arr_green = np.zeros((224, 224, 3), dtype=np.uint8)
        arr_green[:, :, 1] = 160
        arr_green[:, :, 0] = 40
        arr_green[:, :, 2] = 20
        Image.fromarray(arr_green).save(cls.green_img_bytes, format='JPEG')

        # 2. Random noise image pattern (triggers low confidence < 40%)
        cls.noise_img_bytes = io.BytesIO()
        np.random.seed(42)
        arr_noise = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        Image.fromarray(arr_noise).save(cls.noise_img_bytes, format='JPEG')

        # Flask test client
        cls.client = app.test_client()

    def test_01_unsupported_crop_rice(self):
        """
        Guardrail 1: Rice is not in PlantVillage 14 crops.
        Must return status 'unsupported_crop', explicit safe message, and NO chemical treatments.
        """
        self.green_img_bytes.seek(0)
        result, error = process_leaf_advisory(self.green_img_bytes, crop_type='rice')

        self.assertIsNone(error)
        self.assertIsNotNone(result)
        self.assertEqual(result['status'], 'unsupported_crop')
        self.assertTrue(result['is_unsupported'])
        self.assertEqual(
            result['prediction'],
            'Unsupported crop — reliable disease prediction is unavailable for this image'
        )
        # Treatment must be suppressed
        self.assertIsNone(result['treatment_guidelines'], "Pesticide treatments must be None for unsupported crop!")
        # Confidence score must NOT be invented
        self.assertIsNone(result['confidence'])
        self.assertIn("unavailable", result['confidence_label'].lower())
        # Message must explicitly mention unsupported crop
        self.assertIn("PlantVillage", result['message'])
        self.assertIn("Rice", result['message'])
        print("\n[PASS] Test 1: Unsupported crop (Rice) correctly rejected without chemical advice.")

    def test_02_unsupported_crop_cotton_and_other(self):
        """
        Guardrail 1: Other unsupported crops (Cotton, Wheat, unknown) are also safely handled.
        """
        for crop in ['cotton', 'wheat', 'unknown_field_crop']:
            self.green_img_bytes.seek(0)
            result, error = process_leaf_advisory(self.green_img_bytes, crop_type=crop)
            self.assertIsNone(error)
            self.assertEqual(result['status'], 'unsupported_crop')
            self.assertIsNone(result['treatment_guidelines'])
        print("[PASS] Test 2: Other unsupported crops (Cotton, Wheat, Unknown) safely blocked.")

    def test_03_low_confidence_prediction(self):
        """
        Guardrail 2: Low-confidence prediction (< 40.0% heuristic threshold).
        Must return status 'low_confidence', flag uncertainty, and suppress pesticide advice.
        """
        self.noise_img_bytes.seek(0)
        result, error = process_leaf_advisory(self.noise_img_bytes, crop_type='auto')

        self.assertIsNone(error)
        self.assertIsNotNone(result)
        self.assertEqual(result['status'], 'low_confidence')
        self.assertTrue(result['is_low_confidence'])
        self.assertLess(result['confidence'], PRELIMINARY_CONFIDENCE_THRESHOLD)
        self.assertEqual(
            result['prediction'],
            'Uncertain prediction — reliable disease diagnosis is unavailable for this image'
        )
        # Treatment must be strictly None
        self.assertIsNone(result['treatment_guidelines'], "Pesticide treatments must be None for low-confidence prediction!")
        # Must show preliminary warning in confidence label
        self.assertIn(str(PRELIMINARY_CONFIDENCE_THRESHOLD), result['confidence_label'])
        print(f"[PASS] Test 3: Low-confidence prediction ({result['confidence']}%) triggered safe warning without treatments.")

    def test_04_supported_crop_high_confidence(self):
        """
        Supported Crop + High Confidence (>= 40.0%):
        Must return status 'prediction_success', actual calculated probability,
        and educational disclaimer that model confidence does not guarantee suitability/pathology.
        """
        self.green_img_bytes.seek(0)
        result, error = process_leaf_advisory(self.green_img_bytes, crop_type='auto')

        self.assertIsNone(error)
        self.assertIsNotNone(result)
        self.assertEqual(result['status'], 'prediction_success')
        self.assertFalse(result['is_unsupported'])
        self.assertFalse(result['is_low_confidence'])
        self.assertGreaterEqual(result['confidence'], PRELIMINARY_CONFIDENCE_THRESHOLD)
        self.assertIn('% Model confidence', result['confidence_label'])
        self.assertIsNotNone(result['confidence_disclaimer'])
        # Verify educational disclaimer mentions real-world factors
        self.assertIn('statistical similarity', result['confidence_disclaimer'].lower())
        print(f"[PASS] Test 4: Supported crop high-confidence prediction ({result['confidence']}%) verified with disclaimer.")

    def test_05_generic_rust_substring_eliminated(self):
        """
        Guardrail 6: Ensure generic substring matching for 'rust' is eliminated.
        Advice must come strictly from per-class dictionary CLASS_SPECIFIC_ADVISORY.
        """
        # Verify 38 classes are mapped in CLASS_SPECIFIC_ADVISORY
        self.assertEqual(len(CLASS_SPECIFIC_ADVISORY), 38, "All 38 PlantVillage classes must have explicit entries.")
        
        # Verify Cedar Apple Rust has Apple-specific advisory
        cedar_rust = CLASS_SPECIFIC_ADVISORY["Cedar Apple Rust"]
        self.assertEqual(cedar_rust['crop'], 'apple')
        self.assertIn('apple', cedar_rust['treatment'].lower())
        
        # Verify Corn Common Rust has Corn-specific advisory
        corn_rust = CLASS_SPECIFIC_ADVISORY["Corn (Maize) with Common Rust"]
        self.assertEqual(corn_rust['crop'], 'corn')
        self.assertIn('corn', corn_rust['summary'].lower())
        print("[PASS] Test 5: Generic 'rust' substring matching eliminated; strict 38-class mappings verified.")

    def test_06_crop_mismatch_guardrail(self):
        """
        Guardrail 3: Crop mismatch detection.
        When user specifies 'tomato' but leaf visually matches corn.
        """
        self.green_img_bytes.seek(0)
        # green leaf matches Corn (Maize) Plant
        result, error = process_leaf_advisory(self.green_img_bytes, crop_type='tomato')
        self.assertIsNone(error)
        self.assertEqual(result['status'], 'crop_mismatch')
        self.assertIsNone(result['treatment_guidelines'], "Pesticides must be None when crop mismatches!")
        print("[PASS] Test 6: Crop mismatch guardrail safely triggers when selected crop differs from visual match.")

    def test_07_api_analyze_disease_unsupported_crop(self):
        """
        Test POST /api/analyze-disease endpoint with crop_type='rice'.
        """
        self.green_img_bytes.seek(0)
        data = {
            'leaf_image': (self.green_img_bytes, 'rice_test_leaf.jpg'),
            'crop_type': 'rice'
        }
        response = self.client.post(
            '/api/analyze-disease',
            data=data,
            content_type='multipart/form-data'
        )
        self.assertEqual(response.status_code, 200)
        res_json = response.get_json()
        self.assertTrue(res_json['success'])
        self.assertEqual(res_json['data']['status'], 'unsupported_crop')
        self.assertIsNone(res_json['data']['treatment_guidelines'])
        print("[PASS] Test 7: API POST /api/analyze-disease successfully returned unsupported crop guardrail.")

    def test_08_api_validation_errors(self):
        """
        Test API image upload validation: empty request and invalid file format.
        """
        # Empty request
        resp1 = self.client.post('/api/analyze-disease', data={})
        self.assertEqual(resp1.status_code, 400)

        # Invalid file format (.txt)
        bad_file = io.BytesIO(b"not an image file")
        resp2 = self.client.post(
            '/api/analyze-disease',
            data={'leaf_image': (bad_file, 'test.txt'), 'crop_type': 'auto'},
            content_type='multipart/form-data'
        )
        self.assertEqual(resp2.status_code, 400)
        print("[PASS] Test 8: API input validation handles missing/invalid files properly.")

    def test_09_crop_recommendation_regression(self):
        """
        Regression Test: Feature 1 (Crop Recommendation) must remain fully functional
        with real model probabilities and no hardcoded 100% confidence scores.
        """
        sample_input = {
            'N': 90.0,
            'P': 42.0,
            'K': 43.0,
            'temperature': 20.8,
            'humidity': 82.0,
            'ph': 6.5,
            'rainfall': 202.9
        }
        res = predict_crop(sample_input)
        self.assertIn('crop_name', res)
        self.assertTrue(res['has_confidence'])
        self.assertLess(res['confidence'], 100.0, "Model confidence must be genuine, not hardcoded 100%!")
        print(f"[PASS] Test 9: Crop Recommendation regression passed ({res['crop_name']} at {res['confidence']}% confidence).")


if __name__ == '__main__':
    unittest.main()
