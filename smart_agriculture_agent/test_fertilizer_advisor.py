"""
Automated Test Suite for Feature 3: Soil Health & Fertilizer Advisor
Validates:
1. Soil input validation (units, ranges, error messages)
2. ICAR soil nutrient rating benchmarks (Low, Medium/Adequate, High)
3. Cautious qualitative guidance (no fabricated doses, split applications)
4. Legume biological nitrogen fixation logic vs Cereal heavy feeder logic
5. Acidic, neutral, and alkaline soil conditioning recommendations
6. Transparency checks (no false AI claims, explicit disclaimer on KVK soil tests)
7. Flask web endpoints (/fertilizer-advisor, /api/analyze-soil, /health)
8. Regression checks on Feature 1 (Crop Recommendation) and Feature 2 (Disease Advisory)
"""

import unittest
from app import app
from ml.fertilizer_advisor import (
    validate_soil_inputs,
    assess_soil_nutrients,
    generate_fertilizer_guidance,
    process_soil_advisory,
    SOIL_INPUT_RANGES,
    NUTRIENT_BENCHMARKS,
    SUPPORTED_CROPS_INFO
)
from ml.predictor import predict_crop
from ml.disease_advisor import process_leaf_advisory
import io
from PIL import Image


class TestFertilizerAdvisor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_01_input_validation_boundaries(self):
        """
        Validate input boundaries and realistic agricultural ranges.
        """
        # Test valid inputs
        valid_data = {'ph': '6.8', 'n': '120', 'p': '35', 'k': '180', 'crop': 'rice'}
        cleaned, err = validate_soil_inputs(valid_data)
        self.assertIsNone(err)
        self.assertEqual(cleaned['ph'], 6.8)
        self.assertEqual(cleaned['n'], 120.0)
        self.assertEqual(cleaned['p'], 35.0)
        self.assertEqual(cleaned['k'], 180.0)
        self.assertEqual(cleaned['crop'], 'rice')

        # Test out-of-range pH (< 3.5)
        bad_ph_low, err = validate_soil_inputs({'ph': '2.1', 'n': '120', 'p': '35', 'k': '180'})
        self.assertIsNone(bad_ph_low)
        self.assertIn('between 3.5', err)

        # Test out-of-range pH (> 9.5)
        bad_ph_high, err = validate_soil_inputs({'ph': '11.0', 'n': '120', 'p': '35', 'k': '180'})
        self.assertIsNone(bad_ph_high)
        self.assertIn('between 3.5', err)

        # Test negative N (< 0)
        bad_n, err = validate_soil_inputs({'ph': '6.5', 'n': '-10', 'p': '35', 'k': '180'})
        self.assertIsNone(bad_n)
        self.assertIn('Nitrogen', err)

        # Test excessive P (> 150)
        bad_p, err = validate_soil_inputs({'ph': '6.5', 'n': '120', 'p': '250', 'k': '180'})
        self.assertIsNone(bad_p)
        self.assertIn('Phosphorus', err)

        # Test non-numeric input
        non_num, err = validate_soil_inputs({'ph': 'neutral', 'n': '120', 'p': '35', 'k': '180'})
        self.assertIsNone(non_num)
        self.assertIn('valid number', err)

        # Test missing field
        missing, err = validate_soil_inputs({'ph': '6.5', 'n': '120', 'p': '35'})
        self.assertIsNone(missing)
        self.assertIn('Potassium', err)
        print("\n[PASS] Test 1: Soil input validation boundaries and error messages verified.")

    def test_02_icar_nutrient_benchmarks(self):
        """
        Verify that soil nutrients are accurately categorized based on ICAR Soil Health Card standards.
        """
        # Low fertility soil: N=90 (<140), P=15 (<23), K=100 (<140), pH=5.0 (<5.5)
        asm_low = assess_soil_nutrients(n=90.0, p=15.0, k=100.0, ph=5.0)
        self.assertEqual(asm_low['nitrogen']['status'], 'Low')
        self.assertEqual(asm_low['phosphorus']['status'], 'Low')
        self.assertEqual(asm_low['potassium']['status'], 'Low')
        self.assertEqual(asm_low['ph']['status'], 'Strongly Acidic')

        # Adequate fertility soil: N=200 (140-280), P=35 (23-56), K=220 (140-330), pH=7.0 (6.5-7.5)
        asm_med = assess_soil_nutrients(n=200.0, p=35.0, k=220.0, ph=7.0)
        self.assertEqual(asm_med['nitrogen']['status'], 'Medium / Adequate')
        self.assertEqual(asm_med['phosphorus']['status'], 'Medium / Adequate')
        self.assertEqual(asm_med['potassium']['status'], 'Medium / Adequate')
        self.assertEqual(asm_med['ph']['status'], 'Neutral / Optimum')

        # High/Rich fertility soil: N=320 (>280), P=70 (>56), K=380 (>330), pH=8.8 (>8.5)
        asm_high = assess_soil_nutrients(n=320.0, p=70.0, k=380.0, ph=8.8)
        self.assertIn('High', asm_high['nitrogen']['status'])
        self.assertIn('High', asm_high['phosphorus']['status'])
        self.assertIn('High', asm_high['potassium']['status'])
        self.assertEqual(asm_high['ph']['status'], 'Strongly Alkaline / Calcareous')
        print("[PASS] Test 2: ICAR Soil Health Card nutrient classification benchmarks verified.")

    def test_03_cautious_guidance_and_no_fabricated_doses(self):
        """
        Requirements 5 & 6: Guidance must be cautious, qualitative, and emphasize
        soil test lab reports without inventing unverified exact doses.
        """
        result, err = process_soil_advisory({
            'ph': 6.5,
            'n': 100.0,
            'p': 20.0,
            'k': 120.0,
            'crop': 'rice'
        })
        self.assertIsNone(err)
        self.assertIsNotNone(result)

        # Check that prominent disclaimer is present
        self.assertIn('Krishi Vigyan Kendra', result['disclaimer'])
        self.assertIn('soil testing laboratory', result['disclaimer'])

        # Check split application guidance for low N in Rice
        n_guide = result['guidance']['nitrogen_guideline']
        self.assertIn('split', n_guide.lower())
        self.assertIn('leaching', n_guide.lower())

        # Check basal placement for low P
        p_guide = result['guidance']['phosphorus_guideline']
        self.assertIn('basal', p_guide.lower())

        # Check 4R stewardship principles
        stewardship = result['guidance']['stewardship_points']
        self.assertEqual(len(stewardship), 4)
        titles = [s['title'] for s in stewardship]
        self.assertIn('Right Source', titles)
        self.assertIn('Right Rate', titles)
        self.assertIn('Right Time', titles)
        self.assertIn('Right Place', titles)
        print("[PASS] Test 3: Cautious guidance, split-application advice, and 4R stewardship verified.")

    def test_04_legume_vs_cereal_specialization(self):
        """
        Verify that legumes (Soybean, Pulses) receive biological N-fixation advice
        while cereals (Rice, Wheat, Corn) receive split-synthetic N advice.
        """
        # Cereal (Corn) with low N
        corn_res, _ = process_soil_advisory({'ph': 6.5, 'n': 90.0, 'p': 30.0, 'k': 200.0, 'crop': 'corn'})
        self.assertIn('split', corn_res['guidance']['nitrogen_guideline'].lower())
        self.assertIn('heavy feeder', corn_res['guidance']['crop_summary'].lower())

        # Legume (Soybean) with low N
        soy_res, _ = process_soil_advisory({'ph': 6.5, 'n': 90.0, 'p': 30.0, 'k': 200.0, 'crop': 'soybean'})
        soy_n_guide = soy_res['guidance']['nitrogen_guideline'].lower()
        self.assertIn('nodule', soy_n_guide)
        self.assertIn('rhizobium', soy_n_guide)
        self.assertIn('starter dose', soy_n_guide)
        print("[PASS] Test 4: Crop-specific nutrition context (Cereal vs Legume N-fixation) verified.")

    def test_05_soil_ph_conditioning_advice(self):
        """
        Verify that acidic soil recommends agricultural lime and alkaline soil recommends gypsum/organic matter.
        """
        # Acidic soil (pH 4.8)
        acidic_res, _ = process_soil_advisory({'ph': 4.8, 'n': 180.0, 'p': 30.0, 'k': 200.0, 'crop': 'wheat'})
        ph_acid_guide = acidic_res['guidance']['ph_guideline'].lower()
        self.assertIn('lime', ph_acid_guide)

        # Alkaline soil (pH 8.8)
        alkaline_res, _ = process_soil_advisory({'ph': 8.8, 'n': 180.0, 'p': 30.0, 'k': 200.0, 'crop': 'wheat'})
        ph_alk_guide = alkaline_res['guidance']['ph_guideline'].lower()
        self.assertTrue('gypsum' in ph_alk_guide or 'organic' in ph_alk_guide)
        print("[PASS] Test 5: Soil pH amendment and conditioning advice (lime/gypsum) verified.")

    def test_06_model_transparency_and_no_ai_claims(self):
        """
        Requirement 7: Do NOT claim to use AI or a trained model unless one is integrated.
        Must explicitly declare rule-based agronomic engine.
        """
        result, _ = process_soil_advisory({'ph': 6.5, 'n': 180.0, 'p': 30.0, 'k': 200.0, 'crop': 'general'})
        self.assertFalse(result['has_ai_model'])
        self.assertIn('rule-based', result['engine_type'].lower())
        self.assertIn('no black-box', result['ai_transparency_note'].lower())
        print("[PASS] Test 6: Model transparency and absence of false AI claims verified.")

    def test_07_flask_endpoints(self):
        """
        Verify Flask web routes: GET /fertilizer-advisor, POST /api/analyze-soil, GET /health.
        """
        # GET /fertilizer-advisor
        resp_get = self.client.get('/fertilizer-advisor')
        self.assertEqual(resp_get.status_code, 200)
        self.assertIn(b'Soil Health & Fertilizer Advisor', resp_get.data)

        # POST /api/analyze-soil (valid)
        payload = {'ph': 6.8, 'n': 130.0, 'p': 25.0, 'k': 160.0, 'crop': 'rice'}
        resp_post = self.client.post('/api/analyze-soil', json=payload)
        self.assertEqual(resp_post.status_code, 200)
        data = resp_post.get_json()
        self.assertTrue(data['success'])
        self.assertIn('assessment', data['data'])
        self.assertIn('guidance', data['data'])

        # POST /api/analyze-soil (invalid)
        bad_resp = self.client.post('/api/analyze-soil', json={'ph': 15.0, 'n': 100})
        self.assertEqual(bad_resp.status_code, 400)

        # GET /health
        health_resp = self.client.get('/health')
        self.assertEqual(health_resp.status_code, 200)
        health_data = health_resp.get_json()
        self.assertEqual(health_data['modules']['soil_fertilizer_advisor'], 'ready')
        print("[PASS] Test 7: Flask web endpoints (/fertilizer-advisor, /api/analyze-soil, /health) verified.")

    def test_08_regression_features_1_and_2(self):
        """
        Requirements 8 & 9: Verify that Feature 1 (Crop Recommendation) and Feature 2 (Crop Disease Advisory)
        continue to function perfectly alongside Feature 3.
        """
        # Feature 1 Regression
        crop_input = {
            'N': 90.0,
            'P': 42.0,
            'K': 43.0,
            'temperature': 20.8,
            'humidity': 82.0,
            'ph': 6.5,
            'rainfall': 202.9
        }
        rec = predict_crop(crop_input)
        self.assertIn('crop_name', rec)
        self.assertTrue(rec['has_confidence'])

        # Feature 2 Regression: Unsupported Rice leaf must trigger Guardrail 1
        dummy_img = io.BytesIO()
        Image.new('RGB', (224, 224), color=(34, 139, 34)).save(dummy_img, format='JPEG')
        dummy_img.seek(0)
        dis_res, err = process_leaf_advisory(dummy_img, crop_type='rice')
        self.assertIsNone(err)
        self.assertEqual(dis_res['status'], 'unsupported_crop')
        self.assertIsNone(dis_res['treatment_guidelines'])
        print("[PASS] Test 8: Regression check on Feature 1 and Feature 2 passed seamlessly.")


if __name__ == '__main__':
    unittest.main()
