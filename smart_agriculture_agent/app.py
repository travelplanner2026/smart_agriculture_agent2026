"""
Smart Agriculture AI Agent - Web Application Backend
Developed with Flask for College Project Demonstration
"""

from flask import Flask, render_template, request, jsonify
from ml.predictor import predict_crop, validate_crop_inputs, FEATURE_RANGES
from ml.disease_advisor import (
    process_leaf_advisory,
    check_model_availability,
    VERIFIED_DISEASE_LIBRARY,
    SUPPORTED_CROPS,
    UNSUPPORTED_CROPS,
    PRELIMINARY_CONFIDENCE_THRESHOLD
)
from ml.fertilizer_advisor import (
    process_soil_advisory,
    SUPPORTED_CROPS_INFO,
    SOIL_INPUT_RANGES
)

# Initialize Flask application
app = Flask(__name__)

# Configure maximum upload size: 10 Megabytes limit for Flask request body
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# -----------------------------------------------------------------------------
# 1. Homepage Route
# -----------------------------------------------------------------------------
@app.route('/')
def home():
    """
    Renders the main green-and-white homepage for the Smart Agriculture AI Agent.
    """
    return render_template('index.html')

# -----------------------------------------------------------------------------
# 2. Crop Recommendation Routes (Module 1)
# -----------------------------------------------------------------------------
@app.route('/crop-recommendation', methods=['GET', 'POST'])
def crop_recommendation():
    result = None
    error = None
    form_data = {}

    if request.method == 'POST':
        form_data = {
            'N': request.form.get('N', '').strip(),
            'P': request.form.get('P', '').strip(),
            'K': request.form.get('K', '').strip(),
            'temperature': request.form.get('temperature', '').strip(),
            'humidity': request.form.get('humidity', '').strip(),
            'ph': request.form.get('ph', '').strip(),
            'rainfall': request.form.get('rainfall', '').strip(),
        }
        cleaned, error = validate_crop_inputs(form_data)
        if cleaned:
            try:
                result = predict_crop(cleaned)
            except Exception as e:
                error = f"Prediction error: {str(e)}"

    return render_template('crop_recommendation.html', result=result, error=error, form_data=form_data)

@app.route('/api/recommend-crop', methods=['POST'])
def api_recommend_crop():
    payload = request.get_json(silent=True)
    if not payload:
        payload = request.form.to_dict()

    cleaned, error = validate_crop_inputs(payload)
    if error:
        return jsonify({'success': False, 'error': error}), 400

    try:
        prediction = predict_crop(cleaned)
        return jsonify({'success': True, 'data': prediction})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# -----------------------------------------------------------------------------
# 3. Crop Disease Advisory Routes (Module 2)
# -----------------------------------------------------------------------------
@app.route('/disease-advisory', methods=['GET', 'POST'])
def disease_advisory():
    result = None
    error = None
    has_model, model_msg = check_model_availability()

    if request.method == 'POST':
        file = request.files.get('leaf_image')
        crop_type = request.form.get('crop_type', 'auto')
        result, error = process_leaf_advisory(file, crop_type=crop_type)

    return render_template(
        'disease_advisory.html',
        result=result,
        error=error,
        has_model=has_model,
        model_msg=model_msg,
        reference_diseases=VERIFIED_DISEASE_LIBRARY,
        supported_crops=SUPPORTED_CROPS,
        unsupported_crops=UNSUPPORTED_CROPS,
        preliminary_threshold=PRELIMINARY_CONFIDENCE_THRESHOLD
    )

@app.route('/api/analyze-disease', methods=['POST'])
def api_analyze_disease():
    if 'leaf_image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file uploaded. Please select a leaf photo.'}), 400

    file = request.files['leaf_image']
    crop_type = request.form.get('crop_type', 'auto')
    result, error = process_leaf_advisory(file, crop_type=crop_type)
    if error:
        return jsonify({'success': False, 'error': error}), 400

    return jsonify({'success': True, 'data': result})

# -----------------------------------------------------------------------------
# 4. Soil Health & Fertilizer Advisor Routes (Module 3)
# -----------------------------------------------------------------------------
@app.route('/fertilizer-advisor', methods=['GET', 'POST'])
def fertilizer_advisor():
    result = None
    error = None

    if request.method == 'POST':
        form_dict = {
            'ph': request.form.get('ph'),
            'n': request.form.get('n'),
            'p': request.form.get('p'),
            'k': request.form.get('k'),
            'crop': request.form.get('crop', 'general')
        }
        result, error = process_soil_advisory(form_dict)

    return render_template(
        'fertilizer_advisor.html',
        result=result,
        error=error,
        supported_crops=SUPPORTED_CROPS_INFO,
        soil_ranges=SOIL_INPUT_RANGES
    )

@app.route('/api/analyze-soil', methods=['POST'])
def api_analyze_soil():
    payload = request.get_json(silent=True) or request.form.to_dict()
    if not payload:
        return jsonify({'success': False, 'error': 'No input data provided. Please enter soil test values.'}), 400

    result, error = process_soil_advisory(payload)
    if error:
        return jsonify({'success': False, 'error': error}), 400

    return jsonify({'success': True, 'data': result})

# -----------------------------------------------------------------------------
# 5. Health Check Route
# -----------------------------------------------------------------------------
@app.route('/health')
def health():
    has_disease_model, _ = check_model_availability()
    return {
        "status": "running",
        "project": "Smart Agriculture AI Agent",
        "modules": {
            "crop_recommendation": "ready",
            "crop_disease_advisory": "model_ready" if has_disease_model else "manual_knowledgebase_ready",
            "soil_fertilizer_advisor": "ready",
            "qa_assistant": "upcoming"
        }
    }

if __name__ == '__main__':
    has_disease_model, _ = check_model_availability()
    print("\n" + "="*60)
    print(">> Smart Agriculture AI Agent is starting...")
    print(">> Website is available at: http://127.0.0.1:5000")
    print(">> Crop Recommendation: http://127.0.0.1:5000/crop-recommendation")
    print(">> Crop Disease Advisory: http://127.0.0.1:5000/disease-advisory")
    print(">> Soil & Fertilizer Advisor: http://127.0.0.1:5000/fertilizer-advisor")
    if not has_disease_model:
        print(">> Note: Computer Vision model not loaded; Knowledgebase mode active.")
    print("="*60 + "\n")
    app.run(debug=True, host='127.0.0.1', port=5000)
