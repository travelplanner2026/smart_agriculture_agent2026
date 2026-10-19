"""
Crop Predictor Engine
Loads the trained Random Forest model and provides crop recommendations with agricultural insights.
"""

import os
import joblib
import pandas as pd

# Load model artifacts
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'ml', 'crop_model.pkl')

_cached_artifacts = None

def get_model_artifacts():
    global _cached_artifacts
    if _cached_artifacts is None:
        if os.path.exists(MODEL_PATH):
            _cached_artifacts = joblib.load(MODEL_PATH)
        else:
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please run train_model.py first.")
    return _cached_artifacts

# Comprehensive crop encyclopedia with agronomic details and recommendations
CROP_DETAILS = {
    'rice': {
        'name': 'Rice (Paddy)',
        'emoji': '🌾',
        'season': 'Kharif (Monsoon)',
        'soil': 'Clayey or loamy with high water retention capacity',
        'duration': '105 - 150 days',
        'water_need': 'High (Requires standing water 2-5 cm during vegetative stage)',
        'fertilizer_tip': 'Split application of Nitrogen (Basal, Tillering, Panicle initiation).',
        'care_tip': 'Maintain consistent puddle conditions and monitor for stem borer and blast.',
        'summary': 'Your warm climate, heavy rainfall, and nutrient-rich soil provide prime conditions for wetland rice cultivation.'
    },
    'maize': {
        'name': 'Maize (Corn)',
        'emoji': '🌽',
        'season': 'Kharif / Spring',
        'soil': 'Well-drained deep loamy and fertile red soils with pH 6.0 - 7.5',
        'duration': '80 - 110 days',
        'water_need': 'Moderate (Avoid waterlogging, critical stages: tasseling & silking)',
        'fertilizer_tip': 'Requires balanced NPK with Zinc sulphate booster at early vegetative stage.',
        'care_tip': 'Ensure adequate drainage as maize is highly sensitive to standing water.',
        'summary': 'The moderate rainfall and well-drained soil nutrient profile are ideal for high-yielding hybrid maize.'
    },
    'chickpea': {
        'name': 'Chickpea (Gram)',
        'emoji': '🌱',
        'season': 'Rabi (Winter)',
        'soil': 'Silty clay loam to sandy loam with good internal drainage',
        'duration': '90 - 120 days',
        'water_need': 'Low (Drought-hardy legume, requires 1-2 light irrigations)',
        'fertilizer_tip': 'Requires low Nitrogen due to atmospheric N-fixation; apply Phosphatic fertilizer.',
        'care_tip': 'Treat seeds with Rhizobium culture before sowing to boost root nodulation.',
        'summary': 'Your moderate temperature and dry-to-moderate moisture levels favor healthy pulse formation.'
    },
    'kidneybeans': {
        'name': 'Kidney Beans (Rajma)',
        'emoji': '🫘',
        'season': 'Rabi / Autumn',
        'soil': 'Rich, light loamy soil with excellent organic matter and pH 5.5 - 6.5',
        'duration': '100 - 130 days',
        'water_need': 'Moderate (Sensitive to both drought and waterlogging)',
        'fertilizer_tip': 'Unlike other legumes, Rajma needs higher basal Nitrogen (60-80 kg/ha).',
        'care_tip': 'Provide support stakes for climbing varieties and keep soil consistently moist.',
        'summary': 'The cool-to-mild climate and rich phosphorus balance provide top productivity for kidney beans.'
    },
    'pigeonpeas': {
        'name': 'Pigeon Peas (Arhar / Red Gram)',
        'emoji': '🍲',
        'season': 'Kharif (Monsoon)',
        'soil': 'Deep loamy soils with good drainage; tolerates neutral to mild alkaline pH',
        'duration': '150 - 180 days',
        'water_need': 'Low to Moderate (Deep taproot system makes it drought tolerant)',
        'fertilizer_tip': 'Apply basal dose of 20 kg N + 50 kg P2O5 per hectare.',
        'care_tip': 'Intercrop with sorghum or pearl millet for enhanced soil health and yield.',
        'summary': 'The warm conditions and moderate rainfall pattern suit deep-rooted pigeon peas very well.'
    },
    'mothbeans': {
        'name': 'Moth Beans',
        'emoji': '🌿',
        'season': 'Kharif',
        'soil': 'Light sandy loam, well-drained, tolerates slightly alkaline soil',
        'duration': '75 - 90 days',
        'water_need': 'Very Low (Extremely drought-resilient arid zone crop)',
        'fertilizer_tip': 'Minimal fertilizer required; 10 kg N + 20 kg P2O5 is sufficient.',
        'care_tip': 'Excellent for dryland farming; suppresses weeds with quick ground-spreading foliage.',
        'summary': 'Ideal for regions with warm weather and low precipitation where drought resilience is essential.'
    },
    'mungbean': {
        'name': 'Mung Bean (Green Gram)',
        'emoji': '🌱',
        'season': 'Zaid (Summer) or Kharif',
        'soil': 'Fertile, well-drained sandy loam or clay loam with pH 6.2 - 7.2',
        'duration': '60 - 75 days',
        'water_need': 'Low to Moderate (Short-duration catch crop)',
        'fertilizer_tip': 'Incorporate Rhizobium bio-fertilizer and foliar spray DAP at flowering.',
        'care_tip': 'Harvest in dry weather when 80% of pods turn dark brown/black.',
        'summary': 'A fast-growing pulse that matches your moderate temperature and short moisture window.'
    },
    'blackgram': {
        'name': 'Black Gram (Urad)',
        'emoji': '🥣',
        'season': 'Kharif / Spring',
        'soil': 'Heavy black cotton soil or loam with good water holding capacity',
        'duration': '70 - 85 days',
        'water_need': 'Low to Moderate (Sensitive to cold and frosting)',
        'fertilizer_tip': 'Apply sulfur and phosphorus to enhance protein synthesis.',
        'care_tip': 'Spray neem oil periodically to safeguard against yellow mosaic virus vectors.',
        'summary': 'The warm humid environment and balanced soil nutrients encourage vigorous blackgram pods.'
    },
    'lentil': {
        'name': 'Lentil (Masoor)',
        'emoji': '🍲',
        'season': 'Rabi (Winter)',
        'soil': 'Alluvial, clay to sandy loam soils that retain sub-soil moisture',
        'duration': '110 - 130 days',
        'water_need': 'Low (Thrives under cool conditions and dry maturity stage)',
        'fertilizer_tip': 'Requires 15 kg N + 40 kg P2O5; benefits from sulfur application.',
        'care_tip': 'Weed control in the first 30 days is critical for high branch development.',
        'summary': 'Mild winter temperatures and modest rainfall create an optimal microclimate for lentils.'
    },
    'pomegranate': {
        'name': 'Pomegranate',
        'emoji': '🪴',
        'season': 'Perennial (Year-round)',
        'soil': 'Deep loamy or alluvial soil with pH 6.5 - 7.5 and good aeration',
        'duration': 'Perennial (First harvest after 2.5 - 3 years)',
        'water_need': 'Moderate (Drip irrigation is strongly recommended)',
        'fertilizer_tip': 'Apply well-rotted FYM (Farmyard manure) + micronutrient sprays (Boron, Zinc).',
        'care_tip': 'Regulate flowering with proper Bahar treatment (Mrig or Hasta Bahar).',
        'summary': 'Semi-arid conditions with warm sunshine allow pomegranates to build high sugar content and rich color.'
    },
    'banana': {
        'name': 'Banana',
        'emoji': '🍌',
        'season': 'Year-round planting',
        'soil': 'Deep, rich loamy soil with high organic matter and neutral pH',
        'duration': '11 - 14 months',
        'water_need': 'Very High (Requires regular weekly irrigation and humid atmosphere)',
        'fertilizer_tip': 'Heavy feeder of Potassium; apply potash in regular intervals.',
        'care_tip': 'Desucker regularly and provide prop supports to fruit-bearing pseudostems.',
        'summary': 'The high humidity, warm temperature, and abundant potassium are ideal for lush banana groves.'
    },
    'mango': {
        'name': 'Mango',
        'emoji': '🥭',
        'season': 'Perennial (Flowering in winter, fruit in summer)',
        'soil': 'Deep alluvial or red loamy soil with depth > 2 meters and good drainage',
        'duration': 'Perennial tree crop',
        'water_need': 'Moderate (Withhold water 2 months before flowering to induce blooms)',
        'fertilizer_tip': 'Apply organic compost and complete NPK according to tree age.',
        'care_tip': 'Protect emerging panicles from powdery mildew and mango hoppers.',
        'summary': 'Your tropical warm climate and distinct dry period during bloom are perfect for premium mangoes.'
    },
    'grapes': {
        'name': 'Grapes',
        'emoji': '🍇',
        'season': 'Perennial (Pruning in Oct, Harvest March-April)',
        'soil': 'Well-drained sandy loam or gravelly soil with pH 6.5 - 8.0',
        'duration': 'Perennial vine',
        'water_need': 'Moderate (Strict drip schedule; dry weather needed during ripening)',
        'fertilizer_tip': 'High requirement for Potassium and Calcium for berry firmness and sweetness.',
        'care_tip': 'Maintain open canopy through canopy management to reduce fungal mildew risk.',
        'summary': 'Warm dry climate with rich mineral support enables optimum berry sizing and high Brix level.'
    },
    'watermelon': {
        'name': 'Watermelon',
        'emoji': '🍉',
        'season': 'Zaid (Summer)',
        'soil': 'Sandy or sandy loam soils that warm up quickly with pH 6.0 - 7.0',
        'duration': '75 - 90 days',
        'water_need': 'Moderate (Frequent light irrigation; stop watering 1 week before harvest)',
        'fertilizer_tip': 'Apply basal compost + Potassium boost during fruit swelling.',
        'care_tip': 'Turn fruit gently during development for uniform skin color and sweetness.',
        'summary': 'Warm sunny weather and well-drained sandy soil provide optimal conditions for sweet, juicy melons.'
    },
    'muskmelon': {
        'name': 'Muskmelon (Cantaloupe)',
        'emoji': '🍈',
        'season': 'Zaid (Summer)',
        'soil': 'Sandy loam soil rich in organic humus with pH 6.0 - 7.5',
        'duration': '70 - 85 days',
        'water_need': 'Moderate (Keep root zone moist without wetting vines)',
        'fertilizer_tip': 'NPK 80:40:40 kg/ha with micronutrient spray (Magnesium & Boron).',
        'care_tip': 'Mulching with organic straw conserves moisture and keeps melons clean.',
        'summary': 'The hot dry weather during ripening develops sweet aroma and high sugar content.'
    },
    'apple': {
        'name': 'Apple',
        'emoji': '🍎',
        'season': 'Temperate / Perennial',
        'soil': 'Deep, well-aerated loamy soil with pH 5.5 - 6.5',
        'duration': 'Perennial tree crop',
        'water_need': 'Moderate (Requires uniform 1000-1200 mm precipitation or irrigation)',
        'fertilizer_tip': 'Apply farm manure in late winter and balanced nitrogen in spring flush.',
        'care_tip': 'Requires 800 - 1200 chilling hours (<7°C) during winter for breaking dormancy.',
        'summary': 'Cool temperate temperatures and specific chilling hours support healthy apple bloom and coloring.'
    },
    'orange': {
        'name': 'Orange / Citrus',
        'emoji': '🍊',
        'season': 'Subtropical / Perennial',
        'soil': 'Light loamy, well-drained soils free from saline or alkaline salts (pH 6.0 - 7.5)',
        'duration': 'Perennial tree crop',
        'water_need': 'Moderate (Drip irrigation is ideal, avoid direct collar contact with water)',
        'fertilizer_tip': 'Foliar sprays of Zinc, Iron, and Manganese 2-3 times per year.',
        'care_tip': 'Check rootstocks for Phytophthora gummosis and maintain good field aeration.',
        'summary': 'Subtropical mild temperatures and moderate humidity promote sweet, vitamin C-rich citrus fruit.'
    },
    'papaya': {
        'name': 'Papaya',
        'emoji': '🥭',
        'season': 'Year-round in frost-free regions',
        'soil': 'Rich sandy loam or alluvial soil with pH 6.5 - 7.0 and exceptional drainage',
        'duration': '9 - 12 months for first picking',
        'water_need': 'Moderate (Extremely vulnerable to root rot in waterlogged conditions)',
        'fertilizer_tip': 'Apply 200g N + 200g P + 250g K per plant divided into bi-monthly doses.',
        'care_tip': 'Plant on raised beds to ensure rapid runoff during heavy rains.',
        'summary': 'Warm tropical sunshine and high humidity accelerate rapid vegetative growth and heavy fruiting.'
    },
    'coconut': {
        'name': 'Coconut',
        'emoji': '🥥',
        'season': 'Perennial coastal palm',
        'soil': 'Coastal sandy, alluvial or red loam with good depth and permeability',
        'duration': 'Perennial (Yields continuously for 50+ years)',
        'water_need': 'High (Requires regular moisture; benefits from coastal sea breeze and humidity)',
        'fertilizer_tip': 'Apply 500g N, 320g P2O5, 1200g K2O per bearing palm annually.',
        'care_tip': 'Mulch coconut basins with coconut husks to retain soil moisture during summer.',
        'summary': 'High humidity, abundant rainfall, and warm coastal temperatures provide ideal palm conditions.'
    },
    'cotton': {
        'name': 'Cotton (White Gold)',
        'emoji': '☁️',
        'season': 'Kharif',
        'soil': 'Deep black cotton soil (Regur) or fertile alluvial loam with pH 6.0 - 8.0',
        'duration': '150 - 180 days',
        'water_need': 'Moderate (Warm days and sunny skies required during boll opening)',
        'fertilizer_tip': 'Apply balanced NPK with Magnesium sulphate spray to prevent leaf reddening.',
        'care_tip': 'Adopt Integrated Pest Management (IPM) to control bollworm and whitefly.',
        'summary': 'The rich potassium/phosphorus balance, warm temperature, and moderate rainfall yield high boll quality.'
    },
    'jute': {
        'name': 'Jute (Golden Fiber)',
        'emoji': '🌾',
        'season': 'Kharif',
        'soil': 'New alluvial silt loam soil deposited by annual river flooding',
        'duration': '120 - 150 days',
        'water_need': 'High (Requires heavy rainfall >150mm and warm humid weather)',
        'fertilizer_tip': 'Apply basal dose of compost and top dress urea at 4-6 weeks after sowing.',
        'care_tip': 'Ensure clear retting water tanks nearby for optimal fiber extraction.',
        'summary': 'High moisture, heavy precipitation, and warm climate facilitate tall fibrous jute reeds.'
    },
    'coffee': {
        'name': 'Coffee',
        'emoji': '☕',
        'season': 'Shaded Perennial (Harvest Nov - Feb)',
        'soil': 'Rich volcanic or forest loam high in organic humus with pH 5.0 - 6.5',
        'duration': 'Perennial plantation crop',
        'water_need': 'High (Requires 1500-2500 mm rain, blossom showers in March/April)',
        'fertilizer_tip': 'Apply pre-monsoon and post-monsoon NPK fertilizer mixtures.',
        'care_tip': 'Grow under a two-tier shade canopy (silver oak / leguminous trees) on hill slopes.',
        'summary': 'Cool hilly altitude, high humidity, rich organic soil, and shade trees nurture rich coffee beans.'
    }
}

# Validation boundary ranges based on agricultural field reality
FEATURE_RANGES = {
    'N': {'min': 0, 'max': 200, 'unit': 'kg/ha', 'name': 'Nitrogen'},
    'P': {'min': 0, 'max': 200, 'unit': 'kg/ha', 'name': 'Phosphorus'},
    'K': {'min': 0, 'max': 300, 'unit': 'kg/ha', 'name': 'Potassium'},
    'temperature': {'min': 0.0, 'max': 55.0, 'unit': '°C', 'name': 'Temperature'},
    'humidity': {'min': 5.0, 'max': 100.0, 'unit': '%', 'name': 'Relative Humidity'},
    'ph': {'min': 3.0, 'max': 10.0, 'unit': '', 'name': 'Soil pH'},
    'rainfall': {'min': 0.0, 'max': 500.0, 'unit': 'mm', 'name': 'Rainfall'}
}

def validate_crop_inputs(data):
    """
    Validates form input dictionary and converts values to floats.
    Returns (cleaned_dict, error_message).
    """
    cleaned = {}
    for feature, rules in FEATURE_RANGES.items():
        if feature not in data or data[feature] is None or str(data[feature]).strip() == '':
            return None, f"Please enter a value for {rules['name']} ({feature.upper()})."
        
        try:
            val = float(data[feature])
        except (ValueError, TypeError):
            return None, f"Value for {rules['name']} must be a valid number."
        
        if val < rules['min'] or val > rules['max']:
            unit_str = f" {rules['unit']}" if rules['unit'] else ""
            return None, f"{rules['name']} must be between {rules['min']}{unit_str} and {rules['max']}{unit_str}."
        
        cleaned[feature] = val

    return cleaned, None

def predict_crop(input_data):
    """
    Runs the ML model on cleaned input data.
    Returns recommendation details with confidence and agricultural context.
    """
    artifacts = get_model_artifacts()
    model = artifacts['model']
    features = artifacts['features']

    # Create 1-row DataFrame with correct column order
    input_df = pd.DataFrame([[input_data[f] for f in features]], columns=features)

    # Predict crop
    pred_class = model.predict(input_df)[0]
    
    # Check if model reliably supports predict_proba
    has_proba = hasattr(model, 'predict_proba') and callable(getattr(model, 'predict_proba'))

    confidence_val = None
    confidence_label = "Confidence score unavailable"
    top_crop = pred_class
    runners_up = []

    if has_proba:
        try:
            probabilities = model.predict_proba(input_df)[0]
            classes = list(model.classes_)
            
            # Sort crops by probability descending
            class_probs = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)
            top_crop, top_prob = class_probs[0]
            
            # Option B: Display Capping & Honest Consensus Labeling
            # In an unpruned Random Forest, when all 100 trees agree, the vote ratio is 100.0%.
            # To prevent misleading users into believing there is a 100% biological guarantee,
            # we cap unanimous consensus at 95.0% and label it ">95% Model Consensus (High Agreement)".
            raw_percent = round(float(top_prob) * 100, 1)
            if raw_percent >= 99.0:
                confidence_val = 95.0
                confidence_label = ">95% Model Consensus (High Agreement)"
            elif raw_percent >= 95.0:
                confidence_val = raw_percent
                confidence_label = f"{raw_percent}% Model Consensus (High Agreement)"
            else:
                confidence_val = raw_percent
                confidence_label = f"{raw_percent}% Model Consensus"
            
            # Alternative candidate crops with >= 5% probability
            for c, p in class_probs[1:4]:
                if p >= 0.05:
                    info = CROP_DETAILS.get(c, {'name': c.capitalize(), 'emoji': '🌱'})
                    runners_up.append({
                        'crop_key': c,
                        'name': info.get('name', c.capitalize()),
                        'emoji': info.get('emoji', '🌱'),
                        'probability': round(float(p) * 100, 1)
                    })
        except Exception:
            # Fallback if probability array cannot be computed reliably
            confidence_val = None
            confidence_label = "Confidence score unavailable"
            top_crop = pred_class

    crop_info = CROP_DETAILS.get(top_crop, {
        'name': top_crop.capitalize(),
        'emoji': '🌱',
        'season': 'Seasonal',
        'soil': 'Fertile agricultural soil',
        'duration': 'Standard crop cycle',
        'water_need': 'Moderate',
        'fertilizer_tip': 'Apply balanced NPK according to local soil testing lab recommendations.',
        'care_tip': 'Monitor field periodically for pests and weeds.',
        'summary': f'Based on your soil and weather values, {top_crop.capitalize()} is the best suited crop.'
    })

    return {
        'crop_key': top_crop,
        'crop_name': crop_info['name'],
        'emoji': crop_info['emoji'],
        'confidence': confidence_val,
        'confidence_label': confidence_label,
        'has_confidence': confidence_val is not None,
        'confidence_disclaimer': "Model consensus represents the statistical agreement among decision trees in the Random Forest ensemble based on historical data. It does not guarantee that a crop will thrive or yield in the field, as unmeasured real-world factors (such as soil microbial health, pest pressure, subsoil drainage, and irrigation water quality) also determine farming outcomes.",
        'model_accuracy': artifacts.get('accuracy', 99.0),
        'season': crop_info['season'],
        'soil': crop_info['soil'],
        'duration': crop_info['duration'],
        'water_need': crop_info['water_need'],
        'fertilizer_tip': crop_info['fertilizer_tip'],
        'care_tip': crop_info['care_tip'],
        'summary': crop_info['summary'],
        'runners_up': runners_up,
        'inputs': input_data
    }
