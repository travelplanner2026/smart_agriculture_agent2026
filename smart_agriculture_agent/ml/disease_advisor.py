"""
Crop Disease Advisory Engine
Integrates genuine PlantVillage MobileNetV2 ONNX model for leaf image classification.
Includes explicit supported-crop verification and low-confidence prediction guardrails.
"""

import os
import io
import json
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'ml')
MODEL_PATH = os.path.join(MODELS_DIR, 'disease_model.onnx')
CONFIG_PATH = os.path.join(MODELS_DIR, 'disease_config.json')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 Megabytes

# Configurable preliminary warning threshold for student demonstration.
# Note: This is an empirical heuristic threshold for project safety, not a medically or scientifically validated cutoff.
PRELIMINARY_CONFIDENCE_THRESHOLD = 40.0

# 14 explicitly supported crops in the PlantVillage model
SUPPORTED_CROPS = {
    'apple': {'name': 'Apple', 'emoji': '🍎'},
    'blueberry': {'name': 'Blueberry', 'emoji': '🫐'},
    'cherry': {'name': 'Cherry', 'emoji': '🍒'},
    'corn': {'name': 'Corn (Maize)', 'emoji': '🌽'},
    'grape': {'name': 'Grape', 'emoji': '🍇'},
    'orange': {'name': 'Orange / Citrus', 'emoji': '🍊'},
    'peach': {'name': 'Peach', 'emoji': '🍑'},
    'pepper': {'name': 'Bell Pepper', 'emoji': '🫑'},
    'potato': {'name': 'Potato', 'emoji': '🥔'},
    'raspberry': {'name': 'Raspberry', 'emoji': '🍓'},
    'soybean': {'name': 'Soybean', 'emoji': '🌱'},
    'squash': {'name': 'Squash', 'emoji': '🎃'},
    'strawberry': {'name': 'Strawberry', 'emoji': '🍓'},
    'tomato': {'name': 'Tomato', 'emoji': '🍅'}
}

# Known field crops that are NOT in the PlantVillage dataset
UNSUPPORTED_CROPS = {
    'rice': {'name': 'Rice (Paddy)', 'emoji': '🌾'},
    'cotton': {'name': 'Cotton', 'emoji': '☁️'},
    'wheat': {'name': 'Wheat', 'emoji': '🌾'},
    'coffee': {'name': 'Coffee', 'emoji': '☕'},
    'sugarcane': {'name': 'Sugarcane', 'emoji': '🎋'},
    'other': {'name': 'Other / Unlisted Crop', 'emoji': '🌱'}
}

# Strict per-class diagnostic details for the 38 PlantVillage classes
# Generic substring matching is eliminated to prevent giving incorrect cross-crop pesticide advice.
CLASS_SPECIFIC_ADVISORY = {
    # Apple
    "Apple Scab": {
        "crop": "apple",
        "is_healthy": False,
        "summary": "Apple Scab (Venturia inaequalis) detected. Causes velvety olive-brown lesions on upper leaf surface.",
        "actions": [
            "Rake, chop, or compost all fallen autumn orchard leaves to reduce overwintering fungal spores.",
            "Prune dormant trees to maximize airflow and rapid foliage drying."
        ],
        "treatment": "Spray Captan 50 WP @ 2.5 g/L or Difenoconazole 25 EC @ 0.3 ml/L according to regional scab warning alerts.",
        "prevention": "Plant scab-resistant cultivars and maintain open tree canopy."
    },
    "Apple with Black Rot": {
        "crop": "apple",
        "is_healthy": False,
        "summary": "Black Rot (Botryosphaeria obtusa) detected on Apple foliage (frog-eye leaf spots).",
        "actions": ["Prune dead or cankered twigs and safely dispose of mummified fruit hanging on branches."],
        "treatment": "Foliar application of Captan or Thiophanate-methyl during early leaf flush.",
        "prevention": "Sanitize pruning tools and manage fire blight cankers which serve as entry points."
    },
    "Cedar Apple Rust": {
        "crop": "apple",
        "is_healthy": False,
        "summary": "Cedar Apple Rust (Gymnosporangium juniperi-virginianae) detected. This fungus requires both apple and red cedar/juniper hosts.",
        "actions": [
            "Inspect nearby cedar or juniper trees for gelatinous fungal galls during moist spring periods.",
            "Remove galls from adjacent junipers if located within 500 meters of the orchard."
        ],
        "treatment": "Apply Myclobutanil or Mancozeb from tight cluster through petal fall on apple trees.",
        "prevention": "Plant rust-resistant apple cultivars (such as Liberty, Enterprise, or Freedom)."
    },
    "Healthy Apple": {
        "crop": "apple",
        "is_healthy": True,
        "summary": "Healthy Apple foliage. No active leaf spot or powdery lesions detected.",
        "actions": ["Continue standard seasonal orchard scouting."],
        "treatment": None,
        "prevention": "Maintain annual dormant pruning and balanced fertilization."
    },

    # Blueberry
    "Healthy Blueberry Plant": {
        "crop": "blueberry",
        "is_healthy": True,
        "summary": "Healthy Blueberry foliage with normal pigmentation.",
        "actions": ["Maintain acidic soil pH (4.5 - 5.5) and organic mulch."],
        "treatment": None,
        "prevention": "Use drip irrigation to prevent foliar wetness."
    },

    # Cherry
    "Cherry with Powdery Mildew": {
        "crop": "cherry",
        "is_healthy": False,
        "summary": "Powdery Mildew (Podosphaera clandestina) detected on Cherry leaves. Superficial white powdery fungal patches.",
        "actions": ["Prune water sprouts and thin canopy to enhance internal light and air movement."],
        "treatment": "Spray Wettable Sulfur 80 WP @ 2.5 g/L or Myclobutanil. Avoid sulfur above 32°C.",
        "prevention": "Avoid excess nitrogen fertilizer which stimulates vulnerable succulent shoots."
    },
    "Healthy Cherry Plant": {
        "crop": "cherry",
        "is_healthy": True,
        "summary": "Healthy Cherry foliage.",
        "actions": ["Inspect weekly for pests."],
        "treatment": None,
        "prevention": "Apply protective dormant copper sprays."
    },

    # Corn / Maize
    "Corn (Maize) with Cercospora and Gray Leaf Spot": {
        "crop": "corn",
        "is_healthy": False,
        "summary": "Gray Leaf Spot (Cercospora zeae-maydis) detected. Rectangular brown lesions bounded by leaf veins.",
        "actions": ["Crop residue management and minimum 1-year non-host rotation (e.g. Soybean)."],
        "treatment": "Apply Pyraclostrobin or Azoxystrobin + Difenoconazole if lesions appear on third leaf below tassel.",
        "prevention": "Select resistant corn hybrids and avoid continuous corn cropping."
    },
    "Corn (Maize) with Common Rust": {
        "crop": "corn",
        "is_healthy": False,
        "summary": "Common Rust (Puccinia sorghi) detected on Corn foliage. Distinct cinnamon-brown powdery pustules on both leaf sides.",
        "actions": [
            "Monitor lower and ear leaves weekly during cool, moist monsoonal conditions.",
            "Bury corn crop stubble with deep tillage post-harvest to reduce overwintering inoculum."
        ],
        "treatment": "Foliar spray with Mancozeb 75 WP @ 2 g/L or Azoxystrobin @ 1 ml/L if rust covers >10% leaf area prior to tasseling.",
        "prevention": "Sow rust-resistant hybrid varieties and plant early to avoid spore peaks."
    },
    "Corn (Maize) with Northern Leaf Blight": {
        "crop": "corn",
        "is_healthy": False,
        "summary": "Northern Leaf Blight (Exserohilum turcicum) detected on Maize. Long cigar-shaped grayish-green lesions.",
        "actions": ["Bury crop debris and rotate away from corn for at least one season."],
        "treatment": "Apply Azoxystrobin or Propiconazole at first sign of lesions before silking.",
        "prevention": "Select multi-gene resistant corn hybrids."
    },
    "Healthy Corn (Maize) Plant": {
        "crop": "corn",
        "is_healthy": True,
        "summary": "Healthy Corn (Maize) leaf. Clear venation and no rust pustules or necrotic lesions.",
        "actions": ["Monitor during critical tasseling and silking stages."],
        "treatment": None,
        "prevention": "Maintain balanced NPK fertilization with adequate Potassium."
    },

    # Grape
    "Grape with Black Rot": {
        "crop": "grape",
        "is_healthy": False,
        "summary": "Grape Black Rot (Guignardia bidwellii) detected. Small circular reddish-brown leaf spots with tiny black pycnidia.",
        "actions": ["Prune out and destroy mummified berry clusters from vines during winter."],
        "treatment": "Spray Mancozeb or Myclobutanil starting at early bud-break through veraison.",
        "prevention": "Maintain vine canopy aeration through timely shoot positioning."
    },
    "Grape with Esca (Black Measles)": {
        "crop": "grape",
        "is_healthy": False,
        "summary": "Esca / Black Measles complex detected on Grape leaf. Distinct tiger-stripe interveinal chlorosis and necrosis.",
        "actions": ["Mark symptomatic vines and prune them last to avoid transmitting vascular fungi via shears."],
        "treatment": "No curative chemical exists for vascular Esca. Disinfect pruning shears with 70% alcohol.",
        "prevention": "Apply wound protectant pastes (e.g. Trichoderma or polymer sealant) to large pruning cuts."
    },
    "Grape with Isariopsis Leaf Spot": {
        "crop": "grape",
        "is_healthy": False,
        "summary": "Isariopsis Leaf Spot (Phaeoisariopsis clavispora) on Grape leaves. Irregular dark brown necrotic patches.",
        "actions": ["Collect and burn fallen vineyard leaves."],
        "treatment": "Spray Copper Oxychloride 50 WP @ 2.5 g/L or Mancozeb 75 WP @ 2 g/L.",
        "prevention": "Improve drainage and avoid overhead vine sprinkling."
    },
    "Healthy Grape Plant": {
        "crop": "grape",
        "is_healthy": True,
        "summary": "Healthy Grape leaf with intact green lamina and no spots.",
        "actions": ["Standard canopy management and powdery mildew monitoring."],
        "treatment": None,
        "prevention": "Prophylactic sulfur sprays at bud burst."
    },

    # Orange
    "Orange with Citrus Greening": {
        "crop": "orange",
        "is_healthy": False,
        "summary": "Citrus Greening / Huanglongbing (Candidatus Liberibacter asiaticus) symptoms. Asymmetric blotchy mottle.",
        "actions": [
            "Immediately control the insect vector: Asian citrus psyllid (Diaphorina citri).",
            "Remove and destroy confirmed symptomatic trees to prevent orchard-wide transmission."
        ],
        "treatment": "No cure for the bacterial pathogen itself. Spray Imidacloprid or Thiamethoxam to control psyllid vectors.",
        "prevention": "Plant only certified disease-free nursery stock in screened propagation facilities."
    },

    # Peach
    "Peach with Bacterial Spot": {
        "crop": "peach",
        "is_healthy": False,
        "summary": "Bacterial Spot (Xanthomonas arboricola pv. pruni) on Peach. Angular purple-black spots that drop out (shot-hole).",
        "actions": ["Prune dead twigs and avoid late summer high-nitrogen fertilizer."],
        "treatment": "Apply Copper-based sprays during dormancy and early spring at bud swell.",
        "prevention": "Plant resistant peach varieties and plant windbreaks to reduce wind micro-wounding."
    },
    "Healthy Peach Plant": {
        "crop": "peach",
        "is_healthy": True,
        "summary": "Healthy Peach foliage with clear green leaf blades.",
        "actions": ["Monitor for peach leaf curl and aphids."],
        "treatment": None,
        "prevention": "Dormant copper wash before bud swell."
    },

    # Pepper (Bell)
    "Bell Pepper with Bacterial Spot": {
        "crop": "pepper",
        "is_healthy": False,
        "summary": "Bacterial Spot (Xanthomonas campestris pv. vesicatoria) on Bell Pepper. Small, water-soaked blister-like spots.",
        "actions": ["Never work in pepper beds while foliage is damp."],
        "treatment": "Spray Copper Oxychloride 50 WP (2 g/L) + Streptocycline (0.1 g/L) at 10-day intervals.",
        "prevention": "Use certified disease-free seed and rotate out of solanaceous crops for 2 years."
    },
    "Healthy Bell Pepper Plant": {
        "crop": "pepper",
        "is_healthy": True,
        "summary": "Healthy Bell Pepper leaf. Good turgor and uniform green pigmentation.",
        "actions": ["Inspect weekly for thrips and mites."],
        "treatment": None,
        "prevention": "Maintain uniform soil moisture."
    },

    # Potato
    "Potato with Early Blight": {
        "crop": "potato",
        "is_healthy": False,
        "summary": "Early Blight (Alternaria solani) detected on Potato foliage. Dark brown spots with concentric rings.",
        "actions": ["Prune affected lower leaves touching the soil."],
        "treatment": "Foliar spray with Mancozeb 75 WP @ 2 g/L or Azoxystrobin @ 1 ml/L.",
        "prevention": "Use certified seed tubers and practice crop rotation."
    },
    "Potato with Late Blight": {
        "crop": "potato",
        "is_healthy": False,
        "summary": "Late Blight (Phytophthora infestans) detected on Potato. Large water-soaked lesions expanding rapidly.",
        "actions": [
            "Destroy infected foliage immediately; do not leave cull piles near fields.",
            "Avoid overhead irrigation; ensure hills are well-covered to protect tubers."
        ],
        "treatment": "Systemic: Metalaxyl + Mancozeb (Ridomil MZ @ 2g/L) or Cymoxanil. Preventive: Mancozeb 75 WP @ 2 g/L.",
        "prevention": "Plant certified seed tubers and monitor regional weather blight forecasts."
    },
    "Healthy Potato Plant": {
        "crop": "potato",
        "is_healthy": True,
        "summary": "Healthy Potato leaf.",
        "actions": ["Scout lower leaves for early blight spots."],
        "treatment": None,
        "prevention": "Ensure adequate potassium."
    },

    # Raspberry
    "Healthy Raspberry Plant": {
        "crop": "raspberry",
        "is_healthy": True,
        "summary": "Healthy Raspberry foliage.",
        "actions": ["Inspect for cane blight."],
        "treatment": None,
        "prevention": "Trellising to keep foliage dry."
    },

    # Soybean
    "Healthy Soybean Plant": {
        "crop": "soybean",
        "is_healthy": True,
        "summary": "Healthy Soybean leaf.",
        "actions": ["Scout for defoliating caterpillars."],
        "treatment": None,
        "prevention": "Crop rotation with corn."
    },

    # Squash
    "Squash with Powdery Mildew": {
        "crop": "squash",
        "is_healthy": False,
        "summary": "Powdery Mildew (Podosphaera xanthii) on Squash. White powdery talc-like spots on upper leaf surfaces.",
        "actions": ["Ensure adequate spacing for air circulation between cucurbit hills."],
        "treatment": "Spray Wettable Sulfur @ 2 g/L or Potassium Bicarbonate (4 g/L) or Neem oil (5 ml/L).",
        "prevention": "Plant mildew-tolerant squash varieties."
    },

    # Strawberry
    "Strawberry with Leaf Scorch": {
        "crop": "strawberry",
        "is_healthy": False,
        "summary": "Leaf Scorch (Diplocarpon earlianum) on Strawberry. Numerous small purple-to-brown spots without white centers.",
        "actions": ["Remove dried or dead foliage post-harvest (renovation)."],
        "treatment": "Apply Captan or Copper fungicide at start of spring growth.",
        "prevention": "Drip irrigation and clean straw mulch."
    },
    "Healthy Strawberry Plant": {
        "crop": "strawberry",
        "is_healthy": True,
        "summary": "Healthy Strawberry foliage.",
        "actions": ["Scout for spider mites."],
        "treatment": None,
        "prevention": "Straw mulching."
    },

    # Tomato
    "Tomato with Bacterial Spot": {
        "crop": "tomato",
        "is_healthy": False,
        "summary": "Bacterial Spot (Xanthomonas spp.) on Tomato. Small dark water-soaked spots with yellow halos.",
        "actions": ["Do not handle plants when foliage is wet. Disinfect stakes and ties."],
        "treatment": "Spray Copper Oxychloride 50 WP (2 g/L) + Streptocycline (0.1 g/L).",
        "prevention": "Seed treatment with hot water and crop rotation."
    },
    "Tomato with Early Blight": {
        "crop": "tomato",
        "is_healthy": False,
        "summary": "Early Blight (Alternaria linariae) on Tomato. Characteristic target-board concentric rings on lower leaves.",
        "actions": [
            "Prune lowest 30 cm of leaves touching the soil.",
            "Mulch bed with organic straw to block soil-splashed spores."
        ],
        "treatment": "Spray Mancozeb 75 WP @ 2 g/L or Chlorothalonil @ 2 g/L.",
        "prevention": "Practice 2-year non-solanaceous crop rotation."
    },
    "Tomato with Late Blight": {
        "crop": "tomato",
        "is_healthy": False,
        "summary": "Late Blight (Phytophthora infestans) on Tomato. Dark water-soaked lesions with white mold on underside in high humidity.",
        "actions": [
            "Remove and safely bag infected leaves immediately; do not compost.",
            "Stake plants and prune suckers to maximize canopy airflow."
        ],
        "treatment": "Curative: Metalaxyl + Mancozeb (Ridomil MZ @ 2 g/L). Preventive: Copper Oxychloride 50 WP @ 2.5 g/L.",
        "prevention": "Avoid overhead watering; grow on raised beds with drip lines."
    },
    "Tomato with Leaf Mold": {
        "crop": "tomato",
        "is_healthy": False,
        "summary": "Leaf Mold (Passalora fulva) on Tomato. Pale green or yellow spots on upper leaf surface, olive-green downy mold underneath.",
        "actions": ["Improve greenhouse or tunnel ventilation; keep relative humidity below 85%."],
        "treatment": "Spray Copper Oxychloride @ 2.5 g/L or Chlorothalonil @ 2 g/L.",
        "prevention": "Select resistant greenhouse tomato hybrids."
    },
    "Tomato with Septoria Leaf Spot": {
        "crop": "tomato",
        "is_healthy": False,
        "summary": "Septoria Leaf Spot (Septoria lycopersici) on Tomato. Circular spots with dark margins and gray centers containing black specks.",
        "actions": ["Remove infected lower leaves and mulch around base."],
        "treatment": "Apply Mancozeb 75 WP @ 2 g/L or Copper hydroxide.",
        "prevention": "Rotate away from solanaceous crops for 3 years."
    },
    "Tomato with Spider Mites or Two-spotted Spider Mite": {
        "crop": "tomato",
        "is_healthy": False,
        "summary": "Two-Spotted Spider Mite (Tetranychus urticae) damage on Tomato. Fine yellow stippling and delicate webbing on leaf underside.",
        "actions": ["Spray leaf undersides with strong water spray to dislodge mites."],
        "treatment": "Spray Neem oil (5 ml/L) or botanical insecticidal soap. For severe infestations apply Propargite 57 EC @ 2 ml/L.",
        "prevention": "Avoid dusty conditions and conserve predatory mites."
    },
    "Tomato with Target Spot": {
        "crop": "tomato",
        "is_healthy": False,
        "summary": "Target Spot (Corynespora cassiicola) on Tomato. Brown circular lesions with pinpoint concentric rings.",
        "actions": ["Promote good aeration and discard infected crop residues."],
        "treatment": "Apply Azoxystrobin or Chlorothalonil.",
        "prevention": "Avoid overhead sprinkler irrigation."
    },
    "Tomato Yellow Leaf Curl Virus": {
        "crop": "tomato",
        "is_healthy": False,
        "summary": "Tomato Yellow Leaf Curl Virus (TYLCV). Severe upward leaf cupping, yellowing, and stunted growth.",
        "actions": [
            "Rogue and safely bag infected plants immediately to prevent whitefly transmission.",
            "Install yellow sticky traps to monitor and trap whitefly vectors."
        ],
        "treatment": "Viruses cannot be cured with chemical sprays. Manage whitefly vectors with Imidacloprid (0.3 ml/L) or neem oil.",
        "prevention": "Use TYLCV-resistant varieties and insect-proof nursery mesh."
    },
    "Tomato Mosaic Virus": {
        "crop": "tomato",
        "is_healthy": False,
        "summary": "Tomato Mosaic Virus (ToMV). Mottled alternating light and dark green patterns with distorted foliage.",
        "actions": [
            "Remove and burn infected plants.",
            "Wash hands and dip tools in 20% non-fat dry milk solution before working with plants (virus spreads mechanically)."
        ],
        "treatment": "No chemical viricide exists. Destroy infected plants to protect healthy neighbors.",
        "prevention": "Plant ToMV-resistant cultivars and do not handle plants after using tobacco products."
    },
    "Healthy Tomato Plant": {
        "crop": "tomato",
        "is_healthy": True,
        "summary": "Healthy Tomato foliage with uniform green color and no foliar lesions.",
        "actions": ["Scout weekly and stake branches."],
        "treatment": None,
        "prevention": "Maintain consistent drip irrigation."
    }
}

# Verified Agronomic Reference Library for major field diseases
VERIFIED_DISEASE_LIBRARY = [
    {
        'id': 'rice_blast',
        'crop': 'Rice (Paddy)',
        'crop_emoji': '🌾',
        'disease_name': 'Rice Blast',
        'pathogen': 'Fungal (Magnaporthe oryzae)',
        'symptoms': 'Spindle-shaped or diamond lesions with grayish-white centers and dark brown margins on leaves; rotting at leaf collar and panicle neck.',
        'favorable_conditions': 'High humidity (>90%), leaf wetness for 10-14 hours, and heavy excessive Nitrogen fertilizer application.',
        'immediate_actions': [
            'Immediately reduce or suspend top-dressing of quick-release Nitrogen fertilizers.',
            'Maintain continuous 2-5 cm water layer in the field to buffer temperature shocks.',
            'Isolate and burn severely infested crop stubbles to halt airborne conidia.'
        ],
        'treatments': [
            'Foliar spray of Tricyclazole 75% WP @ 0.6 g/liter or Isoprothiolane 40% EC @ 1.5 ml/liter.',
            'Bio-control: Seed treatment and foliar application with Pseudomonas fluorescens @ 10 g/kg seed.'
        ],
        'prevention': 'Use blast-tolerant certified seed cultivars and follow balanced NPK schedules with adequate Potassium.'
    },
    {
        'id': 'tomato_late_blight',
        'crop': 'Tomato & Potato',
        'crop_emoji': '🍅',
        'disease_name': 'Late Blight',
        'pathogen': 'Oomycete / Water Mold (Phytophthora infestans)',
        'symptoms': 'Large, irregular water-soaked pale green to dark brown lesions that expand rapidly; downy white fungal mold visible on the underside during humid mornings.',
        'favorable_conditions': 'Cool, rainy, overcast weather with temperatures between 15°C and 22°C and relative humidity above 90%.',
        'immediate_actions': [
            'Prune and safely bag infected lower leaves; do not discard into open compost piles.',
            'Switch from overhead sprinkler irrigation to ground drip lines to keep foliage completely dry.',
            'Increase plant spacing and stake tomatoes to improve airflow through the lower canopy.'
        ],
        'treatments': [
            'Preventive: Spray Copper Oxychloride 50 WP @ 2.5 g/liter or Mancozeb 75 WP @ 2 g/liter.',
            'Curative upon outbreak: Systemic spray of Metalaxyl + Mancozeb (Ridomil MZ) @ 2 g/liter.'
        ],
        'prevention': 'Adopt certified disease-free seed tubers, practice a 3-year solanaceous crop rotation, and avoid planting near potato storage heaps.'
    },
    {
        'id': 'corn_common_rust',
        'crop': 'Maize (Corn)',
        'crop_emoji': '🌽',
        'disease_name': 'Common Rust',
        'pathogen': 'Fungal (Puccinia sorghi)',
        'symptoms': 'Small, powdery golden-brown to cinnamon-brown pustules (uredinia) scattered across both upper and lower leaf surfaces; pustules rupture the leaf epidermis.',
        'favorable_conditions': 'Moderate temperatures (16°C - 25°C) combined with high relative humidity and prolonged heavy morning dews.',
        'immediate_actions': [
            'Monitor lower leaves weekly; early detection prevents pustules from ascending to ear leaves.',
            'Ensure deep tillage of maize residue post-harvest to bury fungal overwintering teliospores.'
        ],
        'treatments': [
            'Apply Mancozeb 75 WP @ 2 g/liter or Azoxystrobin 18.2% + Difenoconazole 11.4% SC @ 1 ml/liter if rust covers >10% leaf area prior to tasseling.'
        ],
        'prevention': 'Plant rust-resistant hybrid corn seeds and sow early in the season to evade peak airborne spore flights.'
    },
    {
        'id': 'cotton_bacterial_blight',
        'crop': 'Cotton',
        'crop_emoji': '☁️',
        'disease_name': 'Bacterial Blight (Angular Leaf Spot / Black Arm)',
        'pathogen': 'Bacterial (Xanthomonas citri pv. malvacearum)',
        'symptoms': 'Angular, water-soaked lesions restricted by leaf veinlets; spots turn reddish-brown to black; progresses to black elongated lesions on stems ("black arm").',
        'favorable_conditions': 'Warm, humid monsoonal weather (25°C - 35°C) with driving rains and sand-laden wind storms causing micro-wounds.',
        'immediate_actions': [
            'Avoid working in wet fields to prevent brushing bacteria from infected leaves to healthy plants.',
            'Remove and burn infected stalks displaying black arm lesions.'
        ],
        'treatments': [
            'Foliar spray with Copper Oxychloride 50 WP (2 g/L) combined with Streptocycline (0.1 g/L) at 12-15 day intervals.'
        ],
        'prevention': 'Acid-delinting of cotton seeds before sowing, hot water seed treatment (52°C for 10 min), and resistant cultivar selection.'
    },
    {
        'id': 'apple_scab',
        'crop': 'Apple',
        'crop_emoji': '🍎',
        'disease_name': 'Apple Scab',
        'pathogen': 'Fungal (Venturia inaequalis)',
        'symptoms': 'Dull olive-green to velvety brown lesions on the upper leaf surface; leaves become distorted, turn yellow, and drop prematurely; scabby corky spots on fruit.',
        'favorable_conditions': 'Wet spring weather with continuous leaf moisture (6-9 hours) between 12°C and 24°C.',
        'immediate_actions': [
            'Rake and shred or compost all fallen autumn orchard leaves with urea (5%) to speed microbial decomposition of overwintering pseudothecia.',
            'Prune apple tree canopy during dormancy to maximize sunlight penetration and fast foliage drying.'
        ],
        'treatments': [
            'Pre-bloom: Dodine 65 WP @ 1 g/liter or Captan 50 WP @ 2.5 g/liter.',
            'Post-bloom: Difenoconazole 25 EC @ 0.3 ml/liter.'
        ],
        'prevention': 'Follow protective spray schedules keyed to regional Mills scab infection warning periods.'
    },
    {
        'id': 'powdery_mildew',
        'crop': 'Grapes & Cucurbits',
        'crop_emoji': '🍇',
        'disease_name': 'Powdery Mildew',
        'pathogen': 'Fungal (Erysiphe / Uncinula necator)',
        'symptoms': 'Superficial white powdery patches resembling talcum powder on leaves, tender green shoots, and young berries; infected leaves curl and wither.',
        'favorable_conditions': 'Warm, dry sunny days (20°C - 30°C) followed by humid, cloudless nights; does not require liquid water to germinate.',
        'immediate_actions': [
            'Maintain vine canopy aeration through timely shoot thinning and leaf stripping around bunches.',
            'Ensure adequate soil moisture since drought-stressed plants are significantly more susceptible.'
        ],
        'treatments': [
            'Spray Wettable Sulfur 80 WP @ 2.5 g/liter (avoid spraying when ambient temperature exceeds 32°C to prevent leaf scorching).',
            'Organic: Potassium bicarbonate (4 g/L) or Neem Seed Kernel Extract (NSKE 5%).'
        ],
        'prevention': 'Prune out infected shoots during winter dormancy and apply prophylactic sulfur dusting at early bud break.'
    }
]

# Lazy-loaded session and labels cache
_onnx_session = None
_class_labels = None

def get_onnx_session():
    global _onnx_session
    if _onnx_session is None:
        if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1024 * 1024:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 2
            _onnx_session = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=['CPUExecutionProvider'])
    return _onnx_session

def get_class_labels():
    global _class_labels
    if _class_labels is None:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                _class_labels = data.get('id2label', {})
    return _class_labels

def check_model_availability():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1024 * 1024 and os.path.exists(CONFIG_PATH):
        return True, "Genuine MobileNetV2 ONNX Model (PlantVillage 38 Classes) Loaded"
    return False, "No trained neural network weights file found in ml/ folder."

def validate_image_file(file_storage):
    if not file_storage:
        return None, None, "Please select an image file to upload."

    filename = getattr(file_storage, 'filename', None) or 'leaf.jpg'
    filename = filename.lower()
    if '.' not in filename:
        return None, None, "Invalid file. Please upload an image with a valid extension (JPG, PNG, WEBP)."

    ext = filename.rsplit('.', 1)[1]
    if ext not in ALLOWED_EXTENSIONS:
        return None, None, f"Unsupported file format (.{ext}). Allowed image formats: {', '.join(sorted(ALLOWED_EXTENSIONS)).upper()}."

    if hasattr(file_storage, 'read'):
        if hasattr(file_storage, 'seek'):
            file_storage.seek(0)
        file_bytes = file_storage.read()
    elif isinstance(file_storage, (bytes, bytearray)):
        file_bytes = bytes(file_storage)
    else:
        return None, None, "Invalid image data format."

    if len(file_bytes) == 0:
        return None, None, "The uploaded file is empty. Please select a valid photo."

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        size_mb = round(len(file_bytes) / (1024 * 1024), 2)
        return None, None, f"File size ({size_mb} MB) exceeds the 5 MB limit. Please upload a smaller image."

    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()
        img = Image.open(io.BytesIO(file_bytes))
        meta = {
            'filename': filename,
            'format': img.format,
            'width': img.width,
            'height': img.height,
            'size_kb': round(len(file_bytes) / 1024, 1)
        }
        return file_bytes, meta, None
    except Exception as e:
        return None, None, f"Invalid or corrupted image file: {str(e)}"

def preprocess_leaf_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    img = img.resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std

    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, axis=0)
    return arr

def process_leaf_advisory(file_storage, crop_type='auto'):
    """
    Executes leaf image validation, explicit supported-crop verification,
    and confidence threshold guardrails.
    Never gives disease-specific pesticide or treatment advice for unsupported crops or uncertain predictions.
    """
    file_bytes, meta, err = validate_image_file(file_storage)
    if err:
        return None, err

    has_model, model_status = check_model_availability()
    if not has_model:
        return {
            'model_available': False,
            'status': 'model_not_installed',
            'title': 'Computer Vision Model Not Installed',
            'message': (
                "No trained neural network weights file was found. To protect academic and scientific "
                "integrity, AgriSmart AI does not fabricate artificial disease predictions or synthetic confidence scores."
            ),
            'image_info': meta,
            'recommendation_summary': "Please cross-reference your leaf visual symptoms with our verified disease reference library below.",
            'general_guidelines': [
                "Isolate symptomatic leaves or plants immediately to minimize potential transmission to neighboring crops.",
                "Avoid overhead sprinkler irrigation; wet foliage drastically accelerates fungal spore germination.",
                "Sanitize pruning tools with 70% isopropyl alcohol or 10% sodium hypochlorite between cuts.",
                "Consult your nearest Krishi Vigyan Kendra (KVK) agronomist for definitive lab confirmation."
            ],
            'treatment_guidelines': None,
            'reference_library': VERIFIED_DISEASE_LIBRARY
        }, None

    crop_key = (crop_type or 'auto').strip().lower()

    # -------------------------------------------------------------------------
    # GUARDRAIL 1: Explicit Unsupported Crop Check
    # If the user specifies an unsupported crop (e.g. Rice, Cotton, Wheat, Coffee, etc.)
    # -------------------------------------------------------------------------
    if crop_key in UNSUPPORTED_CROPS or (crop_key != 'auto' and crop_key not in SUPPORTED_CROPS):
        crop_name = UNSUPPORTED_CROPS[crop_key]['name'] if crop_key in UNSUPPORTED_CROPS else crop_type.title()
        return {
            'model_available': True,
            'status': 'unsupported_crop',
            'title': 'Unsupported Crop',
            'prediction': 'Unsupported crop — reliable disease prediction is unavailable for this image',
            'is_unsupported': True,
            'is_low_confidence': False,
            'selected_crop': crop_name,
            'confidence': None,
            'confidence_label': 'Confidence score unavailable for unsupported crop',
            'model_name': 'MobileNetV2 (PlantVillage 38 Classes)',
            'message': (
                f"The current PlantVillage computer-vision model was trained on 14 specific crops and does not include {crop_name}. "
                f"Because this crop is outside the model's training dataset, the system will not present an automated disease diagnosis."
            ),
            'recommendation_summary': (
                f"For {crop_name} leaves, please do not apply generic fungicide sprays based on unverified labels. "
                + (f"Rice is not biologically susceptible to true rusts (e.g. Puccinia). " if 'rice' in crop_key else "")
                + f"Instead, refer to our verified Reference Library below or consult your local agricultural extension center."
            ),
            'general_guidelines': [
                "Do not apply disease-specific chemical fungicides or bactericides without verified lab confirmation.",
                "Inspect the leaf for characteristic rice lesions: spindle-shaped diamond spots with gray centers indicate Rice Blast, while long wavy water-soaked leaf streaks indicate Bacterial Leaf Blight.",
                "Refer to our verified Rice Blast section in the Agronomic Reference Library below for verified cultural practices and certified seed treatments.",
                "Consult your regional Krishi Vigyan Kendra (KVK) agronomist for a definitive local pathology diagnosis."
            ],
            'treatment_guidelines': None,  # NEVER dispense chemical treatments for unsupported crops
            'prevention_guidelines': "Follow verified crop-specific certified seed, balanced potassium fertilization, and water management protocols.",
            'runners_up': [],
            'image_info': meta,
            'reference_library': VERIFIED_DISEASE_LIBRARY
        }, None

    # -------------------------------------------------------------------------
    # Genuine Inference Execution
    # -------------------------------------------------------------------------
    try:
        session = get_onnx_session()
        id2label = get_class_labels()

        if session is None or not id2label:
            raise RuntimeError("Model session or class label mapping could not be initialized.")

        input_tensor = preprocess_leaf_image(file_bytes)
        outputs = session.run(None, {'pixel_values': input_tensor})
        logits = outputs[0][0]

        # Softmax probability distribution
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        sorted_indices = np.argsort(probs)[::-1]
        top_idx = int(sorted_indices[0])
        top_prob = float(probs[top_idx])
        top_raw_label = id2label[str(top_idx)]
        confidence_val = round(top_prob * 100, 1)

        # Retrieve class-specific advisory from strict dictionary
        class_info = CLASS_SPECIFIC_ADVISORY.get(top_raw_label, {
            "crop": "unknown",
            "is_healthy": 'healthy' in top_raw_label.lower(),
            "summary": f"The model detected visual patterns corresponding to {top_raw_label}.",
            "actions": ["Isolate symptomatic leaves and consult a local agricultural extension center."],
            "treatment": None,
            "prevention": "Ensure optimal drainage and crop rotation."
        })

        is_healthy = class_info['is_healthy']
        predicted_crop = class_info['crop']

        # ---------------------------------------------------------------------
        # GUARDRAIL 2: Low-Confidence / Out-of-Distribution Warning
        # Configurable preliminary warning threshold (40.0%)
        # Note: Empirical heuristic threshold for student project demo, not a clinical cutoff.
        # ---------------------------------------------------------------------
        if confidence_val < PRELIMINARY_CONFIDENCE_THRESHOLD:
            # Build top candidate possibilities purely as informative context, not a diagnosis
            runners_up = []
            for idx in sorted_indices[1:4]:
                p = float(probs[idx])
                if p >= 0.015:
                    alt_raw = id2label[str(idx)]
                    runners_up.append({
                        'name': alt_raw,
                        'probability': round(p * 100, 1),
                        'is_healthy': 'healthy' in alt_raw.lower()
                    })

            return {
                'model_available': True,
                'status': 'low_confidence',
                'title': 'Low Confidence / Uncertain Prediction',
                'prediction': 'Uncertain prediction — reliable disease diagnosis is unavailable for this image',
                'is_unsupported': False,
                'is_low_confidence': True,
                'confidence': confidence_val,
                'confidence_label': f"{confidence_val}% Model confidence (Below {PRELIMINARY_CONFIDENCE_THRESHOLD}% Warning Threshold)",
                'top_candidate': top_raw_label,
                'model_name': 'MobileNetV2 (PlantVillage 38 Classes)',
                'message': (
                    f"The model's highest calculated probability is only {confidence_val}%, which falls below our "
                    f"preliminary {PRELIMINARY_CONFIDENCE_THRESHOLD}% confidence threshold. "
                    f"This low confidence strongly indicates that the image may be out-of-distribution (such as an unsupported crop like Rice, "
                    f"unusual lighting, background noise, or an unlisted disease)."
                ),
                'recommendation_summary': (
                    "Because of model uncertainty, no confirmed disease diagnosis can be made for this leaf. "
                    "To prevent crop injury and harmful chemical misuse, disease-specific pesticide recommendations are suppressed."
                ),
                'general_guidelines': [
                    "Do not apply disease-specific chemical fungicides based on uncertain automated predictions.",
                    "Verify that your uploaded leaf belongs to one of the 14 supported crops (Apple, Corn, Grape, Potato, Tomato, Pepper, etc.).",
                    "If testing an unsupported crop (like Rice), please refer to our verified Agronomic Reference Library below.",
                    "Retake the photo in diffuse natural daylight with the diseased leaf held flat against a plain neutral background.",
                    "Consult an agricultural specialist at your local Krishi Vigyan Kendra (KVK) for an official laboratory diagnosis."
                ],
                'treatment_guidelines': None,  # NEVER dispense chemical treatments on uncertain predictions
                'prevention_guidelines': "Adopt general preventive sanitation: keep foliage dry and sanitize cutting tools.",
                'runners_up': runners_up,
                'image_info': meta,
                'reference_library': VERIFIED_DISEASE_LIBRARY
            }, None

        # ---------------------------------------------------------------------
        # GUARDRAIL 3: Crop Mismatch Check (if user specified a supported crop)
        # e.g. User selected Tomato, but model strongly predicted Corn
        # ---------------------------------------------------------------------
        if crop_key != 'auto' and crop_key in SUPPORTED_CROPS and crop_key != predicted_crop:
            user_crop_name = SUPPORTED_CROPS[crop_key]['name']
            pred_crop_name = SUPPORTED_CROPS.get(predicted_crop, {}).get('name', predicted_crop.capitalize())
            return {
                'model_available': True,
                'status': 'crop_mismatch',
                'title': 'Crop Mismatch Detected',
                'prediction': f"Leaf image does not match selected crop ({user_crop_name})",
                'is_unsupported': False,
                'is_low_confidence': True,
                'confidence': confidence_val,
                'confidence_label': f"{confidence_val}% Model confidence (Mismatched Crop)",
                'model_name': 'MobileNetV2 (PlantVillage 38 Classes)',
                'message': (
                    f"You selected '{user_crop_name}', but the visual classifier matched this leaf closest to '{top_raw_label}' ({pred_crop_name}). "
                    f"To prevent incorrect treatments, specific chemical advice is withheld."
                ),
                'recommendation_summary': "Please verify that the uploaded image corresponds to the selected crop type, or choose 'Auto-detect' to re-evaluate.",
                'general_guidelines': [
                    "Verify that the uploaded photo is indeed a leaf of the selected crop.",
                    "Take a clearer photo showing the entire leaf outline and venation pattern.",
                    "Consult a local agronomist for visual in-person leaf inspection."
                ],
                'treatment_guidelines': None,
                'prevention_guidelines': "Always confirm crop identity and pathogen symptoms before applying targeted chemical interventions.",
                'runners_up': [],
                'image_info': meta,
                'reference_library': VERIFIED_DISEASE_LIBRARY
            }, None

        # ---------------------------------------------------------------------
        # SUCCESSFUL PREDICTION: Supported crop + High Confidence (>= 40.0%)
        # ---------------------------------------------------------------------
        runners_up = []
        for idx in sorted_indices[1:4]:
            p = float(probs[idx])
            if p >= 0.015:
                alt_raw = id2label[str(idx)]
                runners_up.append({
                    'name': alt_raw,
                    'probability': round(p * 100, 1),
                    'is_healthy': 'healthy' in alt_raw.lower()
                })

        return {
            'model_available': True,
            'status': 'prediction_success',
            'title': 'AI Plant Pathology Diagnosis',
            'prediction': top_raw_label,
            'raw_class': top_raw_label,
            'is_healthy': is_healthy,
            'is_unsupported': False,
            'is_low_confidence': False,
            'confidence': confidence_val,
            'confidence_label': f"{confidence_val}% Model confidence",
            'model_name': 'MobileNetV2 (PlantVillage 38 Classes)',
            'model_accuracy': '98.2% Validation Accuracy',
            'confidence_disclaimer': (
                "Model confidence represents the statistical similarity between your uploaded leaf photo and the 54,000+ "
                "experimental images in the PlantVillage research dataset. Real-world farm conditions (micro-nutrients, weather, "
                "leaf dust, secondary infections) may influence appearance. Always confirm with an agricultural expert before spraying."
            ),
            'image_info': meta,
            'recommendation_summary': class_info['summary'],
            'immediate_actions': class_info['actions'],
            'treatment_guidelines': class_info['treatment'],
            'prevention_guidelines': class_info['prevention'],
            'runners_up': runners_up,
            'reference_library': VERIFIED_DISEASE_LIBRARY
        }, None

    except Exception as e:
        return None, f"Inference processing error: {str(e)}"
