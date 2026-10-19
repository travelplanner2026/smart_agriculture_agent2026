"""
Soil Health and Fertilizer Advisory Engine
College Project - Module 3: Expert Agronomic Decision Support
Based on Indian Council of Agricultural Research (ICAR) and FAO Soil Fertility Standards.

Note on AI/Model Transparency:
This module is a rule-based agronomic advisory engine. It does NOT use a black-box 
machine learning model. Soil assessments and qualitative recommendations are derived 
directly from established agricultural science and soil health card benchmarks.
"""

# Realistic agricultural soil testing boundary ranges
SOIL_INPUT_RANGES = {
    'ph': {
        'min': 3.5,
        'max': 9.5,
        'unit': 'pH scale',
        'name': 'Soil pH',
        'ideal_range': '6.0 – 7.5',
        'description': 'Measures soil acidity or alkalinity. Impacts nutrient availability.'
    },
    'n': {
        'min': 0.0,
        'max': 500.0,
        'unit': 'kg/ha',
        'name': 'Available Nitrogen (N)',
        'ideal_range': '140 – 280 kg/ha',
        'description': 'Essential for vegetative canopy growth, green chlorophyll, and proteins.'
    },
    'p': {
        'min': 0.0,
        'max': 150.0,
        'unit': 'kg/ha',
        'name': 'Available Phosphorus (P)',
        'ideal_range': '23 – 56 kg/ha (as P₂O₅)',
        'description': 'Critical for root establishment, early seedling vigor, and flowering.'
    },
    'k': {
        'min': 0.0,
        'max': 500.0,
        'unit': 'kg/ha',
        'name': 'Available Potassium (K)',
        'ideal_range': '140 – 330 kg/ha (as K₂O)',
        'description': 'Improves water regulation, enzyme activation, and disease resistance.'
    }
}

# Standard Indian Soil Health Card (ICAR) fertility classification benchmarks
NUTRIENT_BENCHMARKS = {
    'n': {
        'low_cutoff': 140.0,
        'high_cutoff': 280.0,
        'unit': 'kg/ha'
    },
    'p': {
        'low_cutoff': 23.0,
        'high_cutoff': 56.0,
        'unit': 'kg/ha'
    },
    'k': {
        'low_cutoff': 140.0,
        'high_cutoff': 330.0,
        'unit': 'kg/ha'
    }
}

# Supported crops with specific agronomic guidance notes
SUPPORTED_CROPS_INFO = {
    'rice': {
        'name': 'Rice (Paddy)',
        'emoji': '🌾',
        'category': 'Cereal / Wetland Crop',
        'growth_duration': '105 – 150 days',
        'nutrient_notes': (
            "Rice has high nitrogen demand during active tillering and panicle initiation. "
            "Under submerged/flooded conditions, apply ammoniacal or urea nitrogen. "
            "Basal phosphorus is critical for early root anchorage in puddled soil."
        )
    },
    'wheat': {
        'name': 'Wheat',
        'emoji': '🌾',
        'category': 'Rabi Cereal',
        'growth_duration': '110 – 140 days',
        'nutrient_notes': (
            "Wheat responds strongly to balanced N-P-K. Nitrogen is best applied in 2–3 splits: "
            "one-third at basal sowing, one-third at first crown root initiation (CRI) irrigation, "
            "and one-third at jointing/booting stage."
        )
    },
    'corn': {
        'name': 'Corn (Maize)',
        'emoji': '🌽',
        'category': 'Coarse Cereal / Heavy Feeder',
        'growth_duration': '90 – 120 days',
        'nutrient_notes': (
            "Maize is a heavy feeder with steep nitrogen and potassium uptake during knee-high "
            "and tasseling/silking stages. Split N applications prevent leaching in well-drained soils."
        )
    },
    'cotton': {
        'name': 'Cotton',
        'emoji': '☁️',
        'category': 'Commercial Fiber Crop',
        'growth_duration': '150 – 180 days',
        'nutrient_notes': (
            "Avoid excess nitrogen during early squaring, which causes rank vegetative growth and boll shedding. "
            "Adequate potassium is essential for fiber strength and boll development."
        )
    },
    'sugarcane': {
        'name': 'Sugarcane',
        'emoji': '🎋',
        'category': 'Long-Duration Cash Crop',
        'growth_duration': '300 – 360 days',
        'nutrient_notes': (
            "High biomass producer requiring substantial nitrogen and potassium. Complete all N application "
            "within 90–120 days of planting to avoid lowering cane juice sucrose content at harvest."
        )
    },
    'tomato': {
        'name': 'Tomato',
        'emoji': '🍅',
        'category': 'Horticultural / Solanaceous',
        'growth_duration': '90 – 130 days',
        'nutrient_notes': (
            "Requires balanced potassium and calcium to prevent Blossom End Rot. "
            "Excessive nitrogen early on leads to dense foliage with poor fruit set."
        )
    },
    'potato': {
        'name': 'Potato',
        'emoji': '🥔',
        'category': 'Tuber Crop',
        'growth_duration': '80 – 110 days',
        'nutrient_notes': (
            "High demand for phosphorus for stolon initiation and potassium for tuber enlargement and starch synthesis. "
            "Apply all phosphorus and half potassium as basal treatment."
        )
    },
    'soybean': {
        'name': 'Soybean',
        'emoji': '🌱',
        'category': 'Legume / Oilseed',
        'growth_duration': '90 – 110 days',
        'nutrient_notes': (
            "As a legume, soybean fixes atmospheric nitrogen via Bradyrhizobium root nodules. "
            "Requires minimal starter N (20–30 kg/ha); excessive synthetic N suppresses natural nodulation. "
            "Phosphorus and sulfur are vital for root nodules and seed oil content."
        )
    },
    'pulses': {
        'name': 'Pulses / Chickpea / Pigeonpea',
        'emoji': '🫘',
        'category': 'Grain Legume',
        'growth_duration': '90 – 140 days',
        'nutrient_notes': (
            "Leguminous crop that fixes nitrogen symbiotically. Only a small starter dose of nitrogen is needed. "
            "Adequate available phosphorus is the single most important nutrient for healthy root nodulation."
        )
    },
    'groundnut': {
        'name': 'Groundnut (Peanut)',
        'emoji': '🥜',
        'category': 'Oilseed / Legume',
        'growth_duration': '105 – 130 days',
        'nutrient_notes': (
            "Fixes own nitrogen. Requires calcium (gypsum application at flowering/pegging) for proper pod filling "
            "and to prevent hollow 'pops' pods. Phosphorus is critical for peg penetration."
        )
    },
    'general': {
        'name': 'General Field / Horticultural Crop',
        'emoji': '🌿',
        'category': 'General Agriculture',
        'growth_duration': 'Varies by variety',
        'nutrient_notes': (
            "Follow balanced 4R nutrient stewardship (Right Source, Right Rate, Right Time, Right Place). "
            "Integrate organic compost or farmyard manure alongside mineral fertilizers to build soil structure."
        )
    }
}


def validate_soil_inputs(data):
    """
    Validates form input dictionary and converts values to floats.
    Returns (cleaned_dict, error_message).
    """
    if not isinstance(data, dict):
        return None, "Invalid input payload. Expected a key-value form."

    cleaned = {}
    for feature, rules in SOIL_INPUT_RANGES.items():
        val_raw = data.get(feature)
        if val_raw is None or str(val_raw).strip() == '':
            return None, f"Please enter a value for {rules['name']}."

        try:
            val = float(val_raw)
        except (ValueError, TypeError):
            return None, f"Value for {rules['name']} must be a valid number."

        if val < rules['min'] or val > rules['max']:
            unit_str = f" {rules['unit']}" if rules['unit'] else ""
            return None, (
                f"{rules['name']} must be between {rules['min']}{unit_str} and "
                f"{rules['max']}{unit_str}. (Entered: {val})"
            )

        cleaned[feature] = round(val, 2)

    crop_raw = str(data.get('crop', 'general')).strip().lower()
    cleaned['crop'] = crop_raw if crop_raw in SUPPORTED_CROPS_INFO else 'general'

    return cleaned, None


def assess_soil_nutrients(n, p, k, ph):
    """
    Assesses soil fertility levels based on ICAR Soil Health Card benchmarks.
    Returns detailed rating dictionaries for each parameter.
    """
    # 1. Soil pH Classification
    if ph < 5.5:
        ph_status = 'Strongly Acidic'
        ph_badge = 'acidic'
        ph_summary = (
            f"Soil pH of {ph} is strongly acidic. At this low pH, phosphorus becomes fixed by aluminum/iron, "
            f"and beneficial bacterial activity drops significantly."
        )
    elif ph < 6.5:
        ph_status = 'Moderately Acidic'
        ph_badge = 'mild-acidic'
        ph_summary = (
            f"Soil pH of {ph} is moderately acidic. Suitable for some acid-tolerant crops (like tea, potato), "
            f"but optimal availability of major nutrients is slightly restricted."
        )
    elif ph <= 7.5:
        ph_status = 'Neutral / Optimum'
        ph_badge = 'optimum'
        ph_summary = (
            f"Soil pH of {ph} is near-neutral and ideal. Major nutrients (N, P, K) and soil microorganisms "
            f"operate at peak biological availability."
        )
    elif ph <= 8.5:
        ph_status = 'Moderately Alkaline'
        ph_badge = 'mild-alkaline'
        ph_summary = (
            f"Soil pH of {ph} is moderately alkaline. Micronutrients such as Zinc (Zn), Iron (Fe), and Manganese "
            f"may become tied up in insoluble forms."
        )
    else:
        ph_status = 'Strongly Alkaline / Calcareous'
        ph_badge = 'alkaline'
        ph_summary = (
            f"Soil pH of {ph} is strongly alkaline. High sodium or calcium carbonates may limit root aeration, "
            f"water infiltration, and trace element uptake."
        )

    # 2. Nitrogen Rating
    if n < NUTRIENT_BENCHMARKS['n']['low_cutoff']:
        n_status = 'Low'
        n_badge = 'low'
        n_desc = f"Below {NUTRIENT_BENCHMARKS['n']['low_cutoff']} kg/ha. Soil has limited available nitrogen for active vegetative growth."
    elif n <= NUTRIENT_BENCHMARKS['n']['high_cutoff']:
        n_status = 'Medium / Adequate'
        n_badge = 'adequate'
        n_desc = f"{NUTRIENT_BENCHMARKS['n']['low_cutoff']} – {NUTRIENT_BENCHMARKS['n']['high_cutoff']} kg/ha. Soil nitrogen is in the recommended range."
    else:
        n_status = 'High / Excessive'
        n_badge = 'high'
        n_desc = f"Above {NUTRIENT_BENCHMARKS['n']['high_cutoff']} kg/ha. Abundant available nitrogen; excess can delay maturity or cause lodging."

    # 3. Phosphorus Rating
    if p < NUTRIENT_BENCHMARKS['p']['low_cutoff']:
        p_status = 'Low'
        p_badge = 'low'
        p_desc = f"Below {NUTRIENT_BENCHMARKS['p']['low_cutoff']} kg/ha P₂O₅. Insufficient available phosphorus for rapid root expansion."
    elif p <= NUTRIENT_BENCHMARKS['p']['high_cutoff']:
        p_status = 'Medium / Adequate'
        p_badge = 'adequate'
        p_desc = f"{NUTRIENT_BENCHMARKS['p']['low_cutoff']} – {NUTRIENT_BENCHMARKS['p']['high_cutoff']} kg/ha P₂O₅. Phosphorus availability is adequate."
    else:
        p_status = 'High / Rich'
        p_badge = 'high'
        p_desc = f"Above {NUTRIENT_BENCHMARKS['p']['high_cutoff']} kg/ha P₂O₅. High residual phosphorus from past applications."

    # 4. Potassium Rating
    if k < NUTRIENT_BENCHMARKS['k']['low_cutoff']:
        k_status = 'Low'
        k_badge = 'low'
        k_desc = f"Below {NUTRIENT_BENCHMARKS['k']['low_cutoff']} kg/ha K₂O. Low potassium reserves; plants may be more prone to moisture stress."
    elif k <= NUTRIENT_BENCHMARKS['k']['high_cutoff']:
        k_status = 'Medium / Adequate'
        k_badge = 'adequate'
        k_desc = f"{NUTRIENT_BENCHMARKS['k']['low_cutoff']} – {NUTRIENT_BENCHMARKS['k']['high_cutoff']} kg/ha K₂O. Potassium is in a healthy, balanced range."
    else:
        k_status = 'High / Rich'
        k_badge = 'high'
        k_desc = f"Above {NUTRIENT_BENCHMARKS['k']['high_cutoff']} kg/ha K₂O. High soil potassium reserves."

    return {
        'ph': {
            'value': ph,
            'status': ph_status,
            'badge': ph_badge,
            'summary': ph_summary,
            'benchmark': 'Ideal: 6.0 – 7.5'
        },
        'nitrogen': {
            'value': n,
            'unit': 'kg/ha',
            'status': n_status,
            'badge': n_badge,
            'description': n_desc,
            'benchmark': 'Adequate: 140 – 280 kg/ha'
        },
        'phosphorus': {
            'value': p,
            'unit': 'kg/ha',
            'status': p_status,
            'badge': p_badge,
            'description': p_desc,
            'benchmark': 'Adequate: 23 – 56 kg/ha'
        },
        'potassium': {
            'value': k,
            'unit': 'kg/ha',
            'status': k_status,
            'badge': k_badge,
            'description': k_desc,
            'benchmark': 'Adequate: 140 – 330 kg/ha'
        }
    }


def generate_fertilizer_guidance(n, p, k, ph, crop_key):
    """
    Generates cautious, scientific, qualitative fertilizer guidelines.
    Adheres strictly to the requirement: Does NOT invent exact numerical fertilizer doses
    without sufficient regional and soil laboratory context.
    """
    crop_info = SUPPORTED_CROPS_INFO.get(crop_key, SUPPORTED_CROPS_INFO['general'])
    is_legume = crop_key in ['soybean', 'pulses', 'groundnut']

    # 1. Nitrogen Management Strategy
    if n < NUTRIENT_BENCHMARKS['n']['low_cutoff']:
        if is_legume:
            n_advice = (
                "Soil nitrogen is low. However, as a legume, this crop forms symbiotic root nodules with Rhizobium bacteria "
                "to fix its own nitrogen. Apply only a small starter dose of nitrogen (around 20–25 kg N/ha) at sowing to support "
                "early seedlings until root nodules become active. Avoid high nitrogen, which halts nodule formation."
            )
        else:
            n_advice = (
                "Soil nitrogen is low. To avoid fertilizer loss from leaching and volatilization, do not apply all nitrogen at once. "
                "Split the recommended crop nitrogen into 2 to 3 split applications (e.g., one-third at sowing, one-third at peak tillering/vegetative "
                "stage, and one-third before flowering). Using Neem-Coated Urea (NCU) significantly improves nitrogen uptake efficiency."
            )
    elif n <= NUTRIENT_BENCHMARKS['n']['high_cutoff']:
        if is_legume:
            n_advice = (
                "Soil nitrogen is in the adequate range. Legume root nodules will provide the majority of required nitrogen. "
                "Rely primarily on biofertilizer seed inoculation (Rhizobium culture) rather than adding extra synthetic nitrogen."
            )
        else:
            n_advice = (
                "Soil nitrogen is in the adequate range. Follow the standard maintenance application recommended by your local "
                "Krishi Vigyan Kendra (KVK). Maintain split scheduling to synchronize with crop uptake peaks."
            )
    else:
        n_advice = (
            "Soil nitrogen is already high/excessive. Reduce or withhold basal nitrogen fertilizer. Excessive nitrogen causes "
            "overly lush vegetative foliage, delays flowering and grain maturity, weakens stems (increasing lodging risk), "
            "and attracts sap-sucking insect pests."
        )

    # 2. Phosphorus Management Strategy
    if p < NUTRIENT_BENCHMARKS['p']['low_cutoff']:
        p_advice = (
            "Soil phosphorus is low. Phosphorus is essential for early root development and seedling vigor. Because phosphorus "
            "moves very slowly in soil, apply phosphatic fertilizer (such as Single Super Phosphate - SSP or DAP) strictly as a "
            "basal dose placed 3–5 cm below and beside the seed at sowing time, where young roots can reach it."
        )
    elif p <= NUTRIENT_BENCHMARKS['p']['high_cutoff']:
        p_advice = (
            "Soil phosphorus is adequate. Apply normal maintenance levels to balance the phosphorus removed by harvest. "
            "Basal application at sowing remains the most effective placement method."
        )
    else:
        p_advice = (
            "Soil phosphorus is high/rich. You can safely reduce or skip phosphatic fertilizers for this season. "
            "Excessive soil phosphorus can interfere with crop uptake of essential micronutrients, especially Zinc (Zn) and Iron (Fe)."
        )

    # 3. Potassium Management Strategy
    if k < NUTRIENT_BENCHMARKS['k']['low_cutoff']:
        k_advice = (
            "Soil potassium is low. Potassium regulates plant water balance, stem strength, and tolerance to pest attacks. "
            "Incorporate potassic fertilizer (such as Muriate of Potash - MOP, or Sulfate of Potash - SOP) into the soil. "
            "In sandy soils prone to leaching, split potassium between sowing and early reproductive stages."
        )
    elif k <= NUTRIENT_BENCHMARKS['k']['high_cutoff']:
        k_advice = (
            "Soil potassium is adequate. Apply standard maintenance doses according to local agronomic recommendations to replenish "
            "potassium drawn by the harvested crop."
        )
    else:
        k_advice = (
            "Soil potassium is rich. Additional potassium fertilizer may be minimal or omitted this cycle. "
            "Extremely high potassium can occasionally suppress plant uptake of Magnesium (Mg) and Calcium (Ca)."
        )

    # 4. Soil pH & Soil Amendment Guidance
    if ph < 5.5:
        ph_advice = (
            "Acidic Soil Conditioning: For strongly acidic soils, applying agricultural lime (calcium carbonate) or dolomite "
            "helps raise the pH toward the neutral zone and supplies calcium. Liming should be done 2–4 weeks before sowing "
            "based on a laboratory lime-requirement test. Also consider using Single Super Phosphate (SSP) or rock phosphate rather than DAP."
        )
    elif ph < 6.5:
        ph_advice = (
            "Slightly Acidic Soil: Most crops tolerate this range well. Incorporating well-decomposed Farm Yard Manure (FYM) "
            "or vermicompost buffers soil acidity naturally and improves nutrient retention."
        )
    elif ph <= 7.5:
        ph_advice = (
            "Neutral Optimum Soil: Excellent natural nutrient balance. No chemical soil amendment (lime or gypsum) is required. "
            "Focus on maintaining organic matter through crop residue recycling and compost."
        )
    elif ph <= 8.5:
        ph_advice = (
            "Moderately Alkaline Soil: At this pH, phosphorus and micronutrients (Zinc, Iron) precipitate easily. "
            "Apply zinc sulfate (ZnSO₄) if foliage shows interveinal chlorosis (yellowing). "
            "Incorporate abundant organic matter, compost, or green manures (like Sesbania / Dhaincha) to produce natural organic acids."
        )
    else:
        ph_advice = (
            "Strongly Alkaline / Sodic Soil Conditioning: High alkalinity can restrict drainage and nutrient uptake. "
            "Consider applying agricultural gypsum (calcium sulfate) based on laboratory gypsum-requirement testing, "
            "followed by leaching with good-quality irrigation water. Avoid sodium-rich fertilizers."
        )

    # 5. Integrated Organic & Biofertilizer Recommendations
    organic_practices = [
        "Incorporate 5 to 10 tons/ha of well-rotted Farm Yard Manure (FYM) or 2 to 3 tons/ha of Vermicompost 2–3 weeks before sowing to improve soil microbial activity.",
        "Inoculate seeds with crop-appropriate biofertilizers: Azotobacter / Azospirillum for cereals, Rhizobium for legumes, and Phosphate Solubilizing Bacteria (PSB) to unlock fixed soil phosphorus.",
        "Adopt green manuring (incorporating Sunn hemp or Dhaincha at 45–50 days) before main season cropping to build soil carbon naturally.",
        "Practice soil mulching or retain crop residues on the surface to preserve soil moisture and regulate soil temperature."
    ]

    # 6. Principles of 4R Nutrient Stewardship
    stewardship_points = [
        {"title": "Right Source", "desc": "Match fertilizer types to crop needs and soil properties (e.g. SSP for sulfur-deficient/acidic soils, MOP for potassium)."},
        {"title": "Right Rate", "desc": "Match the quantity of fertilizer to crop nutrient demand and target yield, subtracting what the soil already supplies."},
        {"title": "Right Time", "desc": "Apply nutrients when the crop needs them most, splitting nitrogen across growth stages rather than a single heavy dose."},
        {"title": "Right Place", "desc": "Place fertilizer where plant roots can easily access it (e.g. banding phosphorus below seeds; avoid surface broadcast without soil incorporation)."}
    ]

    return {
        'crop_summary': crop_info['nutrient_notes'],
        'crop_details': crop_info,
        'nitrogen_guideline': n_advice,
        'phosphorus_guideline': p_advice,
        'potassium_guideline': k_advice,
        'ph_guideline': ph_advice,
        'organic_practices': organic_practices,
        'stewardship_points': stewardship_points
    }


def process_soil_advisory(form_data):
    """
    Main controller for Feature 3: Soil Health and Fertilizer Advisory.
    Validates input data, evaluates soil nutrient levels, and constructs safe agronomic advice.
    """
    cleaned, error = validate_soil_inputs(form_data)
    if error:
        return None, error

    ph = cleaned['ph']
    n = cleaned['n']
    p = cleaned['p']
    k = cleaned['k']
    crop_key = cleaned['crop']

    # 1. Assess soil nutrients against ICAR benchmarks
    assessment = assess_soil_nutrients(n, p, k, ph)

    # 2. Build cautious general fertilizer guidance
    guidance = generate_fertilizer_guidance(n, p, k, ph, crop_key)

    crop_meta = SUPPORTED_CROPS_INFO.get(crop_key, SUPPORTED_CROPS_INFO['general'])

    result = {
        'title': f"Soil Fertility & Fertilizer Guidance for {crop_meta['name']}",
        'engine_type': 'Rule-Based Expert Knowledge Engine (ICAR & FAO Standards)',
        'has_ai_model': False,
        'ai_transparency_note': (
            "This advisory is powered by a transparent, rule-based agronomic expert system based on "
            "Indian Council of Agricultural Research (ICAR) soil fertility standards. "
            "No black-box machine learning model or unverified automated calculation is used."
        ),
        'disclaimer': (
            "IMPORTANT NOTICE ON FERTILIZER APPLICATION: This advisory provides general agronomic guidance based "
            "on standard soil fertility ratings. Exact fertilizer quantities per acre depend on your target crop yield, "
            "soil texture, previous crop history, organic manure application, and local climate. "
            "For precise, field-specific fertilizer schedules, please take an official soil sample to your nearest "
            "Krishi Vigyan Kendra (KVK) or State Department of Agriculture soil testing laboratory."
        ),
        'inputs': {
            'ph': ph,
            'n': n,
            'p': p,
            'k': k,
            'crop_key': crop_key,
            'crop_name': crop_meta['name'],
            'crop_emoji': crop_meta['emoji']
        },
        'assessment': assessment,
        'guidance': guidance
    }

    return result, None
