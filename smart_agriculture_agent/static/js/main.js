/**
 * Smart Agriculture AI Agent - Frontend JavaScript
 * Clean, modular interactive logic for homepage and Crop Recommendation
 */

// Preset Sample Data for One-Click Demonstrations
const CROP_PRESETS = {
    'rice': {
        N: 90, P: 42, K: 43, ph: 6.5, temperature: 20.8, humidity: 82.0, rainfall: 202.9
    },
    'maize': {
        N: 71, P: 54, K: 20, ph: 6.5, temperature: 22.6, humidity: 65.0, rainfall: 95.0
    },
    'coffee': {
        N: 101, P: 28, K: 32, ph: 6.8, temperature: 26.5, humidity: 58.0, rainfall: 158.0
    },
    'watermelon': {
        N: 99, P: 17, K: 50, ph: 6.5, temperature: 26.0, humidity: 85.0, rainfall: 50.0
    },
    'apple': {
        N: 21, P: 134, K: 199, ph: 5.9, temperature: 22.3, humidity: 92.3, rainfall: 110.0
    },
    'cotton': {
        N: 117, P: 46, K: 19, ph: 6.9, temperature: 23.9, humidity: 79.8, rainfall: 79.0
    }
};

// Function to populate form fields when a preset button is clicked
function applyPreset(cropKey) {
    const data = CROP_PRESETS[cropKey];
    if (!data) return;

    hideError();

    document.getElementById('inputN').value = data.N;
    document.getElementById('inputP').value = data.P;
    document.getElementById('inputK').value = data.K;
    document.getElementById('inputPH').value = data.ph;
    document.getElementById('inputTemp').value = data.temperature;
    document.getElementById('inputHumidity').value = data.humidity;
    document.getElementById('inputRainfall').value = data.rainfall;

    // Optional subtle feedback animation
    const formCard = document.querySelector('.form-container-card');
    if (formCard) {
        formCard.style.boxShadow = '0 0 0 3px rgba(76, 175, 80, 0.35)';
        setTimeout(() => {
            formCard.style.boxShadow = '';
        }, 500);
    }
}

// Reset form inputs and clear errors
function resetForm() {
    const form = document.getElementById('cropForm');
    if (form) form.reset();
    hideError();
    const emptyState = document.getElementById('emptyState');
    const dynamicResult = document.getElementById('dynamicResult');
    if (emptyState) emptyState.style.display = 'block';
    if (dynamicResult) dynamicResult.style.display = 'none';
}

// Show error banner
function showError(message) {
    const banner = document.getElementById('errorBanner');
    const msgSpan = document.getElementById('errorMessage');
    if (banner && msgSpan) {
        msgSpan.textContent = message;
        banner.style.display = 'flex';
        banner.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
}

// Hide error banner
function hideError() {
    const banner = document.getElementById('errorBanner');
    if (banner) {
        banner.style.display = 'none';
    }
}

// Client-side Input Validation
function validateInputs(data) {
    const fields = [
        { key: 'N', label: 'Nitrogen (N)', min: 0, max: 200, unit: 'kg/ha' },
        { key: 'P', label: 'Phosphorus (P)', min: 0, max: 200, unit: 'kg/ha' },
        { key: 'K', label: 'Potassium (K)', min: 0, max: 300, unit: 'kg/ha' },
        { key: 'ph', label: 'Soil pH', min: 3.0, max: 10.0, unit: '' },
        { key: 'temperature', label: 'Temperature', min: 0.0, max: 55.0, unit: '°C' },
        { key: 'humidity', label: 'Humidity', min: 5.0, max: 100.0, unit: '%' },
        { key: 'rainfall', label: 'Rainfall', min: 0.0, max: 500.0, unit: 'mm' }
    ];

    for (const f of fields) {
        const valStr = data[f.key];
        if (valStr === undefined || valStr === null || valStr.trim() === '') {
            return `Please enter a value for ${f.label}.`;
        }
        const val = parseFloat(valStr);
        if (isNaN(val)) {
            return `${f.label} must be a valid number.`;
        }
        if (val < f.min || val > f.max) {
            const unit = f.unit ? ` ${f.unit}` : '';
            return `${f.label} must be between ${f.min}${unit} and ${f.max}${unit}.`;
        }
    }
    return null;
}

// Initialize form handling
document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('cropForm');
    if (!form) return;

    form.addEventListener('submit', async function(e) {
        e.preventDefault(); // Prevent standard reload to provide instant AJAX feedback
        hideError();

        // Collect inputs
        const formData = {
            N: document.getElementById('inputN').value,
            P: document.getElementById('inputP').value,
            K: document.getElementById('inputK').value,
            ph: document.getElementById('inputPH').value,
            temperature: document.getElementById('inputTemp').value,
            humidity: document.getElementById('inputHumidity').value,
            rainfall: document.getElementById('inputRainfall').value
        };

        // Client-side validation
        const clientError = validateInputs(formData);
        if (clientError) {
            showError(clientError);
            return;
        }

        // Show loading spinner
        const submitBtn = document.getElementById('submitBtn');
        const spinner = document.getElementById('loadingSpinner');
        const btnText = submitBtn.querySelector('.btn-text');
        
        submitBtn.disabled = true;
        if (spinner) spinner.style.display = 'inline-block';
        if (btnText) btnText.textContent = ' Analyzing Soil...';

        try {
            const response = await fetch('/api/recommend-crop', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });

            const result = await response.json();

            if (!response.ok || !result.success) {
                showError(result.error || 'Failed to generate recommendation. Please check your inputs.');
                return;
            }

            // Render dynamic result
            renderPredictionResult(result.data);

        } catch (err) {
            console.error('Fetch error:', err);
            showError('Unable to connect to server. Please ensure the Python server is running.');
        } finally {
            submitBtn.disabled = false;
            if (spinner) spinner.style.display = 'none';
            if (btnText) btnText.textContent = '🌾 Recommend Crop';
        }
    });
});

// Render the prediction data on screen
function renderPredictionResult(data) {
    const emptyState = document.getElementById('emptyState');
    const dynamicResult = document.getElementById('dynamicResult');

    if (emptyState) emptyState.style.display = 'none';

    let alternativesHtml = '';
    if (data.runners_up && data.runners_up.length > 0) {
        const chips = data.runners_up.map(alt => `
            <span class="alt-chip">${alt.emoji} ${alt.name} (${alt.probability}%)</span>
        `).join('');

        alternativesHtml = `
            <div class="alternatives-box">
                <span class="alternatives-title">Alternative Suitable Crops:</span>
                <div class="alternatives-chips">${chips}</div>
            </div>
        `;
    }

    // Determine confidence pill HTML based on whether model supports reliable probabilities
    const confidenceHtml = data.has_confidence
        ? `<div class="confidence-pill"><span class="confidence-dot"></span><span>${data.confidence_label || (data.confidence + '% Model Consensus')}</span></div>`
        : `<div class="confidence-pill confidence-pill-muted"><span class="confidence-dot dot-muted"></span><span>Confidence score unavailable</span></div>`;

    dynamicResult.innerHTML = `
        <div class="result-header">
            <span class="result-eyebrow">AI Recommendation Generated</span>
            <div class="crop-hero-badge">
                <span class="crop-emoji">${data.emoji}</span>
                <div class="crop-hero-info">
                    <h2 class="crop-name">${data.crop_name}</h2>
                    ${confidenceHtml}
                </div>
            </div>
        </div>

        <!-- Educational Model Confidence Disclaimer -->
        <div class="confidence-disclaimer-box">
            <span class="disclaimer-icon">ℹ️</span>
            <div class="disclaimer-text">
                <strong>Note on Model Confidence:</strong> ${data.confidence_disclaimer || 'Model consensus represents the statistical agreement among decision trees based on historical data. It does not guarantee that a crop will thrive or yield in the field, as unmeasured real-world factors also determine farming outcomes.'}
            </div>
        </div>

        <div class="result-summary-box">
            <p class="summary-text">${data.summary}</p>
        </div>

        <div class="specs-grid">
            <div class="spec-card">
                <span class="spec-label">📅 Sowing Season</span>
                <span class="spec-val">${data.season}</span>
            </div>
            <div class="spec-card">
                <span class="spec-label">🌱 Ideal Soil Type</span>
                <span class="spec-val">${data.soil}</span>
            </div>
            <div class="spec-card">
                <span class="spec-label">⏳ Growth Cycle</span>
                <span class="spec-val">${data.duration}</span>
            </div>
            <div class="spec-card">
                <span class="spec-label">💧 Water Requirement</span>
                <span class="spec-val">${data.water_need}</span>
            </div>
        </div>

        <div class="advisory-box">
            <h4 class="advisory-heading">💡 Agronomic Best Practices</h4>
            <div class="tip-item">
                <strong>Fertilizer Strategy:</strong> ${data.fertilizer_tip}
            </div>
            <div class="tip-item">
                <strong>Care & Monitoring:</strong> ${data.care_tip}
            </div>
        </div>

        ${alternativesHtml}
    `;

    dynamicResult.style.display = 'block';

    // Smooth scroll into view on mobile
    if (window.innerWidth < 960) {
        dynamicResult.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

/* ==========================================================================
   Modal Functions for Homepage Previews
   ========================================================================== */

function showFeatureInfo(title, description) {
    const modal = document.getElementById('featureModal');
    const modalTitle = document.getElementById('modalTitle');
    const modalDesc = document.getElementById('modalDescription');

    if (modal && modalTitle && modalDesc) {
        modalTitle.textContent = title;
        modalDesc.textContent = description;
        modal.classList.add('active');
    }
}

function closeModal() {
    const modal = document.getElementById('featureModal');
    if (modal) {
        modal.classList.remove('active');
    }
}

window.addEventListener('click', function(event) {
    const modal = document.getElementById('featureModal');
    if (event.target === modal) {
        closeModal();
    }
});

window.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        closeModal();
    }
});


/* ==========================================================================
   Crop Disease Advisory Feature Logic (Module 2)
   ========================================================================== */

let selectedLeafFile = null;

// Handle file selection and preview
function handleLeafFileSelection(file) {
    if (!file) return;

    hideError();

    // 1. Extension Validation
    const validExtensions = ['jpg', 'jpeg', 'png', 'webp'];
    const fileName = file.name.toLowerCase();
    const ext = fileName.split('.').pop();

    if (!validExtensions.includes(ext)) {
        showError(`Unsupported file format (.${ext}). Please select a JPG, JPEG, PNG, or WEBP image.`);
        clearSelectedImage();
        return;
    }

    // 2. File Size Validation (Max 5MB)
    const maxSize = 5 * 1024 * 1024;
    if (file.size > maxSize) {
        const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
        showError(`File size (${sizeMb} MB) exceeds the 5 MB limit. Please select a smaller photo.`);
        clearSelectedImage();
        return;
    }

    selectedLeafFile = file;

    // 3. Read and Show Live Preview
    const reader = new FileReader();
    reader.onload = function(e) {
        const previewContainer = document.getElementById('previewContainer');
        const dropZone = document.getElementById('dropZone');
        const img = document.getElementById('imagePreview');
        const metaName = document.getElementById('metaFileName');
        const metaSize = document.getElementById('metaFileSize');

        if (img) img.src = e.target.result;
        if (metaName) metaName.textContent = file.name;
        if (metaSize) metaSize.textContent = formatBytes(file.size);

        if (previewContainer) previewContainer.style.display = 'block';
        if (dropZone) dropZone.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

// Clear currently selected image
function clearSelectedImage() {
    selectedLeafFile = null;
    const input = document.getElementById('leafImageInput');
    if (input) input.value = '';

    const previewContainer = document.getElementById('previewContainer');
    const dropZone = document.getElementById('dropZone');
    const img = document.getElementById('imagePreview');

    if (img) img.src = '';
    if (previewContainer) previewContainer.style.display = 'none';
    if (dropZone) dropZone.style.display = 'block';

    const emptyState = document.getElementById('emptyDiseaseState');
    const resultView = document.getElementById('diseaseResultView');
    if (emptyState) emptyState.style.display = 'block';
    if (resultView) resultView.style.display = 'none';
}

// Helper to format bytes into KB/MB
function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

// Initialize Leaf Advisory Events on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    const diseaseForm = document.getElementById('diseaseForm');
    const dropZone = document.getElementById('dropZone');
    const leafInput = document.getElementById('leafImageInput');

    if (!diseaseForm) return;

    // Input file change
    if (leafInput) {
        leafInput.addEventListener('change', function(e) {
            if (this.files && this.files[0]) {
                handleLeafFileSelection(this.files[0]);
            }
        });
    }

    // Drag & Drop events
    if (dropZone) {
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.add('dragover');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.remove('dragover');
            }, false);
        });

        dropZone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files && files.length > 0) {
                handleLeafFileSelection(files[0]);
            }
        }, false);
    }

    // Form submit via AJAX
    diseaseForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        hideError();

        if (!selectedLeafFile) {
            showError('Please upload or select a plant leaf image first.');
            return;
        }

        const analyzeBtn = document.getElementById('analyzeBtn');
        const spinner = document.getElementById('analyzeSpinner');
        const btnText = analyzeBtn ? analyzeBtn.querySelector('.btn-text') : null;

        if (analyzeBtn) analyzeBtn.disabled = true;
        if (spinner) spinner.style.display = 'inline-block';
        if (btnText) btnText.textContent = ' Analyzing Leaf...';

        try {
            const formData = new FormData();
            formData.append('leaf_image', selectedLeafFile);
            const cropTypeSelect = document.getElementById('cropTypeSelect');
            if (cropTypeSelect) {
                formData.append('crop_type', cropTypeSelect.value);
            }

            const response = await fetch('/api/analyze-disease', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (!response.ok || !result.success) {
                showError(result.error || 'Failed to process leaf image. Please check the file and try again.');
                return;
            }

            renderDiseaseAdvisoryResult(result.data);

        } catch (err) {
            console.error('Disease advisory error:', err);
            showError('Unable to connect to the server. Please ensure the Python server is running.');
        } finally {
            if (analyzeBtn) analyzeBtn.disabled = false;
            if (spinner) spinner.style.display = 'none';
            if (btnText) btnText.textContent = '🔬 Analyze Leaf';
        }
    });
});

// Render the disease advisory result view
function renderDiseaseAdvisoryResult(data) {
    const emptyState = document.getElementById('emptyDiseaseState');
    const resultView = document.getElementById('diseaseResultView');

    if (emptyState) emptyState.style.display = 'none';
    if (!resultView) return;

    const meta = data.image_info || {};

    // 1. If model is NOT available, show transparent status notification
    if (!data.model_available) {
        let guidelinesHtml = '';
        if (data.general_guidelines && data.general_guidelines.length > 0) {
            const items = data.general_guidelines.map(g => `<li>&bull; ${g}</li>`).join('');
            guidelinesHtml = `
                <div class="advisory-box">
                    <h4 class="advisory-heading">🌱 Recommended Immediate Best Practices</h4>
                    <ul class="library-steps-list">
                        ${items}
                    </ul>
                </div>
            `;
        }

        resultView.innerHTML = `
            <div class="result-header">
                <span class="result-eyebrow">Leaf Inspection Summary</span>
                <div class="crop-hero-badge">
                    <span class="crop-emoji">🍃</span>
                    <div class="crop-hero-info">
                        <h2 class="crop-name">Image Inspection Ready</h2>
                        <div class="confidence-pill confidence-pill-muted">
                            <span class="confidence-dot dot-muted"></span>
                            <span>Knowledgebase Mode Active</span>
                        </div>
                    </div>
                </div>
            </div>

            <div class="status-alert-box">
                <span class="status-alert-icon">⚠️</span>
                <div class="status-alert-content">
                    <h3 class="status-alert-title">${data.title}</h3>
                    <p class="status-alert-desc">${data.message}</p>
                </div>
            </div>

            <div class="meta-grid">
                <div class="meta-cell">
                    <span class="meta-label">Format</span>
                    <span class="meta-val">${meta.format || 'Image'}</span>
                </div>
                <div class="meta-cell">
                    <span class="meta-label">Dimensions</span>
                    <span class="meta-val">${meta.width && meta.height ? `${meta.width} × ${meta.height} px` : 'Valid'}</span>
                </div>
                <div class="meta-cell">
                    <span class="meta-label">File Size</span>
                    <span class="meta-val">${meta.size_kb ? `${meta.size_kb} KB` : 'Verified'}</span>
                </div>
            </div>

            <div class="result-summary-box">
                <p class="summary-text">${data.recommendation_summary || 'Please cross-reference your leaf visual symptoms with our verified disease reference library below.'}</p>
            </div>

            ${guidelinesHtml}

            <div style="margin-top: 20px;">
                <a href="#library" class="btn btn-secondary" style="width: 100%; justify-content: center; text-decoration: none;">
                    <span>📚 Browse Verified Disease Reference Library</span>
                    <span>&darr;</span>
                </a>
            </div>
        `;
        resultView.style.display = 'block';
        return;
    }

    // 2. GUARDRAIL 1: Explicit Unsupported Crop
    if (data.status === 'unsupported_crop') {
        let guidelinesHtml = '';
        if (data.general_guidelines && data.general_guidelines.length > 0) {
            const items = data.general_guidelines.map(g => `<li>&bull; ${g}</li>`).join('');
            guidelinesHtml = `
                <div class="advisory-box">
                    <h4 class="advisory-heading">📋 Recommended Safe Agronomic Protocol</h4>
                    <ul class="library-steps-list">
                        ${items}
                    </ul>
                </div>
            `;
        }

        let preventionHtml = '';
        if (data.prevention_guidelines) {
            preventionHtml = `
                <div class="advisory-box">
                    <h4 class="advisory-heading">🛡️ General Crop Sanitation & Management</h4>
                    <p class="info-text">${data.prevention_guidelines}</p>
                </div>
            `;
        }

        resultView.innerHTML = `
            <div class="result-header">
                <span class="result-eyebrow">Unsupported Crop Check</span>
                <div class="crop-hero-badge">
                    <span class="crop-emoji">🌾</span>
                    <div class="crop-hero-info">
                        <span class="badge-tag badge-warning">⚠️ Unsupported Crop</span>
                        <h2 class="crop-name" style="font-size: 1.4rem;">${data.prediction}</h2>
                        <div class="confidence-pill confidence-pill-muted" style="margin-top: 6px;">
                            <span class="confidence-dot dot-muted"></span>
                            <span>${data.confidence_label}</span>
                        </div>
                    </div>
                </div>
                <div class="model-spec-pill">
                    <span>🤖 ${data.model_name}</span>
                </div>
            </div>

            <div class="status-alert-box">
                <span class="status-alert-icon">🌾</span>
                <div class="status-alert-content">
                    <h3 class="status-alert-title">${data.selected_crop ? `${data.selected_crop} — AI Prediction Unavailable` : data.title}</h3>
                    <p class="status-alert-desc">${data.message}</p>
                </div>
            </div>

            <div class="meta-grid">
                <div class="meta-cell">
                    <span class="meta-label">Format</span>
                    <span class="meta-val">${meta.format || 'Image'}</span>
                </div>
                <div class="meta-cell">
                    <span class="meta-label">Dimensions</span>
                    <span class="meta-val">${meta.width && meta.height ? `${meta.width} × ${meta.height} px` : 'Valid'}</span>
                </div>
                <div class="meta-cell">
                    <span class="meta-label">File Size</span>
                    <span class="meta-val">${meta.size_kb ? `${meta.size_kb} KB` : 'Verified'}</span>
                </div>
            </div>

            <div class="result-summary-box">
                <p class="summary-text">${data.recommendation_summary}</p>
            </div>

            ${guidelinesHtml}
            ${preventionHtml}

            <div style="margin-top: 24px;">
                <a href="#library" class="btn btn-secondary" style="width: 100%; justify-content: center; text-decoration: none;">
                    <span>📚 Check Verified Disease Reference Library</span>
                    <span>&darr;</span>
                </a>
            </div>
        `;
        resultView.style.display = 'block';
        return;
    }

    // 3. GUARDRAIL 2: Low-Confidence / Uncertain Prediction
    if (data.status === 'low_confidence') {
        let guidelinesHtml = '';
        if (data.general_guidelines && data.general_guidelines.length > 0) {
            const items = data.general_guidelines.map(g => `<li>&bull; ${g}</li>`).join('');
            guidelinesHtml = `
                <div class="advisory-box">
                    <h4 class="advisory-heading">📋 Safe Diagnostic Protocol</h4>
                    <ul class="library-steps-list">
                        ${items}
                    </ul>
                </div>
            `;
        }

        let candidatesHtml = '';
        if (data.runners_up && data.runners_up.length > 0) {
            const chips = data.runners_up.map(alt => `
                <span class="alt-chip">${alt.name} (${alt.probability}%)</span>
            `).join('');

            candidatesHtml = `
                <div class="alternatives-box">
                    <span class="alternatives-title">Tentative Visual Classes (Unverified due to low probability):</span>
                    <div class="alternatives-chips">${chips}</div>
                </div>
            `;
        }

        resultView.innerHTML = `
            <div class="result-header">
                <span class="result-eyebrow">Preliminary Uncertainty Screening</span>
                <div class="crop-hero-badge">
                    <span class="crop-emoji">🔬</span>
                    <div class="crop-hero-info">
                        <span class="badge-tag badge-warning">⚠️ Low Confidence Warning</span>
                        <h2 class="crop-name" style="font-size: 1.4rem;">${data.prediction}</h2>
                        <div class="confidence-pill confidence-pill-warning" style="margin-top: 6px;">
                            <span class="confidence-dot dot-warning"></span>
                            <span>${data.confidence_label}</span>
                        </div>
                    </div>
                </div>
                <div class="model-spec-pill">
                    <span>🤖 ${data.model_name}</span>
                </div>
            </div>

            <div class="status-alert-box">
                <span class="status-alert-icon">⚠️</span>
                <div class="status-alert-content">
                    <h3 class="status-alert-title">${data.title}</h3>
                    <p class="status-alert-desc">${data.message}</p>
                </div>
            </div>

            <div class="meta-grid">
                <div class="meta-cell">
                    <span class="meta-label">Format</span>
                    <span class="meta-val">${meta.format || 'Image'}</span>
                </div>
                <div class="meta-cell">
                    <span class="meta-label">Dimensions</span>
                    <span class="meta-val">${meta.width && meta.height ? `${meta.width} × ${meta.height} px` : 'Valid'}</span>
                </div>
                <div class="meta-cell">
                    <span class="meta-label">File Size</span>
                    <span class="meta-val">${meta.size_kb ? `${meta.size_kb} KB` : 'Verified'}</span>
                </div>
            </div>

            <div class="result-summary-box">
                <p class="summary-text">${data.recommendation_summary}</p>
            </div>

            ${guidelinesHtml}
            ${candidatesHtml}

            <div style="margin-top: 24px;">
                <a href="#library" class="btn btn-secondary" style="width: 100%; justify-content: center; text-decoration: none;">
                    <span>📚 Check Verified Disease Reference Library</span>
                    <span>&darr;</span>
                </a>
            </div>
        `;
        resultView.style.display = 'block';
        return;
    }

    // 4. GUARDRAIL 3: Crop Mismatch Check
    if (data.status === 'crop_mismatch') {
        let guidelinesHtml = '';
        if (data.general_guidelines && data.general_guidelines.length > 0) {
            const items = data.general_guidelines.map(g => `<li>&bull; ${g}</li>`).join('');
            guidelinesHtml = `
                <div class="advisory-box">
                    <h4 class="advisory-heading">📋 Safe Diagnostic Protocol</h4>
                    <ul class="library-steps-list">
                        ${items}
                    </ul>
                </div>
            `;
        }

        resultView.innerHTML = `
            <div class="result-header">
                <span class="result-eyebrow">Crop Validation Notice</span>
                <div class="crop-hero-badge">
                    <span class="crop-emoji">⚠️</span>
                    <div class="crop-hero-info">
                        <span class="badge-tag badge-warning">⚠️ Crop Mismatch Detected</span>
                        <h2 class="crop-name" style="font-size: 1.4rem;">${data.prediction}</h2>
                        <div class="confidence-pill confidence-pill-warning" style="margin-top: 6px;">
                            <span class="confidence-dot dot-warning"></span>
                            <span>${data.confidence_label}</span>
                        </div>
                    </div>
                </div>
            </div>

            <div class="status-alert-box">
                <span class="status-alert-icon">⚠️</span>
                <div class="status-alert-content">
                    <h3 class="status-alert-title">${data.title}</h3>
                    <p class="status-alert-desc">${data.message}</p>
                </div>
            </div>

            <div class="meta-grid">
                <div class="meta-cell">
                    <span class="meta-label">Format</span>
                    <span class="meta-val">${meta.format || 'Image'}</span>
                </div>
                <div class="meta-cell">
                    <span class="meta-label">Dimensions</span>
                    <span class="meta-val">${meta.width && meta.height ? `${meta.width} × ${meta.height} px` : 'Valid'}</span>
                </div>
                <div class="meta-cell">
                    <span class="meta-label">File Size</span>
                    <span class="meta-val">${meta.size_kb ? `${meta.size_kb} KB` : 'Verified'}</span>
                </div>
            </div>

            <div class="result-summary-box">
                <p class="summary-text">${data.recommendation_summary}</p>
            </div>

            ${guidelinesHtml}
        `;
        resultView.style.display = 'block';
        return;
    }

    // 5. Successful Prediction (Supported Crop + High Confidence >= 40.0%)
    const isHealthy = data.is_healthy;
    const statusTagHtml = isHealthy
        ? `<span class="badge-tag badge-healthy">🌿 Healthy Foliage</span>`
        : `<span class="badge-tag badge-disease">⚠️ Plant Disease Detected</span>`;

    const confidenceDotClass = isHealthy ? 'dot-healthy' : 'dot-disease';

    let actionsHtml = '';
    if (data.immediate_actions && data.immediate_actions.length > 0) {
        const items = data.immediate_actions.map(a => `<li>&bull; ${a}</li>`).join('');
        actionsHtml = `
            <div class="advisory-box">
                <h4 class="advisory-heading">🚨 Immediate Field Actions</h4>
                <ul class="library-steps-list">
                    ${items}
                </ul>
            </div>
        `;
    }

    let treatmentsHtml = '';
    if (data.treatment_guidelines) {
        treatmentsHtml = `
            <div class="advisory-box">
                <h4 class="advisory-heading">💊 Recommended Treatment Protocols</h4>
                <p class="info-text">${data.treatment_guidelines}</p>
            </div>
        `;
    }

    let preventionHtml = '';
    if (data.prevention_guidelines) {
        preventionHtml = `
            <div class="advisory-box">
                <h4 class="advisory-heading">🛡️ Long-Term Prevention Strategy</h4>
                <p class="info-text">${data.prevention_guidelines}</p>
            </div>
        `;
    }

    let alternativesHtml = '';
    if (data.runners_up && data.runners_up.length > 0) {
        const chips = data.runners_up.map(alt => `
            <span class="alt-chip">${alt.name} (${alt.probability}%)</span>
        `).join('');

        alternativesHtml = `
            <div class="alternatives-box">
                <span class="alternatives-title">Alternative Diagnoses Considered:</span>
                <div class="alternatives-chips">${chips}</div>
            </div>
        `;
    }

    resultView.innerHTML = `
        <div class="result-header">
            <span class="result-eyebrow">Computer Vision Diagnosis</span>
            <div class="crop-hero-badge">
                <span class="crop-emoji">${isHealthy ? '🌿' : '🍃'}</span>
                <div class="crop-hero-info">
                    ${statusTagHtml}
                    <h2 class="crop-name" style="font-size: 1.55rem;">${data.prediction}</h2>
                    <div class="confidence-pill" style="margin-top: 6px;">
                        <span class="confidence-dot ${confidenceDotClass}"></span>
                        <span>${data.confidence_label}</span>
                    </div>
                </div>
            </div>
            <div class="model-spec-pill">
                <span>🤖 ${data.model_name} &bull; ${data.model_accuracy}</span>
            </div>
        </div>

        <!-- Educational Disclaimer Box -->
        <div class="confidence-disclaimer-box">
            <span class="disclaimer-icon">ℹ️</span>
            <div class="disclaimer-text">
                <strong>Important Agronomic Note:</strong> ${data.confidence_disclaimer}
            </div>
        </div>

        <div class="meta-grid">
            <div class="meta-cell">
                <span class="meta-label">Format</span>
                <span class="meta-val">${meta.format || 'Image'}</span>
            </div>
            <div class="meta-cell">
                <span class="meta-label">Dimensions</span>
                <span class="meta-val">${meta.width && meta.height ? `${meta.width} × ${meta.height} px` : 'Valid'}</span>
            </div>
            <div class="meta-cell">
                <span class="meta-label">File Size</span>
                <span class="meta-val">${meta.size_kb ? `${meta.size_kb} KB` : 'Verified'}</span>
            </div>
        </div>

        <div class="result-summary-box">
            <p class="summary-text">${data.recommendation_summary}</p>
        </div>

        ${actionsHtml}
        ${treatmentsHtml}
        ${preventionHtml}
        ${alternativesHtml}

        <div style="margin-top: 24px;">
            <a href="#library" class="btn btn-secondary" style="width: 100%; justify-content: center; text-decoration: none;">
                <span>📚 Explore Complete Disease Reference Library</span>
                <span>&darr;</span>
            </a>
        </div>
    `;

    resultView.style.display = 'block';

    if (window.innerWidth < 960) {
        resultView.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}


/* ==========================================================================
   Soil Health & Fertilizer Advisor Feature Logic (Module 3)
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    const fertilizerForm = document.getElementById('fertilizerForm');
    if (!fertilizerForm) return;

    fertilizerForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        hideError();

        const assessBtn = document.getElementById('assessBtn');
        const spinner = document.getElementById('assessSpinner');
        const btnText = assessBtn ? assessBtn.querySelector('.btn-text') : null;

        const ph = parseFloat(document.getElementById('phInput').value);
        const n = parseFloat(document.getElementById('nInput').value);
        const p = parseFloat(document.getElementById('pInput').value);
        const k = parseFloat(document.getElementById('kInput').value);
        const crop = document.getElementById('cropSelect').value;

        // Basic client-side validation
        if (isNaN(ph) || ph < 3.5 || ph > 9.5) {
            showError('Please enter a valid Soil pH between 3.5 and 9.5.');
            return;
        }
        if (isNaN(n) || n < 0 || n > 500) {
            showError('Please enter a valid Nitrogen (N) value between 0 and 500 kg/ha.');
            return;
        }
        if (isNaN(p) || p < 0 || p > 150) {
            showError('Please enter a valid Phosphorus (P) value between 0 and 150 kg/ha.');
            return;
        }
        if (isNaN(k) || k < 0 || k > 500) {
            showError('Please enter a valid Potassium (K) value between 0 and 500 kg/ha.');
            return;
        }

        if (assessBtn) assessBtn.disabled = true;
        if (spinner) spinner.style.display = 'inline-block';
        if (btnText) btnText.textContent = ' Assessing Soil...';

        try {
            const payload = { ph, n, p, k, crop };
            const response = await fetch('/api/analyze-soil', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            const result = await response.json();

            if (!response.ok || !result.success) {
                showError(result.error || 'Failed to evaluate soil nutrients. Please check your inputs.');
                return;
            }

            renderFertilizerResult(result.data);

        } catch (err) {
            console.error('Fertilizer advisor fetch error:', err);
            showError('Unable to connect to the server. Please ensure the Python server is running.');
        } finally {
            if (assessBtn) assessBtn.disabled = false;
            if (spinner) spinner.style.display = 'none';
            if (btnText) btnText.textContent = '🌾 Assess Soil & Advise Fertilizer';
        }
    });
});

// Render the soil health and fertilizer advisory result
function renderFertilizerResult(data) {
    const emptyState = document.getElementById('emptyFertilizerState');
    const resultView = document.getElementById('fertilizerResultView');

    if (emptyState) emptyState.style.display = 'none';
    if (!resultView) return;

    const asm = data.assessment || {};
    const gui = data.guidance || {};
    const inp = data.inputs || {};
    const cropDet = gui.crop_details || {};

    // 4R Stewardship HTML
    let stewardshipHtml = '';
    if (gui.stewardship_points && gui.stewardship_points.length > 0) {
        const items = gui.stewardship_points.map(s => `
            <div class="stewardship-item">
                <span class="stewardship-title">✓ ${s.title}</span>
                <span class="stewardship-desc">${s.desc}</span>
            </div>
        `).join('');
        stewardshipHtml = `
            <div class="advisory-box" style="margin-top: 18px;">
                <h4 class="advisory-heading">🎯 Principles of 4R Nutrient Stewardship</h4>
                <div class="stewardship-grid">${items}</div>
            </div>
        `;
    }

    // Organic & Biofertilizer HTML
    let organicHtml = '';
    if (gui.organic_practices && gui.organic_practices.length > 0) {
        const items = gui.organic_practices.map(o => `<li>&bull; ${o}</li>`).join('');
        organicHtml = `
            <div class="advisory-box" style="margin-top: 18px;">
                <h4 class="advisory-heading">🌱 Organic & Biofertilizer Integration</h4>
                <ul class="library-steps-list">${items}</ul>
            </div>
        `;
    }

    resultView.innerHTML = `
        <div class="result-header">
            <span class="result-eyebrow">Soil Fertility Assessment</span>
            <div class="crop-hero-badge">
                <span class="crop-emoji">${inp.crop_emoji || '🌾'}</span>
                <div class="crop-hero-info">
                    <span class="badge-tag badge-healthy">🧪 Expert Agronomic Engine</span>
                    <h2 class="crop-name" style="font-size: 1.45rem;">${inp.crop_name} Nutrition Plan</h2>
                    <div class="confidence-pill confidence-pill-muted" style="margin-top: 6px;">
                        <span class="confidence-dot dot-healthy"></span>
                        <span>ICAR & FAO Soil Fertility Benchmarks</span>
                    </div>
                </div>
            </div>
            <div class="model-spec-pill">
                <span>📘 Rule-Based Agronomic Knowledge Engine &bull; No Black-Box AI Claims</span>
            </div>
        </div>

        <!-- Official Soil Testing Lab Disclaimer Notice -->
        <div class="confidence-disclaimer-box" style="margin-bottom: 22px;">
            <span class="disclaimer-icon">⚠️</span>
            <div class="disclaimer-text">
                <strong>Important Agronomic Guidance Note:</strong> ${data.disclaimer}
            </div>
        </div>

        <!-- 4-Card Nutrient Assessment Grid -->
        <div class="nutrient-grid">
            
            <!-- Soil pH -->
            <div class="nutrient-card">
                <div class="nutrient-card-header">
                    <span class="nutrient-name">⚗️ Soil pH</span>
                    <span class="badge-tag badge-${asm.ph.badge}">${asm.ph.status}</span>
                </div>
                <div class="nutrient-value-row">
                    <span class="nutrient-big-val">${asm.ph.value}</span>
                    <span class="nutrient-unit">pH scale</span>
                </div>
                <span class="nutrient-benchmark">${asm.ph.benchmark}</span>
                <p class="nutrient-desc">${asm.ph.summary}</p>
            </div>

            <!-- Nitrogen (N) -->
            <div class="nutrient-card">
                <div class="nutrient-card-header">
                    <span class="nutrient-name">🌱 Nitrogen (N)</span>
                    <span class="badge-tag badge-${asm.nitrogen.badge}">${asm.nitrogen.status}</span>
                </div>
                <div class="nutrient-value-row">
                    <span class="nutrient-big-val">${asm.nitrogen.value}</span>
                    <span class="nutrient-unit">kg/ha</span>
                </div>
                <span class="nutrient-benchmark">${asm.nitrogen.benchmark}</span>
                <p class="nutrient-desc">${asm.nitrogen.description}</p>
            </div>

            <!-- Phosphorus (P) -->
            <div class="nutrient-card">
                <div class="nutrient-card-header">
                    <span class="nutrient-name">🌾 Phosphorus (P)</span>
                    <span class="badge-tag badge-${asm.phosphorus.badge}">${asm.phosphorus.status}</span>
                </div>
                <div class="nutrient-value-row">
                    <span class="nutrient-big-val">${asm.phosphorus.value}</span>
                    <span class="nutrient-unit">kg/ha P₂O₅</span>
                </div>
                <span class="nutrient-benchmark">${asm.phosphorus.benchmark}</span>
                <p class="nutrient-desc">${asm.phosphorus.description}</p>
            </div>

            <!-- Potassium (K) -->
            <div class="nutrient-card">
                <div class="nutrient-card-header">
                    <span class="nutrient-name">🪵 Potassium (K)</span>
                    <span class="badge-tag badge-${asm.potassium.badge}">${asm.potassium.status}</span>
                </div>
                <div class="nutrient-value-row">
                    <span class="nutrient-big-val">${asm.potassium.value}</span>
                    <span class="nutrient-unit">kg/ha K₂O</span>
                </div>
                <span class="nutrient-benchmark">${asm.potassium.benchmark}</span>
                <p class="nutrient-desc">${asm.potassium.description}</p>
            </div>

        </div>

        <!-- Cautious Fertilizer Management Strategy -->
        <div class="advisory-box">
            <h4 class="advisory-heading">💊 Scientific Fertilizer Management Strategy</h4>
            
            <div class="tip-item" style="margin-bottom: 12px;">
                <strong>1. Nitrogen (N) Strategy:</strong>
                <p class="info-text" style="margin-top: 4px;">${gui.nitrogen_guideline}</p>
            </div>

            <div class="tip-item" style="margin-bottom: 12px;">
                <strong>2. Phosphorus (P) Strategy:</strong>
                <p class="info-text" style="margin-top: 4px;">${gui.phosphorus_guideline}</p>
            </div>

            <div class="tip-item" style="margin-bottom: 12px;">
                <strong>3. Potassium (K) Strategy:</strong>
                <p class="info-text" style="margin-top: 4px;">${gui.potassium_guideline}</p>
            </div>

            <div class="tip-item">
                <strong>4. Soil Conditioning & pH Amendment:</strong>
                <p class="info-text" style="margin-top: 4px;">${gui.ph_guideline}</p>
            </div>
        </div>

        <!-- Crop-Specific Agronomic Context -->
        <div class="advisory-box" style="margin-top: 18px;">
            <h4 class="advisory-heading">🌾 Crop-Specific Agronomic Notes: ${cropDet.name || inp.crop_name}</h4>
            <div style="display: flex; gap: 10px; margin-bottom: 8px; flex-wrap: wrap;">
                <span class="alt-chip">🏷️ ${cropDet.category || 'Field Crop'}</span>
                <span class="alt-chip">⏳ Duration: ${cropDet.growth_duration || 'Standard season'}</span>
            </div>
            <p class="info-text">${gui.crop_summary}</p>
        </div>

        ${stewardshipHtml}
        ${organicHtml}

        <div style="margin-top: 24px;">
            <a href="#soil-guide" class="btn btn-secondary" style="width: 100%; justify-content: center; text-decoration: none;">
                <span>📚 View Standard ICAR Soil Rating Benchmarks</span>
                <span>&darr;</span>
            </a>
        </div>
    `;

    resultView.style.display = 'block';

    if (window.innerWidth < 960) {
        resultView.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

