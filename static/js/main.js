// CNN Vision Neural Explorer - Client Logic

document.addEventListener('DOMContentLoaded', () => {
    // State
    const state = {
        selectedModel: 'resnet50',
        sourceType: 'preset',
        sourceData: 'elephant',
        modelsMeta: {},
        currentTab: 'tab-predict',
        archModel: 'resnet50'
    };

    // DOM Elements
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const presetBtns = document.querySelectorAll('.preset-btn');
    const modelCards = document.querySelectorAll('.model-select-card');
    const imagePreview = document.getElementById('imagePreview');
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const customUrlInput = document.getElementById('customUrlInput');
    const loadUrlBtn = document.getElementById('loadUrlBtn');
    const runPredictBtn = document.getElementById('runPredictBtn');
    const predictSpinner = document.getElementById('predictSpinner');
    const predictBtnText = document.getElementById('predictBtnText');
    const liveStatus = document.getElementById('liveStatus');
    const latencyBadge = document.getElementById('latencyBadge');
    const latencyText = document.getElementById('latencyText');
    const singleResultsView = document.getElementById('singleResultsView');
    const comparisonResultsView = document.getElementById('comparisonResultsView');
    const comparisonGrid = document.getElementById('comparisonGrid');
    const topClassName = document.getElementById('topClassName');
    const topClassConfidence = document.getElementById('topClassConfidence');
    const topClassId = document.getElementById('topClassId');
    const probBarsContainer = document.getElementById('probBarsContainer');
    const previewResolution = document.getElementById('previewResolution');

    // Feature Extraction Elements
    const extractModelSelect = document.getElementById('extractModelSelect');
    const extractLayerSelect = document.getElementById('extractLayerSelect');
    const runExtractBtn = document.getElementById('runExtractBtn');
    const extractSpinner = document.getElementById('extractSpinner');
    const featDim = document.getElementById('featDim');
    const featMax = document.getElementById('featMax');
    const featMean = document.getElementById('featMean');
    const featSparsity = document.getElementById('featSparsity');
    const activationMatrix = document.getElementById('activationMatrix');
    const topNeuronsContainer = document.getElementById('topNeuronsContainer');
    const extractLatencyText = document.getElementById('extractLatencyText');

    // Architecture Explorer Elements
    const archPillBtns = document.querySelectorAll('.pill-btn');
    const archMetaContainer = document.getElementById('archMetaContainer');
    const layersTableBody = document.getElementById('layersTableBody');
    const layerSearchInput = document.getElementById('layerSearchInput');
    const layerCountText = document.getElementById('layerCountText');

    let cachedLayers = [];

    // Initialize Models Metadata
    async function initModelsMeta() {
        try {
            const res = await fetch('/api/models');
            const data = await res.json();
            if (data.success) {
                state.modelsMeta = data.models;
                updateArchMeta(state.archModel);
                loadModelLayers(state.archModel);
            }
        } catch (err) {
            console.error('Failed to load models metadata:', err);
        }
    }

    // Tab Navigation
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            document.getElementById(targetTab).classList.add('active');
            state.currentTab = targetTab;

            if (targetTab === 'tab-architecture' && cachedLayers.length === 0) {
                loadModelLayers(state.archModel);
            }
        });
    });

    // Preset Image Selection
    presetBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const preset = btn.getAttribute('data-preset');
            if (!preset) return;

            document.querySelectorAll('.preset-btn[data-preset]').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            state.sourceType = 'preset';
            state.sourceData = preset;
            imagePreview.src = `/api/sample-images/${preset}.jpg`;
            customUrlInput.value = '';

            // Update resolution tag
            if (state.selectedModel === 'inceptionv3') {
                previewResolution.innerText = '299 × 299 px';
            } else {
                previewResolution.innerText = '224 × 224 px';
            }
        });
    });

    // File Upload / Dropzone
    dropzone.addEventListener('click', () => fileInput.click());
    
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files && fileInput.files[0]) {
            handleFile(fileInput.files[0]);
        }
    });

    function handleFile(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const base64Data = e.target.result;
            imagePreview.src = base64Data;
            state.sourceType = 'base64';
            state.sourceData = base64Data;
            document.querySelectorAll('.preset-btn[data-preset]').forEach(b => b.classList.remove('active'));
            customUrlInput.value = '';
        };
        reader.readAsDataURL(file);
    }

    // URL Fetch
    loadUrlBtn.addEventListener('click', () => {
        const url = customUrlInput.value.trim();
        if (!url) return;
        state.sourceType = 'url';
        state.sourceData = url;
        imagePreview.src = url;
        document.querySelectorAll('.preset-btn[data-preset]').forEach(b => b.classList.remove('active'));
    });

    // Model Selection
    modelCards.forEach(card => {
        card.addEventListener('click', () => {
            modelCards.forEach(c => c.classList.remove('active'));
            card.classList.add('active');
            state.selectedModel = card.getAttribute('data-model');

            // Update target dimension indicator
            if (state.selectedModel === 'inceptionv3') {
                previewResolution.innerText = '299 × 299 px';
            } else if (state.selectedModel === 'all') {
                previewResolution.innerText = 'Multi-Resolution';
            } else {
                previewResolution.innerText = '224 × 224 px';
            }
        });
    });

    // Execute Inference
    runPredictBtn.addEventListener('click', runPrediction);

    async function runPrediction() {
        setPredictLoading(true);
        liveStatus.innerText = 'Inferring...';
        liveStatus.className = 'status-indicator';

        try {
            const payload = {
                model: state.selectedModel,
                source_type: state.sourceType,
                source_data: state.sourceData,
                top_k: 5
            };

            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await res.json();
            if (!data.success) {
                alert('Prediction Error: ' + (data.error || 'Unknown error'));
                return;
            }

            if (data.mode === 'comparison') {
                renderComparisonResults(data.results);
            } else {
                renderSingleResults(data);
            }

            liveStatus.innerText = 'Live Consensus';
            liveStatus.className = 'status-indicator live';

        } catch (err) {
            console.error('Prediction failed:', err);
            alert('Failed to connect to backend: ' + err.message);
        } finally {
            setPredictLoading(false);
        }
    }

    function renderSingleResults(data) {
        singleResultsView.classList.remove('hidden');
        comparisonResultsView.classList.add('hidden');

        latencyText.innerText = `${data.latency_ms} ms (${data.model_name})`;

        const top = data.predictions[0];
        topClassName.innerText = top.label;
        topClassConfidence.innerText = `${top.percentage}%`;
        topClassId.innerText = `ILSVRC Class ID: ${top.class_id}`;

        // Populate Top 5 Bars
        probBarsContainer.innerHTML = '';
        data.predictions.forEach((pred, idx) => {
            const row = document.createElement('div');
            row.className = 'prob-row';
            row.innerHTML = `
                <div class="prob-meta">
                    <div>
                        <span class="prob-label">${idx + 1}. ${pred.label}</span>
                        <span class="prob-id">${pred.class_id}</span>
                    </div>
                    <span class="prob-percentage">${pred.percentage}%</span>
                </div>
                <div class="prob-bar-track">
                    <div class="prob-bar-fill" style="width: 0%"></div>
                </div>
            `;
            probBarsContainer.appendChild(row);

            // Trigger animation
            setTimeout(() => {
                row.querySelector('.prob-bar-fill').style.width = `${Math.max(pred.percentage, 1)}%`;
            }, 50 * idx);
        });
    }

    function renderComparisonResults(results) {
        singleResultsView.classList.add('hidden');
        comparisonResultsView.classList.remove('hidden');

        let totalLatency = 0;
        comparisonGrid.innerHTML = '';

        Object.keys(results).forEach(mid => {
            const item = results[mid];
            totalLatency += item.latency_ms;
            const top = item.predictions[0];
            const cfg = item.config;

            const card = document.createElement('div');
            card.className = 'comp-card';
            card.innerHTML = `
                <div class="comp-card-header">
                    <span class="comp-model-name">${cfg.name}</span>
                    <span class="comp-latency">${item.latency_ms} ms</span>
                </div>
                <div class="comp-top-prediction">
                    <span class="comp-top-label">#1 ${top.label}</span>
                    <span class="comp-top-score">${top.percentage}%</span>
                </div>
                <div class="probability-bars" style="gap: 8px;">
                    ${item.predictions.slice(1, 3).map((p, i) => `
                        <div class="prob-meta" style="font-size: 0.8rem;">
                            <span style="color: #94a3b8;">#${i + 2} ${p.label}</span>
                            <span style="font-family: monospace;">${p.percentage}%</span>
                        </div>
                    `).join('')}
                </div>
            `;
            comparisonGrid.appendChild(card);
        });

        latencyText.innerText = `All 4 Models: ${Math.round(totalLatency)} ms total`;
    }

    function setPredictLoading(isLoading) {
        if (isLoading) {
            predictSpinner.classList.remove('hidden');
            predictBtnText.innerText = 'Calculating Activations...';
            runPredictBtn.disabled = true;
        } else {
            predictSpinner.classList.add('hidden');
            predictBtnText.innerText = 'Run Neural Inference';
            runPredictBtn.disabled = false;
        }
    }

    // ==========================================
    // Feature Extraction Tab Logic
    // ==========================================
    const layerOptionsMap = {
        'inceptionv3': [
            { id: 'avg_pool', label: 'avg_pool (GlobalAveragePooling2D - 2048 dims)' },
            { id: 'mixed10', label: 'mixed10 (Final Inception Convolutions)' },
            { id: 'mixed9', label: 'mixed9 (Intermediate Inception Block)' }
        ],
        'vgg19': [
            { id: 'fc1', label: 'fc1 (Dense 4096 dimensions)' },
            { id: 'fc2', label: 'fc2 (Dense 4096 dimensions)' },
            { id: 'block5_pool', label: 'block5_pool (Last MaxPooling2D)' }
        ],
        'resnet50': [
            { id: 'avg_pool', label: 'avg_pool (GlobalAveragePooling2D - 2048 dims)' },
            { id: 'conv5_block3_out', label: 'conv5_block3_out (Last Residual Block)' }
        ]
    };

    extractModelSelect.addEventListener('change', () => {
        const m = extractModelSelect.value;
        const options = layerOptionsMap[m] || [];
        extractLayerSelect.innerHTML = options.map(opt => `<option value="${opt.id}">${opt.label}</option>`).join('');
    });

    let extractPreset = 'sharpener';
    document.querySelectorAll('.preset-btn[data-extract-preset]').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.preset-btn[data-extract-preset]').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            extractPreset = btn.getAttribute('data-extract-preset');
        });
    });

    runExtractBtn.addEventListener('click', runFeatureExtraction);

    async function runFeatureExtraction() {
        extractSpinner.classList.remove('hidden');
        runExtractBtn.disabled = true;

        try {
            const payload = {
                model: extractModelSelect.value,
                layer: extractLayerSelect.value,
                source_type: 'preset',
                source_data: extractPreset
            };

            const res = await fetch('/api/extract', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await res.json();
            if (!data.success) {
                alert('Extraction Error: ' + data.error);
                return;
            }

            // Update stats
            featDim.innerText = Number(data.total_dimensions).toLocaleString();
            featMax.innerText = data.stats.max;
            featMean.innerText = data.stats.mean;
            featSparsity.innerText = `${data.stats.sparsity_pct}%`;
            extractLatencyText.innerText = `${data.latency_ms} ms (${data.layer_name})`;

            // Render Heatmap Matrix (64 preview values)
            activationMatrix.innerHTML = '';
            const maxVal = Math.max(data.stats.max, 0.001);

            data.preview_values.forEach((val, i) => {
                const cell = document.createElement('div');
                cell.className = 'act-cell';
                const intensity = Math.min(Math.max(val / maxVal, 0), 1);
                
                // Color scaling from dark blue to bright cyan/yellow
                const r = Math.round(6 + intensity * 240);
                const g = Math.round(182 + intensity * 70);
                const b = Math.round(212 - intensity * 100);
                cell.style.backgroundColor = `rgba(${r}, ${g}, ${b}, ${0.15 + intensity * 0.85})`;
                cell.title = `Neuron #${i}: ${val}`;
                activationMatrix.appendChild(cell);
            });

            // Render Top Firing Neurons
            topNeuronsContainer.innerHTML = '';
            data.top_activations.forEach(item => {
                const chip = document.createElement('div');
                chip.className = 'neuron-chip';
                chip.innerHTML = `
                    <span class="neuron-idx">#${item.index}</span>
                    <span class="neuron-val">${item.value}</span>
                `;
                topNeuronsContainer.appendChild(chip);
            });

        } catch (err) {
            console.error('Feature extraction failed:', err);
            alert('Feature extraction failed: ' + err.message);
        } finally {
            extractSpinner.classList.add('hidden');
            runExtractBtn.disabled = false;
        }
    }

    // ==========================================
    // Architecture Explorer Logic
    // ==========================================
    archPillBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            archPillBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const modelId = btn.getAttribute('data-arch');
            state.archModel = modelId;
            updateArchMeta(modelId);
            loadModelLayers(modelId);
        });
    });

    function updateArchMeta(modelId) {
        const meta = state.modelsMeta[modelId];
        if (!meta) return;

        archMetaContainer.innerHTML = `
            <div class="arch-meta-card">
                <div class="meta-card-label">Total Parameters</div>
                <div class="meta-card-val">${meta.params}</div>
            </div>
            <div class="arch-meta-card">
                <div class="meta-card-label">Network Depth</div>
                <div class="meta-card-val">${meta.depth}</div>
            </div>
            <div class="arch-meta-card">
                <div class="meta-card-label">Input Tensor Resolution</div>
                <div class="meta-card-val">${meta.target_size[0]} × ${meta.target_size[1]} × 3</div>
            </div>
            <div class="arch-meta-card">
                <div class="meta-card-label">Architecture Focus</div>
                <div class="meta-card-val" style="font-size: 0.95rem; font-weight: 500;">${meta.specialty}</div>
            </div>
        `;
    }

    async function loadModelLayers(modelId) {
        layersTableBody.innerHTML = `<tr><td colspan="5" style="text-align: center; padding: 24px; color: #94a3b8;">Loading layer graph...</td></tr>`;
        try {
            const res = await fetch(`/api/model-layers/${modelId}`);
            const data = await res.json();
            if (data.success) {
                cachedLayers = data.layers;
                renderLayersTable(cachedLayers);
            }
        } catch (err) {
            console.error('Failed to load layers:', err);
            layersTableBody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: #f43f5e;">Failed to load layer details.</td></tr>`;
        }
    }

    function renderLayersTable(layers) {
        layerCountText.innerText = `Showing ${layers.length} layers`;
        layersTableBody.innerHTML = '';
        layers.forEach(l => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td style="color: #64748b; font-family: monospace;">${l.index}</td>
                <td style="font-weight: 600;">${l.name}</td>
                <td><span class="layer-tag">${l.class_name}</span></td>
                <td><span class="shape-code">${l.output_shape}</span></td>
                <td style="font-family: monospace;">${l.params.toLocaleString()}</td>
            `;
            layersTableBody.appendChild(tr);
        });
    }

    layerSearchInput.addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase().trim();
        const filtered = cachedLayers.filter(l => 
            l.name.toLowerCase().includes(query) || 
            l.class_name.toLowerCase().includes(query)
        );
        renderLayersTable(filtered);
    });

    // Auto-initialize
    initModelsMeta();
    runPrediction(); // Initial prediction with Elephant preset
});
