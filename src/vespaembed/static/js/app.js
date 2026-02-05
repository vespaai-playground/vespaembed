// VespaEmbed Web UI

// State
let runs = [];
let activeRunId = null;
let selectedRunId = null;
let chart = null;
let lossHistory = [];
let pollInterval = null;
let pollLine = 0;
let currentDataSource = 'file'; // 'file' or 'huggingface'
let tasksData = []; // Cached task data from API
let metricsData = {}; // Cached metrics from TensorBoard
let currentMetric = 'loss'; // Currently selected metric

// DOM Elements
const newTrainingBtn = document.getElementById('new-training-btn');
const newTrainingModal = document.getElementById('new-training-modal');
const closeNewTraining = document.getElementById('close-new-training');
const trainForm = document.getElementById('train-form');
const runList = document.getElementById('run-list');
const projectSummary = document.getElementById('project-summary');
const logContent = document.getElementById('log-content');
const chartPlaceholder = document.getElementById('chart-placeholder');
const artifactsBtn = document.getElementById('artifacts-btn');
const artifactsModal = document.getElementById('artifacts-modal');
const closeArtifactsModal = document.getElementById('close-artifacts-modal');

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    initChart();
    await loadTasks(); // Load tasks from API first
    await loadRuns(true); // Auto-select latest run on initial load
    setupEventListeners();
    setupFileUploads();
    setupAutoSplit();
    setupTabs();
    setupHubToggle();
    setupLoraToggle();
    setupUnslothToggle();
    setupMatryoshkaToggle();
    setupTaskSelector();
    setupWizard();
    setupArtifactsModal();
});

// Generate random project name
function generateProjectName() {
    const chars = 'abcdefghijklmnopqrstuvwxyz0123456789';
    let name = '';
    for (let i = 0; i < 8; i++) {
        name += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return name;
}

// Load tasks from API
async function loadTasks() {
    try {
        const response = await fetch('/api/tasks');
        tasksData = await response.json();
        populateTaskDropdown();
    } catch (error) {
        console.error('Failed to load tasks:', error);
    }
}

// Populate task dropdown from API data
function populateTaskDropdown() {
    const taskSelect = document.getElementById('task');
    taskSelect.innerHTML = tasksData.map(task =>
        `<option value="${task.name}">${task.name.toUpperCase()}</option>`
    ).join('');

    // Set initial task description and defaults
    if (tasksData.length > 0) {
        updateTaskUI(tasksData[0]);
    }
}

// Setup task selector change handler
function setupTaskSelector() {
    const taskSelect = document.getElementById('task');
    taskSelect.addEventListener('change', () => {
        const task = tasksData.find(t => t.name === taskSelect.value);
        if (task) {
            updateTaskUI(task);
        }
    });
}

// Update UI based on selected task
function updateTaskUI(task) {
    // Update task description
    const descEl = document.getElementById('task-description');
    if (descEl) {
        descEl.textContent = task.description;
    }

    // Update required columns display
    const columnsEl = document.getElementById('required-columns-list');
    if (columnsEl && task.expected_columns) {
        columnsEl.innerHTML = task.expected_columns.map(col =>
            `<span class="required-column-tag">${col}</span>`
        ).join('');
    }

    // Update loss variant dropdown
    updateLossVariantUI(task);

    // Update sample data display
    updateSampleData(task);

    // Update Matryoshka section visibility (not supported for TSDAE)
    const matryoshkaSection = document.getElementById('matryoshka-section');
    if (matryoshkaSection) {
        if (task.name === 'tsdae') {
            matryoshkaSection.style.display = 'none';
            // Also uncheck and hide fields when switching to TSDAE
            document.getElementById('matryoshka_enabled').checked = false;
            document.getElementById('matryoshka-fields').style.display = 'none';
        } else {
            matryoshkaSection.style.display = 'block';
        }
    }

    // Update hyperparameters to task defaults
    const hyper = task.hyperparameters;
    if (hyper) {
        setValueIfExists('epochs', hyper.epochs);
        setValueIfExists('batch_size', hyper.batch_size);
        setValueIfExists('learning_rate', hyper.learning_rate);
        setValueIfExists('warmup_ratio', hyper.warmup_ratio);
        setValueIfExists('weight_decay', hyper.weight_decay);
        setValueIfExists('eval_steps', hyper.eval_steps);
        setValueIfExists('save_steps', hyper.save_steps);
        setValueIfExists('logging_steps', hyper.logging_steps);
        setValueIfExists('gradient_accumulation_steps', hyper.gradient_accumulation_steps);
        setValueIfExists('optimizer', hyper.optimizer || 'adamw_torch');
        setValueIfExists('scheduler', hyper.scheduler || 'linear');

        // Set precision dropdown
        if (hyper.bf16) {
            setValueIfExists('precision', 'bf16');
        } else if (hyper.fp16) {
            setValueIfExists('precision', 'fp16');
        } else {
            setValueIfExists('precision', 'fp32');
        }
    }

    // Update task-specific parameters
    const paramsContainer = document.getElementById('task-specific-params');
    paramsContainer.innerHTML = '';

    if (task.task_specific_params && Object.keys(task.task_specific_params).length > 0) {
        const fieldsHtml = Object.entries(task.task_specific_params).map(([key, config]) => {
            return `
                <div class="form-field">
                    <label>${config.label}</label>
                    <input type="${config.type}" id="task_param_${key}" name="task_param_${key}"
                           value="${config.default || ''}">
                    ${config.description ? `<span class="field-hint">${config.description}</span>` : ''}
                </div>
            `;
        }).join('');

        paramsContainer.innerHTML = `
            <div class="form-section">
                <label class="section-label">Task Settings</label>
                ${fieldsHtml}
            </div>
        `;
    }
}

// Update sample data display for selected task
function updateSampleData(task) {
    const container = document.getElementById('sample-data-content');
    if (!container || !task.sample_data || task.sample_data.length === 0) {
        return;
    }

    // Show just the first sample row as a formatted example
    const sample = task.sample_data[0];
    const columns = task.expected_columns;

    const rowsHtml = columns.map(col => {
        const value = sample[col];
        const displayValue = typeof value === 'string'
            ? (value.length > 50 ? value.substring(0, 50) + '...' : value)
            : value;
        return `<span class="sample-data-row"><span class="sample-data-key">${col}:</span> <span class="sample-data-value">"${displayValue}"</span></span>`;
    }).join('');

    container.innerHTML = rowsHtml;
}

// Update loss variant dropdown based on selected task
function updateLossVariantUI(task) {
    const fieldEl = document.getElementById('loss-variant-field');
    const selectEl = document.getElementById('loss_variant');
    const hintEl = document.getElementById('loss-variant-hint');

    if (!fieldEl || !selectEl) return;

    // Check if task has loss options
    if (task.loss_options && task.loss_options.length > 0) {
        // Populate dropdown with multiple options
        selectEl.innerHTML = task.loss_options.map(loss => {
            const isDefault = loss === task.default_loss;
            const label = formatLossLabel(loss) + (isDefault ? ' (default)' : '');
            return `<option value="${loss}" ${isDefault ? 'selected' : ''}>${label}</option>`;
        }).join('');
        selectEl.disabled = false;

        // Update hint based on task type
        if (hintEl) {
            if (task.name === 'pairs' || task.name === 'triplets' || task.name === 'matryoshka') {
                hintEl.textContent = 'MNR uses in-batch negatives. GIST filters false negatives using the model itself as guide.';
            } else if (task.name === 'similarity') {
                hintEl.textContent = 'CoSENT and AnglE often outperform Cosine on STS benchmarks.';
            } else {
                hintEl.textContent = '';
            }
        }

        fieldEl.style.display = 'block';
    } else if (task.name === 'tsdae') {
        // TSDAE has a fixed loss - show it but disabled
        selectEl.innerHTML = '<option value="">Denoising Auto-Encoder Loss</option>';
        selectEl.disabled = true;

        if (hintEl) {
            hintEl.textContent = 'TSDAE uses unsupervised denoising to learn embeddings from unlabeled text.';
        }

        fieldEl.style.display = 'block';
    } else {
        // Hide for other tasks without loss options
        fieldEl.style.display = 'none';
    }
}

// Format loss variant name for display
function formatLossLabel(loss) {
    const labels = {
        'mnr': 'MNR (Multiple Negatives Ranking)',
        'mnr_symmetric': 'MNR Symmetric',
        'gist': 'GIST (Guided In-Sample Triplet)',
        'cached_mnr': 'Cached MNR',
        'cached_gist': 'Cached GIST',
        'cosine': 'Cosine Similarity',
        'cosent': 'CoSENT',
        'angle': 'AnglE',
    };
    return labels[loss] || loss.toUpperCase();
}

// Helper to set input value if element exists
function setValueIfExists(id, value) {
    const el = document.getElementById(id);
    if (el && value !== undefined) {
        el.value = value;
    }
}

// Helper to set radio button checked
function setRadioChecked(name, value) {
    const radio = document.querySelector(`input[name="${name}"][value="${value}"]`);
    if (radio) {
        radio.checked = true;
    }
}

// Get task-specific parameter values
function getTaskSpecificParams() {
    const params = {};
    const task = tasksData.find(t => t.name === document.getElementById('task').value);
    if (task && task.task_specific_params) {
        for (const key of Object.keys(task.task_specific_params)) {
            const el = document.getElementById(`task_param_${key}`);
            if (el && el.value) {
                params[key] = el.value;
            }
        }
    }
    return params;
}

// Polling for updates
let pollingRunId = null; // The run ID we're currently polling for

function startPolling(runId) {
    stopPolling();
    // pollLine is set by loadHistoricalUpdates, so don't reset it here
    pollingRunId = runId;

    pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`/runs/${pollingRunId}/updates?since_line=${pollLine}`);
            if (!response.ok) {
                stopPolling();
                return;
            }

            const data = await response.json();

            // Only update UI if we're viewing the run being polled
            if (selectedRunId === pollingRunId) {
                data.updates.forEach(update => {
                    handleUpdate(update);
                });

                // Refresh metrics from TensorBoard files
                await loadMetrics(pollingRunId);
            }

            // Update poll position
            pollLine = data.next_line;

            // Stop polling if run is no longer active
            if (!data.has_more) {
                stopPolling();
                loadRuns(); // Refresh run list to show final status
                // Final metrics refresh if still viewing this run
                if (selectedRunId === pollingRunId) {
                    await loadMetrics(pollingRunId);
                }
            }
        } catch (error) {
            console.error('Polling error:', error);
        }
    }, 2000); // Poll every 2 seconds
}

function stopPolling() {
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }
    pollingRunId = null;
}

function handleUpdate(data) {
    switch (data.type) {
        case 'log':
            appendLog(data.message);
            break;
        case 'progress':
            updateProgress(data);
            break;
        case 'status':
            updateRunStatus(data.run_id, data.status);
            break;
        case 'complete':
            handleTrainingComplete(data);
            break;
        case 'error':
            handleTrainingError(data);
            break;
    }
}

// Chart
function initChart() {
    const ctx = document.getElementById('loss-chart').getContext('2d');
    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Loss',
                data: [],
                borderColor: '#22c55e',
                backgroundColor: 'rgba(34, 197, 94, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.3,
                pointRadius: 3,
                pointBackgroundColor: '#22c55e',
                pointBorderColor: '#22c55e',
                pointHoverRadius: 6,
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: '#22c55e',
                pointHoverBorderWidth: 2,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'nearest',
                axis: 'x',
                intersect: false,
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: '#22c55e',
                    borderWidth: 1,
                    displayColors: false,
                    callbacks: {
                        title: (items) => `Step ${items[0].label}`,
                        label: (item) => `${item.dataset.label}: ${item.parsed.y.toFixed(4)}`,
                    }
                }
            },
            scales: {
                x: {
                    display: true,
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#999' }
                },
                y: {
                    display: true,
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#999' }
                }
            }
        }
    });
}

function updateChart(step, loss) {
    lossHistory.push({ step, loss });
    chart.data.labels.push(step);
    chart.data.datasets[0].data.push(loss);
    chart.update('none');
    chartPlaceholder.style.display = 'none';
}

function resetChart() {
    lossHistory = [];
    chart.data.labels = [];
    chart.data.datasets[0].data = [];
    chart.update();
    chartPlaceholder.style.display = 'flex';
}

// Reset form fields to task defaults
function resetFormToTaskDefaults() {
    // Reset file uploads
    document.getElementById('train_filename').value = '';
    document.getElementById('eval_filename').value = '';
    document.getElementById('train_file_info').textContent = 'CSV or JSONL';
    document.getElementById('eval_file_info').textContent = 'CSV or JSONL';
    document.getElementById('train-upload').classList.remove('uploaded');
    document.getElementById('eval-upload').classList.remove('uploaded');

    // Reset HF dataset fields
    document.getElementById('hf_dataset').value = '';
    document.getElementById('hf_subset').value = '';
    document.getElementById('hf_train_split').value = 'train';
    document.getElementById('hf_eval_split').value = '';

    // Reset hub settings
    document.getElementById('push_to_hub').checked = false;
    document.getElementById('hf_username').value = '';
    document.getElementById('hub-fields').style.display = 'none';

    // Reset LoRA settings
    document.getElementById('lora_enabled').checked = false;
    document.getElementById('lora_r').value = '64';
    document.getElementById('lora_alpha').value = '128';
    document.getElementById('lora_dropout').value = '0.1';
    document.getElementById('lora_target_preset').value = 'query, key, value, dense';
    document.getElementById('lora_target_modules').value = 'query, key, value, dense';
    document.getElementById('lora-fields').style.display = 'none';

    // Reset model settings
    document.getElementById('max_seq_length').value = '';  // Empty = auto-detect
    document.getElementById('gradient_checkpointing').checked = false;

    // Reset Unsloth settings
    document.getElementById('unsloth_enabled').checked = false;
    document.getElementById('unsloth_save_method').value = 'merged_16bit';
    document.getElementById('unsloth-fields').style.display = 'none';

    // Reset Matryoshka settings
    document.getElementById('matryoshka_enabled').checked = false;
    document.getElementById('matryoshka_dims').value = '768,512,256,128,64';
    document.getElementById('matryoshka-fields').style.display = 'none';

    // Apply defaults for currently selected task (must be last to properly show/hide UI elements)
    const taskSelect = document.getElementById('task');
    const task = tasksData.find(t => t.name === taskSelect.value);
    if (task) {
        updateTaskUI(task);
    }
}

// Event Listeners
function setupEventListeners() {
    // New Training Modal
    newTrainingBtn.addEventListener('click', () => {
        // Generate random project name when opening modal
        document.getElementById('project_name').value = generateProjectName();

        // Reset form and apply defaults for currently selected task
        resetFormToTaskDefaults();

        // Reset wizard to step 1
        resetWizard();

        newTrainingModal.style.display = 'flex';
    });

    closeNewTraining.addEventListener('click', () => {
        newTrainingModal.style.display = 'none';
    });

    newTrainingModal.addEventListener('click', (e) => {
        if (e.target === newTrainingModal) {
            newTrainingModal.style.display = 'none';
        }
    });

    // Form Submit
    trainForm.addEventListener('submit', handleTrainSubmit);

    // Refresh
    document.getElementById('refresh-btn').addEventListener('click', loadRuns);

    // Stop button
    document.getElementById('stop-btn').addEventListener('click', stopTraining);

    // Delete button
    document.getElementById('delete-btn').addEventListener('click', deleteRun);

    // Escape key closes modals
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            newTrainingModal.style.display = 'none';
        }
    });

    // Metric selector
    document.getElementById('metric-select').addEventListener('change', (e) => {
        currentMetric = e.target.value;
        updateChartWithMetric(currentMetric);
    });
}

// Tabs
function setupTabs() {
    const tabs = document.querySelectorAll('.tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Update active tab
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            // Update active content
            const tabId = tab.dataset.tab;
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            document.getElementById(`tab-${tabId}`).classList.add('active');

            // Update data source state
            currentDataSource = tabId;
        });
    });
}

// Hub Push Toggle
function setupHubToggle() {
    const pushToHub = document.getElementById('push_to_hub');
    const hubFields = document.getElementById('hub-fields');

    pushToHub.addEventListener('change', () => {
        hubFields.style.display = pushToHub.checked ? 'block' : 'none';
    });
}

// LoRA Toggle
function setupLoraToggle() {
    const loraEnabled = document.getElementById('lora_enabled');
    const loraFields = document.getElementById('lora-fields');

    loraEnabled.addEventListener('change', () => {
        loraFields.style.display = loraEnabled.checked ? 'block' : 'none';
    });

    // Target modules preset select
    const presetSelect = document.getElementById('lora_target_preset');
    const targetInput = document.getElementById('lora_target_modules');

    presetSelect.addEventListener('change', () => {
        targetInput.value = presetSelect.value;
    });
}

// Unsloth Toggle
function setupUnslothToggle() {
    const unslothEnabled = document.getElementById('unsloth_enabled');
    const unslothFields = document.getElementById('unsloth-fields');

    unslothEnabled.addEventListener('change', () => {
        unslothFields.style.display = unslothEnabled.checked ? 'block' : 'none';
    });
}

// Matryoshka Toggle
function setupMatryoshkaToggle() {
    const matryoshkaEnabled = document.getElementById('matryoshka_enabled');
    const matryoshkaFields = document.getElementById('matryoshka-fields');

    matryoshkaEnabled.addEventListener('change', () => {
        matryoshkaFields.style.display = matryoshkaEnabled.checked ? 'block' : 'none';
    });
}

// Wizard Navigation
let currentWizardStep = 1;
const totalWizardSteps = 3;

function setupWizard() {
    const nextBtn = document.getElementById('wizard-next');
    const backBtn = document.getElementById('wizard-back');
    const startBtn = document.getElementById('start-btn');

    nextBtn.addEventListener('click', () => {
        if (validateCurrentStep()) {
            goToStep(currentWizardStep + 1);
        }
    });

    backBtn.addEventListener('click', () => {
        goToStep(currentWizardStep - 1);
    });
}

function goToStep(step) {
    if (step < 1 || step > totalWizardSteps) return;

    // Hide current step
    document.getElementById(`wizard-step-${currentWizardStep}`).style.display = 'none';
    document.querySelector(`.wizard-step[data-step="${currentWizardStep}"]`).classList.remove('active');

    // Mark previous steps as completed
    if (step > currentWizardStep) {
        document.querySelector(`.wizard-step[data-step="${currentWizardStep}"]`).classList.add('completed');
    }

    // Show new step
    currentWizardStep = step;
    document.getElementById(`wizard-step-${currentWizardStep}`).style.display = 'block';
    document.querySelector(`.wizard-step[data-step="${currentWizardStep}"]`).classList.add('active');
    document.querySelector(`.wizard-step[data-step="${currentWizardStep}"]`).classList.remove('completed');

    // Update button visibility
    const backBtn = document.getElementById('wizard-back');
    const nextBtn = document.getElementById('wizard-next');
    const startBtn = document.getElementById('start-btn');

    backBtn.style.display = currentWizardStep > 1 ? 'block' : 'none';
    nextBtn.style.display = currentWizardStep < totalWizardSteps ? 'block' : 'none';
    startBtn.style.display = currentWizardStep === totalWizardSteps ? 'block' : 'none';
}

function validateCurrentStep() {
    if (currentWizardStep === 1) {
        // Validate step 1: project name, task, model, data
        const projectName = document.getElementById('project_name').value.trim();
        if (!projectName) {
            alert('Please enter a project name');
            return false;
        }
        if (!/^[a-zA-Z0-9][a-zA-Z0-9-]*$/.test(projectName)) {
            alert('Project name must start with alphanumeric and contain only alphanumeric characters and hyphens');
            return false;
        }

        // Validate data source
        if (currentDataSource === 'file') {
            const trainFile = document.getElementById('train_filename').value;
            if (!trainFile) {
                alert('Please upload training data');
                return false;
            }
        } else {
            const hfDataset = document.getElementById('hf_dataset').value.trim();
            if (!hfDataset) {
                alert('Please enter a HuggingFace dataset name');
                return false;
            }
        }
    }
    return true;
}

function resetWizard() {
    currentWizardStep = 1;

    // Reset step indicators
    document.querySelectorAll('.wizard-step').forEach((step, index) => {
        step.classList.remove('active', 'completed');
        if (index === 0) step.classList.add('active');
    });

    // Show only first step
    for (let i = 1; i <= totalWizardSteps; i++) {
        document.getElementById(`wizard-step-${i}`).style.display = i === 1 ? 'block' : 'none';
    }

    // Reset buttons
    document.getElementById('wizard-back').style.display = 'none';
    document.getElementById('wizard-next').style.display = 'block';
    document.getElementById('start-btn').style.display = 'none';
}

// File Uploads
function setupFileUploads() {
    // Training data upload
    setupSingleUpload({
        uploadBox: document.getElementById('train-upload'),
        fileInput: document.getElementById('train_file'),
        fileInfo: document.getElementById('train_file_info'),
        hiddenInput: document.getElementById('train_filename'),
        fileType: 'train'
    });

    // Evaluation data upload
    setupSingleUpload({
        uploadBox: document.getElementById('eval-upload'),
        fileInput: document.getElementById('eval_file'),
        fileInfo: document.getElementById('eval_file_info'),
        hiddenInput: document.getElementById('eval_filename'),
        fileType: 'eval'
    });
}

function setupSingleUpload({ uploadBox, fileInput, fileInfo, hiddenInput, fileType }) {
    uploadBox.addEventListener('click', () => fileInput.click());

    uploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadBox.style.borderColor = 'var(--heather-light)';
    });

    uploadBox.addEventListener('dragleave', () => {
        uploadBox.style.borderColor = '';
    });

    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadBox.style.borderColor = '';
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            handleFileUpload(e.dataTransfer.files[0], fileType, uploadBox, fileInfo, hiddenInput);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) {
            handleFileUpload(fileInput.files[0], fileType, uploadBox, fileInfo, hiddenInput);
        }
    });
}

async function handleFileUpload(file, fileType, uploadBox, fileInfo, hiddenInput) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('file_type', fileType);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const data = await response.json();
            hiddenInput.value = data.filepath;
            fileInfo.textContent = `${file.name} (${data.row_count} rows)`;
            uploadBox.classList.add('uploaded');
        } else {
            const error = await response.json();
            throw new Error(error.detail || 'Upload failed');
        }
    } catch (error) {
        console.error('Upload error:', error);
        alert(`Failed to upload file: ${error.message}`);
    }
}

// Auto-split evaluation data
function setupAutoSplit() {
    // File upload auto-split
    const autoSplitCheckbox = document.getElementById('auto_split_eval');
    const splitPctInput = document.getElementById('eval_split_pct');
    const evalFileInput = document.getElementById('eval_file');
    const evalFilenameInput = document.getElementById('eval_filename');

    // Enable/disable percentage input based on checkbox
    autoSplitCheckbox.addEventListener('change', () => {
        splitPctInput.disabled = !autoSplitCheckbox.checked;
    });

    // Disable auto-split if eval file is uploaded
    evalFileInput.addEventListener('change', () => {
        if (evalFileInput.files.length > 0) {
            autoSplitCheckbox.disabled = true;
            autoSplitCheckbox.checked = false;
            splitPctInput.disabled = true;
        } else {
            autoSplitCheckbox.disabled = false;
        }
    });

    // Also check when eval filename is cleared
    const observer = new MutationObserver(() => {
        if (!evalFilenameInput.value) {
            autoSplitCheckbox.disabled = false;
        }
    });
    observer.observe(evalFilenameInput, { attributes: true, attributeFilter: ['value'] });

    // HuggingFace auto-split
    const hfAutoSplitCheckbox = document.getElementById('hf_auto_split_eval');
    const hfSplitPctInput = document.getElementById('hf_eval_split_pct');
    const hfEvalSplitInput = document.getElementById('hf_eval_split');

    // Enable/disable percentage input based on checkbox
    hfAutoSplitCheckbox.addEventListener('change', () => {
        hfSplitPctInput.disabled = !hfAutoSplitCheckbox.checked;
        // Disable eval split input when auto-split is enabled
        if (hfAutoSplitCheckbox.checked) {
            hfEvalSplitInput.disabled = true;
            hfEvalSplitInput.value = '';
        } else {
            hfEvalSplitInput.disabled = false;
        }
    });

    // Disable auto-split if eval split is specified
    hfEvalSplitInput.addEventListener('input', () => {
        if (hfEvalSplitInput.value.trim()) {
            hfAutoSplitCheckbox.disabled = true;
            hfAutoSplitCheckbox.checked = false;
            hfSplitPctInput.disabled = true;
        } else {
            hfAutoSplitCheckbox.disabled = false;
        }
    });
}

// Training
async function handleTrainSubmit(e) {
    e.preventDefault();

    const projectName = document.getElementById('project_name').value.trim();

    // Validate project name
    if (!projectName) {
        alert('Please enter a project name');
        return;
    }

    if (!/^[a-zA-Z0-9][a-zA-Z0-9-]*$/.test(projectName)) {
        alert('Project name must start with alphanumeric and contain only alphanumeric characters and hyphens');
        return;
    }

    // Validate steps/ratio fields
    const stepsFields = [
        { id: 'logging_steps', name: 'Logging' },
        { id: 'eval_steps', name: 'Eval' },
        { id: 'save_steps', name: 'Save' }
    ];

    for (const field of stepsFields) {
        const value = parseFloat(document.getElementById(field.id).value);
        if (isNaN(value)) {
            alert(`${field.name} steps/ratio must be a valid number`);
            return;
        }
        if (value <= 0) {
            alert(`${field.name} steps/ratio must be greater than 0`);
            return;
        }
        // If it's a float, validate it's a ratio between 0 and 1
        if (value < 1 && value !== Math.floor(value)) {
            if (value > 1) {
                alert(`${field.name}: Float values must be ratios between 0 and 1 (e.g., 0.1 for 10%)`);
                return;
            }
        }
        // If it's meant to be an integer (value >= 1), check it
        if (value >= 1 && value !== Math.floor(value)) {
            alert(`${field.name}: Integer values (>= 1) must be whole numbers (e.g., 100, 500)`);
            return;
        }
    }

    // Get selected precision mode
    const precision = document.getElementById('precision').value || 'fp32';

    // Get loss variant (only if the field is visible/applicable)
    const lossVariantField = document.getElementById('loss-variant-field');
    const lossVariant = lossVariantField && lossVariantField.style.display !== 'none'
        ? document.getElementById('loss_variant').value
        : null;

    const formData = {
        project_name: projectName,
        task: document.getElementById('task').value,
        base_model: document.getElementById('base_model').value,
        loss_variant: lossVariant,
        epochs: parseInt(document.getElementById('epochs').value),
        max_steps: document.getElementById('max_steps').value
            ? parseInt(document.getElementById('max_steps').value)
            : null,
        batch_size: parseInt(document.getElementById('batch_size').value),
        learning_rate: parseFloat(document.getElementById('learning_rate').value),

        // Advanced settings
        warmup_ratio: parseFloat(document.getElementById('warmup_ratio').value),
        weight_decay: parseFloat(document.getElementById('weight_decay').value),
        gradient_accumulation_steps: parseInt(document.getElementById('gradient_accumulation_steps').value),
        logging_steps: parseFloat(document.getElementById('logging_steps').value),
        eval_steps: parseFloat(document.getElementById('eval_steps').value),
        save_steps: parseFloat(document.getElementById('save_steps').value),
        fp16: precision === 'fp16',
        bf16: precision === 'bf16',
        optimizer: document.getElementById('optimizer').value,
        scheduler: document.getElementById('scheduler').value,

        // LoRA settings
        lora_enabled: document.getElementById('lora_enabled').checked,
        lora_r: parseInt(document.getElementById('lora_r').value),
        lora_alpha: parseInt(document.getElementById('lora_alpha').value),
        lora_dropout: parseFloat(document.getElementById('lora_dropout').value),
        lora_target_modules: document.getElementById('lora_target_modules').value.trim(),

        // Model settings
        max_seq_length: document.getElementById('max_seq_length').value
            ? parseInt(document.getElementById('max_seq_length').value)
            : null,  // null = auto-detect from model
        gradient_checkpointing: document.getElementById('gradient_checkpointing').checked,

        // Unsloth settings
        unsloth_enabled: document.getElementById('unsloth_enabled').checked,
        unsloth_save_method: document.getElementById('unsloth_save_method').value,

        // Matryoshka settings (only send dims if enabled)
        matryoshka_dims: document.getElementById('matryoshka_enabled').checked
            ? document.getElementById('matryoshka_dims').value.trim()
            : null,

        // Hub settings
        push_to_hub: document.getElementById('push_to_hub').checked,
        hf_username: document.getElementById('hf_username').value.trim() || null,

        // Task-specific parameters
        ...getTaskSpecificParams(),
    };

    // Add data source based on selected tab
    if (currentDataSource === 'file') {
        formData.train_filename = document.getElementById('train_filename').value;
        formData.eval_filename = document.getElementById('eval_filename').value || null;

        // Auto-split evaluation data
        const autoSplit = document.getElementById('auto_split_eval').checked;
        if (autoSplit && !formData.eval_filename) {
            formData.eval_split_pct = parseFloat(document.getElementById('eval_split_pct').value);
        }

        if (!formData.train_filename) {
            alert('Please upload training data');
            return;
        }
    } else {
        formData.hf_dataset = document.getElementById('hf_dataset').value.trim();
        formData.hf_subset = document.getElementById('hf_subset').value.trim() || null;
        formData.hf_train_split = document.getElementById('hf_train_split').value.trim() || 'train';
        formData.hf_eval_split = document.getElementById('hf_eval_split').value.trim() || null;

        // Auto-split for HuggingFace datasets
        const hfAutoSplit = document.getElementById('hf_auto_split_eval').checked;
        if (hfAutoSplit && !formData.hf_eval_split) {
            formData.eval_split_pct = parseFloat(document.getElementById('hf_eval_split_pct').value);
        }

        if (!formData.hf_dataset) {
            alert('Please enter a HuggingFace dataset name');
            return;
        }
    }

    // Validate hub settings
    if (formData.push_to_hub && !formData.hf_username) {
        alert('Please enter your HuggingFace username');
        return;
    }

    // Show loading state
    const submitBtn = trainForm.querySelector('button[type="submit"]');
    const originalBtnText = submitBtn.textContent;
    submitBtn.disabled = true;
    submitBtn.textContent = 'Starting...';

    try {
        const response = await fetch('/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });

        if (response.ok) {
            const data = await response.json();
            newTrainingModal.style.display = 'none';
            resetChart();
            clearLogs();

            // Show status banner
            showStatusBanner('Initializing training...');

            loadRuns();
            // selectRun will load historical updates and start polling if active
            selectRun(data.run_id);
        } else {
            const error = await response.json();
            alert(`Training failed: ${error.detail || 'Unknown error'}`);
        }
    } catch (error) {
        console.error('Training error:', error);
        alert('Failed to start training');
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = originalBtnText;
    }
}

async function stopTraining() {
    if (!selectedRunId) return;

    const stopBtn = document.getElementById('stop-btn');
    stopBtn.classList.add('loading');
    stopBtn.disabled = true;

    try {
        const response = await fetch('/stop', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ run_id: selectedRunId })
        });

        if (response.ok) {
            stopPolling();
            // Hide stop button and status banner
            stopBtn.style.display = 'none';
            hideStatusBanner();
            // Update status display
            const statusEl = document.getElementById('summary-status');
            statusEl.textContent = 'stopped';
            statusEl.className = 'status-chip small stopped';
            // Refresh run list to get updated status
            await loadRuns();
        }
    } catch (error) {
        console.error('Stop error:', error);
    } finally {
        stopBtn.classList.remove('loading');
        stopBtn.disabled = false;
    }
}

async function deleteRun() {
    if (!selectedRunId) return;

    if (!confirm('Are you sure you want to delete this run?')) return;

    try {
        await fetch(`/runs/${selectedRunId}`, { method: 'DELETE' });
        stopPolling();
        selectedRunId = null;
        projectSummary.style.display = 'none';
        loadRuns();
    } catch (error) {
        console.error('Delete error:', error);
    }
}

// Runs
async function loadRuns(autoSelectLatest = false) {
    try {
        const response = await fetch('/runs');
        runs = await response.json();
        renderRunList();

        // Check for active run
        const activeResponse = await fetch('/active_run_id');
        const activeData = await activeResponse.json();
        activeRunId = activeData.run_id;

        // If there's an active run and we're not polling, start polling
        // Reset pollLine since we haven't loaded historical updates yet
        if (activeRunId && !pollInterval) {
            pollLine = 0;
            startPolling(activeRunId);
        }

        // Auto-select latest run if requested and no run is selected
        if (autoSelectLatest && runs.length > 0 && !selectedRunId) {
            selectRun(runs[0].id);
        }
    } catch (error) {
        console.error('Failed to load runs:', error);
    }
}

function renderRunList() {
    if (runs.length === 0) {
        runList.innerHTML = '<div class="run-item placeholder">No training runs yet</div>';
        return;
    }

    runList.innerHTML = runs.map(run => {
        const config = JSON.parse(run.config || '{}');
        const date = new Date(run.created_at).toLocaleDateString();
        const isSelected = run.id === selectedRunId;
        const projectName = config.project_name || `Run #${run.id}`;

        return `
            <div class="run-item ${isSelected ? 'active' : ''}" data-id="${run.id}">
                <span class="run-status-icon ${run.status}"></span>
                <div class="run-info">
                    <div class="run-id">${projectName}</div>
                    <div class="run-date">${date}</div>
                </div>
            </div>
        `;
    }).join('');

    // Add click handlers
    runList.querySelectorAll('.run-item:not(.placeholder)').forEach(item => {
        item.addEventListener('click', () => {
            selectRun(parseInt(item.dataset.id));
        });
    });
}

async function selectRun(runId) {
    selectedRunId = runId;
    renderRunList();

    // Reset chart and metrics when switching runs
    metricsData = {};
    resetChart();
    clearLogs();

    // Reset progress display
    document.getElementById('current-step').textContent = '0';
    document.getElementById('current-loss').textContent = '--';
    document.getElementById('current-eta').textContent = '--';
    document.getElementById('status-banner').style.display = 'none';

    try {
        const response = await fetch(`/runs/${runId}`);
        const run = await response.json();
        showRunSummary(run);

        // Load historical updates (logs and progress) for this run
        await loadHistoricalUpdates(runId);

        // Load metrics from TensorBoard files for this run
        await loadMetrics(runId);

        // Update header display based on run status
        if (run.status === 'running') {
            if (pollingRunId !== runId) {
                startPolling(runId);
            }
        } else {
            // For finished/stopped/error runs, show final state
            document.getElementById('current-eta').textContent = run.status === 'completed' ? '0' : '--';

            // Show final loss from metrics if available
            const lossData = metricsData['loss'] || metricsData['eval_loss'];
            if (lossData && lossData.length > 0) {
                const finalLoss = lossData[lossData.length - 1];
                if (finalLoss && finalLoss.value !== null) {
                    document.getElementById('current-loss').textContent = finalLoss.value.toFixed(4);
                }
            }
        }
    } catch (error) {
        console.error('Failed to load run:', error);
    }
}

async function loadHistoricalUpdates(runId) {
    try {
        // Load all updates from the beginning (since_line=0)
        const response = await fetch(`/runs/${runId}/updates?since_line=0`);
        if (!response.ok) return;

        const data = await response.json();

        // Process historical updates, but skip status updates since we have
        // the authoritative status from the API (status in update file may be stale)
        data.updates.forEach(update => {
            if (update.type !== 'status') {
                handleUpdate(update);
            }
        });

        // Set pollLine for future polling
        pollLine = data.next_line;

    } catch (error) {
        console.error('Failed to load historical updates:', error);
    }
}

function showRunSummary(run) {
    const config = JSON.parse(run.config || '{}');

    document.getElementById('summary-project-name').textContent = config.project_name || `Run #${run.id}`;
    document.getElementById('summary-status').textContent = run.status;
    document.getElementById('summary-status').className = `status-chip small ${run.status}`;

    document.getElementById('sum-task').textContent = config.task || '--';
    document.getElementById('sum-loss').textContent = config.loss_variant || 'default';
    document.getElementById('sum-model').textContent = config.base_model?.split('/').pop() || '--';

    // Show data source
    let dataSource = '--';
    if (config.train_filename) {
        dataSource = config.train_filename.split('/').pop();
    } else if (config.hf_dataset) {
        dataSource = config.hf_dataset;
    }
    document.getElementById('sum-data').textContent = dataSource;

    document.getElementById('sum-epochs').textContent = config.epochs || '--';
    document.getElementById('sum-batch').textContent = config.batch_size || '--';
    document.getElementById('sum-lr').textContent = config.learning_rate || '--';

    // Show LoRA info if enabled
    const loraRow = document.getElementById('sum-lora-row');
    if (config.lora_enabled) {
        loraRow.style.display = 'flex';
        document.getElementById('sum-lora').textContent = `r=${config.lora_r}, a=${config.lora_alpha}`;
    } else {
        loraRow.style.display = 'none';
    }

    // Show Matryoshka info if enabled
    const matryoshkaRow = document.getElementById('sum-matryoshka-row');
    if (config.matryoshka_dims) {
        matryoshkaRow.style.display = 'flex';
        document.getElementById('sum-matryoshka').textContent = config.matryoshka_dims;
    } else {
        matryoshkaRow.style.display = 'none';
    }

    // Show/hide stop button based on status
    const stopBtn = document.getElementById('stop-btn');
    stopBtn.style.display = run.status === 'running' ? 'block' : 'none';

    // Enable artifacts button for non-running runs (completed, stopped, error)
    // These may have artifacts like checkpoints or partial models
    artifactsBtn.disabled = run.status === 'running' || run.status === 'pending';

    projectSummary.style.display = 'block';
}

function updateRunStatus(runId, status) {
    const run = runs.find(r => r.id === runId);
    if (run) {
        run.status = status;
        renderRunList();
        if (selectedRunId === runId) {
            showRunSummary(run);
        }
    }
}

// Progress
function updateProgress(data) {
    const progressType = data.type || 'progress';

    if (progressType === 'train_start') {
        // Hide status banner when training starts
        hideStatusBanner();
        return;
    }

    if (progressType === 'train_end') {
        // Training complete
        document.getElementById('current-eta').textContent = 'Done';
        return;
    }

    // Update metrics display
    const totalSteps = data.total_steps || 0;
    const stepDisplay = totalSteps ? `${data.step || 0}/${totalSteps}` : (data.step || 0);
    document.getElementById('current-step').textContent = stepDisplay;

    document.getElementById('current-loss').textContent = data.loss?.toFixed(4) || '--';

    // Update ETA
    if (data.eta_seconds && data.eta_seconds > 0) {
        document.getElementById('current-eta').textContent = formatTime(data.eta_seconds);
    }

    if (data.step && data.loss) {
        updateChart(data.step, data.loss);
    }
}

function formatTime(seconds) {
    if (seconds < 60) {
        return `${Math.round(seconds)}s`;
    } else if (seconds < 3600) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.round(seconds % 60);
        return `${mins}m ${secs}s`;
    } else {
        const hours = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        return `${hours}h ${mins}m`;
    }
}

// Logs
function appendLog(message) {
    const logCount = document.getElementById('log-count');
    const count = parseInt(logCount.textContent) + 1;
    logCount.textContent = count;

    logContent.textContent += message + '\n';
    logContent.scrollTop = logContent.scrollHeight;
}

function clearLogs() {
    logContent.textContent = '';
    document.getElementById('log-count').textContent = '0';
}

// Status Banner
function showStatusBanner(message) {
    const banner = document.getElementById('status-banner');
    document.getElementById('status-message').textContent = message;
    banner.style.display = 'flex';
}

function hideStatusBanner() {
    document.getElementById('status-banner').style.display = 'none';
}

// Handlers
function handleTrainingComplete(data) {
    appendLog(`Training completed! Model saved to: ${data.output_dir}`);
    hideStatusBanner();
    // Hide stop button and update status
    document.getElementById('stop-btn').style.display = 'none';
    const statusEl = document.getElementById('summary-status');
    statusEl.textContent = 'completed';
    statusEl.className = 'status-chip small completed';
    stopPolling();
    loadRuns();
}

function handleTrainingError(data) {
    appendLog(`Error: ${data.message}`);
    hideStatusBanner();
    // Hide stop button and update status
    document.getElementById('stop-btn').style.display = 'none';
    const statusEl = document.getElementById('summary-status');
    statusEl.textContent = 'error';
    statusEl.className = 'status-chip small error';
    stopPolling();
    loadRuns();
}

// Metrics from TensorBoard files
async function loadMetrics(runId) {
    try {
        const response = await fetch(`/runs/${runId}/metrics`);
        if (response.ok) {
            const data = await response.json();
            metricsData = data.metrics || {};
            updateMetricSelector();
            updateChartWithMetric(currentMetric);
        }
    } catch (error) {
        console.error('Failed to load metrics:', error);
    }
}

function updateMetricSelector() {
    const select = document.getElementById('metric-select');

    // Filter to only metrics with multiple valid data points (single-point metrics are useless for charts)
    // Also exclude epoch-related metrics as they're not useful for charting
    const availableMetrics = Object.keys(metricsData).filter(key => {
        const keyLower = key.toLowerCase();
        // Skip epoch metrics
        if (keyLower === 'epoch' || keyLower.includes('epoch')) return false;
        const data = metricsData[key];
        if (!data || data.length < 2) return false;  // Need at least 2 points for a line chart
        return data.filter(d => d.value !== null).length >= 2;
    });

    if (availableMetrics.length === 0) {
        select.innerHTML = '<option value="loss">Loss</option>';
        return;
    }

    // Sort metrics: exact 'loss' first, eval_loss second, then alphabetically
    availableMetrics.sort((a, b) => {
        const aLower = a.toLowerCase();
        const bLower = b.toLowerCase();
        // Exact 'loss' first (case-insensitive)
        if (aLower === 'loss') return -1;
        if (bLower === 'loss') return 1;
        // eval_loss second
        if (aLower === 'eval_loss') return -1;
        if (bLower === 'eval_loss') return 1;
        // Push flos/runtime/samples_per_second to the end (less useful metrics)
        const aIsUtility = aLower.includes('flos') || aLower.includes('runtime') || aLower.includes('per_second');
        const bIsUtility = bLower.includes('flos') || bLower.includes('runtime') || bLower.includes('per_second');
        if (aIsUtility && !bIsUtility) return 1;
        if (bIsUtility && !aIsUtility) return -1;
        // Then alphabetically
        return a.localeCompare(b);
    });

    select.innerHTML = availableMetrics.map(metric => {
        const label = metric.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        return `<option value="${metric}">${label}</option>`;
    }).join('');

    // Find the best metric to select (prefer exact 'loss')
    let selectedMetric = currentMetric;
    if (!availableMetrics.includes(selectedMetric)) {
        // Try to find exact 'loss' (case-insensitive)
        selectedMetric = availableMetrics.find(m => m.toLowerCase() === 'loss')
            || availableMetrics.find(m => m.toLowerCase() === 'eval_loss')
            || availableMetrics[0];
    }

    currentMetric = selectedMetric;
    select.value = currentMetric;
}

function updateChartWithMetric(metric) {
    const data = metricsData[metric];
    if (!data || data.length === 0) {
        return;
    }

    // Update chart with metric data
    chart.data.labels = data.map(d => d.step);
    chart.data.datasets[0].data = data.map(d => d.value);
    chart.data.datasets[0].label = metric.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    chart.update();
    chartPlaceholder.style.display = 'none';
}

// Artifacts Modal
function setupArtifactsModal() {
    // Open artifacts modal
    artifactsBtn.addEventListener('click', async () => {
        if (!selectedRunId || artifactsBtn.disabled) return;
        await loadArtifacts(selectedRunId);
        artifactsModal.style.display = 'flex';
    });

    // Close artifacts modal
    closeArtifactsModal.addEventListener('click', () => {
        artifactsModal.style.display = 'none';
    });

    // Close on backdrop click
    artifactsModal.addEventListener('click', (e) => {
        if (e.target === artifactsModal) {
            artifactsModal.style.display = 'none';
        }
    });

    // Close on escape
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && artifactsModal.style.display === 'flex') {
            artifactsModal.style.display = 'none';
        }
    });
}

async function loadArtifacts(runId) {
    const listEl = document.getElementById('artifacts-list');
    const pathEl = document.getElementById('artifacts-path');

    try {
        const response = await fetch(`/runs/${runId}/artifacts`);
        const data = await response.json();

        if (data.artifacts.length === 0) {
            listEl.innerHTML = '<div class="artifacts-empty">No artifacts available</div>';
            pathEl.textContent = '';
            return;
        }

        listEl.innerHTML = data.artifacts.map(artifact => `
            <div class="artifact-item">
                <div class="artifact-info">
                    <span class="artifact-name">${artifact.label}</span>
                    <div class="artifact-meta">
                        <span class="artifact-category">${artifact.category}</span>
                        <span>${formatFileSize(artifact.size)}</span>
                    </div>
                </div>
                <button class="artifact-download" data-path="${artifact.path.replace(/"/g, '&quot;')}">
                    Copy Path
                </button>
            </div>
        `).join('');

        // Add event delegation for copy buttons
        listEl.querySelectorAll('.artifact-download').forEach(btn => {
            btn.addEventListener('click', function() {
                const path = this.getAttribute('data-path');
                copyArtifactPath(path, this);
            });
        });

        pathEl.textContent = data.output_dir;
    } catch (error) {
        console.error('Failed to load artifacts:', error);
        listEl.innerHTML = '<div class="artifacts-empty">Failed to load artifacts</div>';
    }
}

function copyArtifactPath(path, btn) {
    navigator.clipboard.writeText(path).then(() => {
        // Brief visual feedback
        const originalText = btn.textContent;
        btn.textContent = 'Copied!';
        setTimeout(() => {
            btn.textContent = originalText;
        }, 1000);
    }).catch(err => {
        console.error('Failed to copy:', err);
    });
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}
