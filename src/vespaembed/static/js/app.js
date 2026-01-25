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

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    initChart();
    await loadTasks(); // Load tasks from API first
    loadRuns();
    setupEventListeners();
    setupFileUploads();
    setupTabs();
    setupHubToggle();
    setupTaskSelector();
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

    // Update sample data display
    updateSampleData(task);

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

        // Set precision radio button
        if (hyper.bf16) {
            setRadioChecked('precision', 'bf16');
        } else if (hyper.fp16) {
            setRadioChecked('precision', 'fp16');
        } else {
            setRadioChecked('precision', 'fp32');
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
                pointRadius: 0,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
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
    const taskSelect = document.getElementById('task');
    const task = tasksData.find(t => t.name === taskSelect.value);
    if (task) {
        updateTaskUI(task);
    }

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
    document.getElementById('hub_model_id').value = '';
    document.getElementById('hub-model-field').style.display = 'none';

    // Reset unsloth
    document.getElementById('use_unsloth').checked = false;
}

// Event Listeners
function setupEventListeners() {
    // New Training Modal
    newTrainingBtn.addEventListener('click', () => {
        // Generate random project name when opening modal
        document.getElementById('project_name').value = generateProjectName();

        // Reset form and apply defaults for currently selected task
        resetFormToTaskDefaults();

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
    const hubModelField = document.getElementById('hub-model-field');

    pushToHub.addEventListener('change', () => {
        hubModelField.style.display = pushToHub.checked ? 'block' : 'none';
    });
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

    // Get selected precision mode
    const precision = document.querySelector('input[name="precision"]:checked')?.value || 'fp16';

    const formData = {
        project_name: projectName,
        task: document.getElementById('task').value,
        base_model: document.getElementById('base_model').value,
        epochs: parseInt(document.getElementById('epochs').value),
        batch_size: parseInt(document.getElementById('batch_size').value),
        learning_rate: parseFloat(document.getElementById('learning_rate').value),

        // Advanced settings
        warmup_ratio: parseFloat(document.getElementById('warmup_ratio').value),
        weight_decay: parseFloat(document.getElementById('weight_decay').value),
        gradient_accumulation_steps: parseInt(document.getElementById('gradient_accumulation_steps').value),
        logging_steps: parseInt(document.getElementById('logging_steps').value),
        eval_steps: parseInt(document.getElementById('eval_steps').value),
        save_steps: parseInt(document.getElementById('save_steps').value),
        fp16: precision === 'fp16',
        bf16: precision === 'bf16',
        use_unsloth: document.getElementById('use_unsloth').checked,

        // Hub settings
        push_to_hub: document.getElementById('push_to_hub').checked,
        hub_model_id: document.getElementById('hub_model_id').value || null,

        // Task-specific parameters
        ...getTaskSpecificParams(),
    };

    // Add data source based on selected tab
    if (currentDataSource === 'file') {
        formData.train_filename = document.getElementById('train_filename').value;
        formData.eval_filename = document.getElementById('eval_filename').value || null;

        if (!formData.train_filename) {
            alert('Please upload training data');
            return;
        }
    } else {
        formData.hf_dataset = document.getElementById('hf_dataset').value.trim();
        formData.hf_subset = document.getElementById('hf_subset').value.trim() || null;
        formData.hf_train_split = document.getElementById('hf_train_split').value.trim() || 'train';
        formData.hf_eval_split = document.getElementById('hf_eval_split').value.trim() || null;

        if (!formData.hf_dataset) {
            alert('Please enter a HuggingFace dataset name');
            return;
        }
    }

    // Validate hub settings
    if (formData.push_to_hub && !formData.hub_model_id) {
        alert('Please enter a Hub Model ID when pushing to Hub');
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
            // Hide progress bar
            document.getElementById('progress-container').style.display = 'none';
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
async function loadRuns() {
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
    document.getElementById('current-epoch').textContent = '0';
    document.getElementById('current-step').textContent = '0';
    document.getElementById('current-loss').textContent = '--';
    document.getElementById('current-eta').textContent = '--';
    document.getElementById('progress-container').style.display = 'none';
    document.getElementById('progress-fill').style.width = '0%';
    document.getElementById('progress-pct').textContent = '0%';
    document.getElementById('progress-speed').textContent = '-- it/s';
    document.getElementById('status-banner').style.display = 'none';

    try {
        const response = await fetch(`/runs/${runId}`);
        const run = await response.json();
        showRunSummary(run);

        // Load historical updates (logs and progress) for this run
        await loadHistoricalUpdates(runId);

        // Load metrics from TensorBoard files for this run
        await loadMetrics(runId);

        // If this run is active, show progress bar and start polling
        if (run.status === 'running') {
            document.getElementById('progress-container').style.display = 'block';
            if (pollingRunId !== runId) {
                startPolling(runId);
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

    // Show/hide stop button
    const stopBtn = document.getElementById('stop-btn');
    stopBtn.style.display = run.status === 'running' ? 'block' : 'none';

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
        // Hide status banner and show progress bar when training starts
        hideStatusBanner();
        document.getElementById('progress-container').style.display = 'block';
        document.getElementById('progress-fill').style.width = '0%';
        document.getElementById('progress-pct').textContent = '0%';
        document.getElementById('progress-speed').textContent = '-- it/s';
        return;
    }

    if (progressType === 'train_end') {
        // Training complete - fill bar to 100%
        document.getElementById('progress-fill').style.width = '100%';
        document.getElementById('progress-pct').textContent = '100%';
        document.getElementById('progress-speed').textContent = 'Complete';
        document.getElementById('current-eta').textContent = 'Done';
        return;
    }

    // Update metrics display
    const totalEpochs = data.total_epochs || 0;
    const epochDisplay = totalEpochs ? `${data.epoch?.toFixed(1) || 0}/${totalEpochs}` : (data.epoch?.toFixed(2) || 0);
    document.getElementById('current-epoch').textContent = epochDisplay;

    const totalSteps = data.total_steps || 0;
    const stepDisplay = totalSteps ? `${data.step || 0}/${totalSteps}` : (data.step || 0);
    document.getElementById('current-step').textContent = stepDisplay;

    document.getElementById('current-loss').textContent = data.loss?.toFixed(4) || '--';

    // Update ETA
    if (data.eta_seconds && data.eta_seconds > 0) {
        document.getElementById('current-eta').textContent = formatTime(data.eta_seconds);
    }

    // Update progress bar
    if (data.progress_pct !== undefined) {
        document.getElementById('progress-container').style.display = 'block';
        document.getElementById('progress-fill').style.width = `${data.progress_pct}%`;
        document.getElementById('progress-pct').textContent = `${data.progress_pct.toFixed(1)}%`;
    }

    // Update speed
    if (data.steps_per_sec) {
        document.getElementById('progress-speed').textContent = `${data.steps_per_sec.toFixed(2)} it/s`;
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
    document.getElementById('progress-container').style.display = 'none';
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
    document.getElementById('progress-container').style.display = 'none';
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
    const availableMetrics = Object.keys(metricsData);

    if (availableMetrics.length === 0) {
        select.innerHTML = '<option value="loss">Loss</option>';
        return;
    }

    // Sort metrics: loss first, then alphabetically
    availableMetrics.sort((a, b) => {
        if (a === 'loss') return -1;
        if (b === 'loss') return 1;
        return a.localeCompare(b);
    });

    select.innerHTML = availableMetrics.map(metric => {
        const label = metric.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        return `<option value="${metric}" ${metric === currentMetric ? 'selected' : ''}>${label}</option>`;
    }).join('');

    // If current metric not in list, select first one
    if (!availableMetrics.includes(currentMetric)) {
        currentMetric = availableMetrics[0] || 'loss';
        select.value = currentMetric;
    }
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
