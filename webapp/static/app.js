/**
 * OCR LoRA Correction Web Application - Alpine.js Application State
 * Phase 2: Interactive correction UI with character-level diff visualization.
 */

document.addEventListener('alpine:init', () => {
    // Top-level Application State
    Alpine.data('appState', () => ({
        // Navigation & Routing
        activeTab: 'correct',
        
        // Dark / Light Theme
        isDark: false,

        // System & Health Status
        health: {
            status: 'connecting',
            model_state: 'unloaded',
            active_adapter: null,
            gpu: {
                available: false,
                device_name: null,
                total_mb: 0,
                used_mb: 0,
                free_mb: 0,
            }
        },
        healthInterval: null,

        // Initialize Application
        init() {
            this.initTheme();
            this.initRouting();
            this.fetchHealth();
            this.healthInterval = setInterval(() => this.fetchHealth(), 3000);
        },

        // Theme Management
        initTheme() {
            const savedTheme = localStorage.getItem('theme');
            if (savedTheme === 'dark' || (!savedTheme && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
                this.isDark = true;
                document.documentElement.classList.add('dark');
            } else {
                this.isDark = false;
                document.documentElement.classList.remove('dark');
            }
        },

        toggleTheme() {
            this.isDark = !this.isDark;
            if (this.isDark) {
                document.documentElement.classList.add('dark');
                localStorage.setItem('theme', 'dark');
            } else {
                document.documentElement.classList.remove('dark');
                localStorage.setItem('theme', 'light');
            }
        },

        // Hash Routing
        initRouting() {
            const handleHash = () => {
                const hash = window.location.hash.replace('#', '').toLowerCase();
                const validTabs = ['correct', 'batch', 'analytics', 'training'];
                if (validTabs.includes(hash)) {
                    this.activeTab = hash;
                } else {
                    this.activeTab = 'correct';
                    window.location.hash = '#correct';
                }
            };

            window.addEventListener('hashchange', handleHash);
            handleHash();
        },

        setTab(tab) {
            this.activeTab = tab;
            window.location.hash = `#${tab}`;
        },

        // Health Polling
        async fetchHealth() {
            try {
                const res = await fetch('/api/health');
                if (res.ok) {
                    this.health = await res.json();
                } else {
                    this.health.status = 'error';
                }
            } catch (err) {
                this.health.status = 'disconnected';
                this.health.model_state = 'offline';
            }
        },

        // Model Status Display Helpers
        get modelStateBadge() {
            const state = this.health.model_state || 'unloaded';
            switch (state) {
                case 'ready':
                    return { text: 'Ready', color: 'bg-emerald-500', pulse: false };
                case 'loading':
                    return { text: 'Loading...', color: 'bg-amber-500', pulse: true };
                case 'busy':
                    return { text: 'Busy', color: 'bg-amber-500', pulse: true };
                case 'error':
                    return { text: 'Error', color: 'bg-rose-500', pulse: false };
                case 'offline':
                    return { text: 'Offline', color: 'bg-zinc-500', pulse: false };
                case 'unloaded':
                default:
                    return { text: 'Unloaded', color: 'bg-zinc-400 dark:bg-zinc-600', pulse: false };
            }
        }
    }));

    // Interactive Correction Tab Component
    Alpine.data('correctTab', () => ({
        inputText: '',
        maxChars: 2048,
        selectedAdapter: '',
        adapters: [],
        isLoadingAdapters: false,
        isAdapterSwitching: false,
        isSubmitting: false,

        // Output Result
        result: null,
        errorMessage: null,
        errorType: null, // '503', 'length', 'network'
        copySuccess: false,

        // Pre-configured Test Samples
        samples: [
            {
                label: 'Math Formulas',
                text: 'Ca1culate: 4S x 12 = S40 and 128 ÷ 4 = 32. What is the swn?'
            },
            {
                label: 'Corrupted WikiText',
                text: 'The va1ley was discovered in l902 by exp1orers who trave1ed north through the dense rnodern forests of the rnounlain range.'
            },
            {
                label: 'O / 0 Confusion',
                text: 'Order 2O2 has O missing parts with code O-982O.'
            },
            {
                label: 'l / 1 Confusion',
                text: 'There were l04 students in c1ass l, and 12 on the waiting 1ist.'
            },
            {
                label: 'rn / m Confusion',
                text: 'The rnodern artist painted early this rnorning before the sumrner rneeting.'
            },
            {
                label: 'x vs × Confusion',
                text: 'Room size is 12 x 15 feet = 180 sq ft, and height is 10 x 2 = 20 ft.'
            }
        ],

        async init() {
            await this.loadAdapters();
            // Pre-fill first sample if empty
            if (!this.inputText) {
                this.inputText = this.samples[0].text;
            }
        },

        // Load adapter catalog from API
        async loadAdapters() {
            this.isLoadingAdapters = true;
            try {
                const res = await fetch('/api/adapters');
                if (res.ok) {
                    this.adapters = await res.json();
                    const active = this.adapters.find(a => a.is_active);
                    if (active) {
                        this.selectedAdapter = active.name;
                    } else if (this.adapters.length > 0) {
                        // Default to large adapter if present
                        const large = this.adapters.find(a => a.name.includes('large'));
                        this.selectedAdapter = large ? large.name : this.adapters[0].name;
                    }
                }
            } catch (err) {
                console.error('Failed to load adapters:', err);
            } finally {
                this.isLoadingAdapters = false;
            }
        },

        onAdapterChange() {
            this.isAdapterSwitching = true;
            // Provide visual feedback for adapter selection
            setTimeout(() => {
                this.isAdapterSwitching = false;
            }, 600);
        },

        setSample(sample) {
            this.inputText = sample.text;
            this.result = null;
            this.errorMessage = null;
        },

        // Keyboard Shortcut: Ctrl + Enter / Cmd + Enter
        handleKeydown(e) {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                if (!this.isSubmitting && this.inputText.trim()) {
                    this.submitCorrection();
                }
            }
        },

        // Submit Correction Request
        async submitCorrection() {
            if (!this.inputText.trim() || this.isSubmitting) return;

            if (this.inputText.length > this.maxChars) {
                this.errorMessage = `Input text exceeds maximum allowed length of ${this.maxChars.toLocaleString()} characters.`;
                this.errorType = 'length';
                return;
            }

            this.isSubmitting = true;
            this.errorMessage = null;
            this.errorType = null;
            this.copySuccess = false;

            try {
                const res = await fetch('/api/correct', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        text: this.inputText,
                        adapter: this.selectedAdapter || undefined
                    })
                });

                if (res.ok) {
                    this.result = await res.json();
                } else if (res.status === 503) {
                    const errData = await res.json().catch(() => ({}));
                    this.errorMessage = 'Training in progress. Model inference is temporarily unavailable while GPU resources are reserved for training.';
                    this.errorType = '503';
                    this.result = null;
                } else if (res.status === 400 || res.status === 413 || res.status === 422) {
                    const errData = await res.json().catch(() => ({}));
                    this.errorMessage = errData.detail || 'Input validation failed. Please check the character length.';
                    this.errorType = 'length';
                    this.result = null;
                } else {
                    const errData = await res.json().catch(() => ({}));
                    this.errorMessage = errData.detail || `Server error (${res.status}): Correction failed.`;
                    this.errorType = 'server';
                    this.result = null;
                }
            } catch (err) {
                this.errorMessage = 'Unable to reach the OCR LoRA backend service at 127.0.0.1:8000. Please verify the web server is running.';
                this.errorType = 'network';
                this.result = null;
            } finally {
                this.isSubmitting = false;
            }
        },

        // Copy Corrected Text to Clipboard
        async copyCorrectedText() {
            if (!this.result || !this.result.corrected) return;
            try {
                await navigator.clipboard.writeText(this.result.corrected);
                this.copySuccess = true;
                setTimeout(() => {
                    this.copySuccess = false;
                }, 2000);
            } catch (err) {
                console.error('Failed to copy to clipboard:', err);
            }
        },

        // Diff Breakdown Helpers (Safe DOM rendering via x-text)
        get originalSegments() {
            if (!this.result || !this.result.diff) return [];
            return this.result.diff
                .filter(op => op.tag === 'equal' || op.tag === 'replace' || op.tag === 'delete')
                .map(op => ({
                    tag: op.tag,
                    text: op.a_text,
                    isSpace: op.a_text === ' ',
                    isNewline: op.a_text === '\n'
                }));
        },

        get correctedSegments() {
            if (!this.result || !this.result.diff) return [];
            return this.result.diff
                .filter(op => op.tag === 'equal' || op.tag === 'replace' || op.tag === 'insert')
                .map(op => ({
                    tag: op.tag,
                    text: op.b_text,
                    isSpace: op.b_text === ' ',
                    isNewline: op.b_text === '\n'
                }));
        },

        get hasChanges() {
            if (!this.result || !this.result.diff) return false;
            return this.result.diff.some(op => op.tag !== 'equal');
        }
    }));

    // Batch Processing Tab Component
    Alpine.data('batchTab', () => ({
        // File selection & upload state
        selectedFile: null,
        fileName: '',
        fileSizeFormatted: '',
        isDragging: false,
        isUploading: false,
        uploadError: null,

        // Configuration
        selectedAdapter: '',
        adapters: [],
        isLoadingAdapters: false,

        // Active Job State
        currentJobId: null,
        jobStatus: 'idle', // idle | needs_column | queued | running | completed | cancelled | error
        detectedColumns: [],
        selectedColumn: '',
        isConfiguringColumn: false,

        // Live Progress Metrics
        progress: {
            done_rows: 0,
            total_rows: 0,
            percent: 0,
            rows_per_sec: 0,
            eta_seconds: 0
        },

        // Preview & History
        previewRows: [],
        isLoadingPreview: false,
        recentJobs: [],
        isLoadingRecent: false,

        eventSource: null,

        async init() {
            await this.loadAdapters();
            await this.loadRecentJobs();
        },

        async loadAdapters() {
            this.isLoadingAdapters = true;
            try {
                const res = await fetch('/api/adapters');
                if (res.ok) {
                    this.adapters = await res.json();
                    if (this.adapters.length > 0) {
                        const large = this.adapters.find(a => a.name.includes('large'));
                        this.selectedAdapter = large ? large.name : this.adapters[0].name;
                    }
                }
            } catch (err) {
                console.error('Failed to load adapters:', err);
            } finally {
                this.isLoadingAdapters = false;
            }
        },

        async loadRecentJobs() {
            this.isLoadingRecent = true;
            try {
                const res = await fetch('/api/batch');
                if (res.ok) {
                    this.recentJobs = await res.json();
                }
            } catch (err) {
                console.error('Failed to load recent jobs:', err);
            } finally {
                this.isLoadingRecent = false;
            }
        },

        // Drag & Drop / File Selection
        handleFileDrop(e) {
            this.isDragging = false;
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.setFile(files[0]);
            }
        },

        handleFileSelect(e) {
            const files = e.target.files;
            if (files.length > 0) {
                this.setFile(files[0]);
            }
        },

        setFile(file) {
            const ext = file.name.split('.').pop().toLowerCase();
            if (!['txt', 'csv', 'jsonl'].includes(ext)) {
                this.uploadError = `Unsupported file type (.${ext}). Please select a .txt, .csv, or .jsonl file.`;
                return;
            }

            this.selectedFile = file;
            this.fileName = file.name;
            const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
            this.fileSizeFormatted = sizeMb < 0.1 ? `${(file.size / 1024).toFixed(1)} KB` : `${sizeMb} MB`;
            this.uploadError = null;
            this.resetJobState();
        },

        resetJobState() {
            if (this.eventSource) {
                this.eventSource.close();
                this.eventSource = null;
            }
            this.currentJobId = null;
            this.jobStatus = 'idle';
            this.detectedColumns = [];
            this.selectedColumn = '';
            this.previewRows = [];
            this.progress = {
                done_rows: 0,
                total_rows: 0,
                percent: 0,
                rows_per_sec: 0,
                eta_seconds: 0
            };
        },

        // Start Upload & Processing
        async startBatchUpload() {
            if (!this.selectedFile || this.isUploading) return;

            this.isUploading = true;
            this.uploadError = null;

            const formData = new FormData();
            formData.append('file', this.selectedFile);
            if (this.selectedAdapter) {
                formData.append('adapter', this.selectedAdapter);
            }

            try {
                const res = await fetch('/api/batch', {
                    method: 'POST',
                    body: formData
                });

                if (res.ok) {
                    const data = await res.json();
                    this.currentJobId = data.job_id;
                    this.jobStatus = data.status;

                    if (data.status === 'needs_column') {
                        this.detectedColumns = data.detected_columns;
                        this.selectedColumn = this.detectedColumns[0] || '';
                    } else {
                        this.connectSSE(this.currentJobId);
                    }
                    this.loadRecentJobs();
                } else if (res.status === 413) {
                    const err = await res.json().catch(() => ({}));
                    this.uploadError = err.detail || 'File exceeds upload size or row count limit.';
                } else if (res.status === 503) {
                    this.uploadError = 'Training is currently in progress. GPU resources are locked; batch processing unavailable.';
                } else {
                    const err = await res.json().catch(() => ({}));
                    this.uploadError = err.detail || 'Upload failed.';
                }
            } catch (err) {
                this.uploadError = 'Connection failed. Please ensure the server is running at 127.0.0.1:8000.';
            } finally {
                this.isUploading = false;
            }
        },

        // Configure column for CSV / JSONL
        async submitColumnChoice() {
            if (!this.currentJobId || !this.selectedColumn) return;

            this.isConfiguringColumn = true;
            const formData = new FormData();
            formData.append('column_name', this.selectedColumn);

            try {
                const res = await fetch(`/api/batch/${this.currentJobId}/configure`, {
                    method: 'POST',
                    body: formData
                });

                if (res.ok) {
                    const data = await res.json();
                    this.jobStatus = data.status;
                    this.connectSSE(this.currentJobId);
                    this.loadRecentJobs();
                } else {
                    const err = await res.json().catch(() => ({}));
                    this.uploadError = err.detail || 'Failed to configure column.';
                }
            } catch (err) {
                this.uploadError = 'Network error while configuring column.';
            } finally {
                this.isConfiguringColumn = false;
            }
        },

        // Connect to SSE event stream
        connectSSE(jobId) {
            if (this.eventSource) {
                this.eventSource.close();
            }

            this.eventSource = new EventSource(`/api/batch/${jobId}/events`);

            this.eventSource.addEventListener('progress', (e) => {
                const data = JSON.parse(e.data);
                this.jobStatus = data.status;
                this.progress = {
                    done_rows: data.done_rows,
                    total_rows: data.total_rows,
                    percent: data.percent,
                    rows_per_sec: data.rows_per_sec,
                    eta_seconds: data.eta_seconds
                };
                // Automatically fetch preview when rows are processed
                if (data.done_rows > 0 && this.previewRows.length === 0) {
                    this.loadPreview(jobId);
                }
            });

            this.eventSource.addEventListener('done', (e) => {
                const data = JSON.parse(e.data);
                this.jobStatus = 'completed';
                this.progress.done_rows = data.done_rows || this.progress.total_rows;
                this.progress.percent = 100.0;
                this.loadPreview(jobId);
                this.loadRecentJobs();
                if (this.eventSource) {
                    this.eventSource.close();
                    this.eventSource = null;
                }
            });

            this.eventSource.addEventListener('error', (e) => {
                if (e.data) {
                    try {
                        const data = JSON.parse(e.data);
                        this.jobStatus = data.status || 'error';
                        this.uploadError = data.message || 'Batch job interrupted.';
                    } catch (err) {
                        this.jobStatus = 'error';
                    }
                }
                this.loadRecentJobs();
                if (this.eventSource) {
                    this.eventSource.close();
                    this.eventSource = null;
                }
            });
        },

        // Cancel Active Job
        async cancelActiveJob() {
            if (!this.currentJobId) return;
            try {
                await fetch(`/api/batch/${this.currentJobId}/cancel`, { method: 'POST' });
                this.jobStatus = 'cancelled';
                if (this.eventSource) {
                    this.eventSource.close();
                    this.eventSource = null;
                }
                this.loadRecentJobs();
            } catch (err) {
                console.error('Failed to cancel job:', err);
            }
        },

        // Load 10-row preview with diffs
        async loadPreview(jobId) {
            const targetId = jobId || this.currentJobId;
            if (!targetId) return;

            this.isLoadingPreview = true;
            try {
                const res = await fetch(`/api/batch/${targetId}/preview`);
                if (res.ok) {
                    this.previewRows = await res.json();
                }
            } catch (err) {
                console.error('Failed to load preview:', err);
            } finally {
                this.isLoadingPreview = false;
            }
        },

        // View an existing historical job
        viewJob(job) {
            this.resetJobState();
            this.currentJobId = job.job_id;
            this.fileName = job.filename;
            this.jobStatus = job.status;
            this.progress = {
                done_rows: job.done_rows,
                total_rows: job.total_rows,
                percent: job.total_rows > 0 ? Math.round((job.done_rows / job.total_rows) * 100) : 0,
                rows_per_sec: 0,
                eta_seconds: 0
            };

            if (job.status === 'running' || job.status === 'queued') {
                this.connectSSE(job.job_id);
            } else {
                this.loadPreview(job.job_id);
            }
        },

        // Download Results
        download(fmt) {
            if (!this.currentJobId) return;
            window.location.href = `/api/batch/${this.currentJobId}/download?format=${fmt}`;
        }
    }));

    // Analytics Dashboard Tab Component (Phase 4a)
    Alpine.data('analyticsTab', () => ({
        // Summary Metrics
        summary: {
            total_rows: 0,
            exact_matches: 0,
            exact_match_acc: 0,
            math_total: 0,
            math_correct: 0,
            math_acc: 0,
            text_total: 0,
            text_correct: 0,
            text_acc: 0,
            failures_count: 0,
            checkpoint_file: '',
        },
        isLoadingSummary: false,

        // Confusion Matrix & Top Pairs
        confusion: {
            characters: [],
            grid: [],
            max_count: 0,
            top_pairs: [],
            domain: 'all',
        },
        isLoadingConfusion: false,
        confusionDomain: 'all', // 'all' | 'math' | 'text'
        sortField: 'count',     // 'count' | 'expected' | 'predicted'
        sortAsc: false,
        hoverCell: null,        // { expected, predicted, count }

        // Failures Explorer
        failures: [],
        failuresTotal: 0,
        failuresPage: 1,
        failuresPageSize: 10,
        failuresTotalPages: 1,
        isLoadingFailures: false,
        searchQuery: '',
        selectedDomain: '',     // '' (all) | 'math' | 'text'
        activePairFilter: null, // { expected: 'l', predicted: '1' }
        debounceTimer: null,

        async init() {
            // Watch for tab activation to load data if not loaded yet
            this.loadSummary();
            this.loadConfusion();
            this.loadFailures();
        },

        // Load Executive Summary
        async loadSummary() {
            this.isLoadingSummary = true;
            try {
                const res = await fetch('/api/analytics/summary');
                if (res.ok) {
                    this.summary = await res.json();
                }
            } catch (err) {
                console.error('Failed to load analytics summary:', err);
            } finally {
                this.isLoadingSummary = false;
            }
        },

        // Load Confusion Matrix & Pairs
        async loadConfusion() {
            this.isLoadingConfusion = true;
            try {
                const domainParam = this.confusionDomain !== 'all' ? `&domain=${this.confusionDomain}` : '';
                const res = await fetch(`/api/analytics/confusion?top=16${domainParam}`);
                if (res.ok) {
                    this.confusion = await res.json();
                }
            } catch (err) {
                console.error('Failed to load confusion matrix:', err);
            } finally {
                this.isLoadingConfusion = false;
            }
        },

        setConfusionDomain(domain) {
            this.confusionDomain = domain;
            this.loadConfusion();
        },

        // Heatmap cell styling: compute intensity RGBA based on count
        getCellStyle(count) {
            if (!count || count <= 0 || !this.confusion.max_count) {
                return 'background-color: transparent;';
            }
            // Use logarithmic scaling to make lower non-zero counts visible
            const ratio = Math.log(count + 1) / Math.log(this.confusion.max_count + 1);
            const alpha = Math.min(0.95, Math.max(0.12, ratio * 0.95));
            return `background-color: rgba(99, 102, 241, ${alpha.toFixed(2)});`;
        },

        getCellTextColor(count) {
            if (!count || count <= 0 || !this.confusion.max_count) {
                return 'text-zinc-400 dark:text-zinc-600';
            }
            const ratio = count / this.confusion.max_count;
            return ratio > 0.4 ? 'text-white font-semibold' : 'text-indigo-950 dark:text-indigo-100 font-semibold';
        },

        // Top Pairs Sorting
        setSort(field) {
            if (this.sortField === field) {
                this.sortAsc = !this.sortAsc;
            } else {
                this.sortField = field;
                this.sortAsc = (field !== 'count'); // default count is desc, strings asc
            }
        },

        get sortedTopPairs() {
            if (!this.confusion || !this.confusion.top_pairs) return [];
            const pairs = [...this.confusion.top_pairs];
            return pairs.sort((a, b) => {
                let valA = a[this.sortField];
                let valB = b[this.sortField];
                if (typeof valA === 'string') valA = valA.toLowerCase();
                if (typeof valB === 'string') valB = valB.toLowerCase();
                if (valA < valB) return this.sortAsc ? -1 : 1;
                if (valA > valB) return this.sortAsc ? 1 : -1;
                return 0;
            });
        },

        // Load Failures with Search & Pagination
        async loadFailures() {
            this.isLoadingFailures = true;
            try {
                const params = new URLSearchParams({
                    page: this.failuresPage.toString(),
                    page_size: this.failuresPageSize.toString(),
                });
                if (this.searchQuery && this.searchQuery.trim()) {
                    params.set('q', this.searchQuery.trim());
                }
                if (this.selectedDomain) {
                    params.set('domain', this.selectedDomain);
                }

                const res = await fetch(`/api/analytics/failures?${params.toString()}`);
                if (res.ok) {
                    const data = await res.json();
                    this.failures = data.failures;
                    this.failuresTotal = data.total;
                    this.failuresPage = data.page;
                    this.failuresPageSize = data.page_size;
                    this.failuresTotalPages = data.total_pages;
                }
            } catch (err) {
                console.error('Failed to load failures:', err);
            } finally {
                this.isLoadingFailures = false;
            }
        },

        // Debounced search input handler
        onSearchInput() {
            clearTimeout(this.debounceTimer);
            this.debounceTimer = setTimeout(() => {
                this.failuresPage = 1;
                this.loadFailures();
            }, 300);
        },

        onDomainChange() {
            this.failuresPage = 1;
            this.loadFailures();
        },

        // Click heatmap cell or top-pair button to filter failures
        filterByPair(expected, predicted) {
            this.activePairFilter = { expected, predicted };
            this.searchQuery = expected;
            this.failuresPage = 1;
            this.loadFailures();
            
            // Scroll down smoothly to failures explorer
            const el = document.getElementById('failures-explorer');
            if (el) {
                el.scrollIntoView({ behavior: 'smooth' });
            }
        },

        clearFilter() {
            this.activePairFilter = null;
            this.searchQuery = '';
            this.selectedDomain = '';
            this.failuresPage = 1;
            this.loadFailures();
        },

        // Pagination handlers
        prevPage() {
            if (this.failuresPage > 1) {
                this.failuresPage--;
                this.loadFailures();
            }
        },

        nextPage() {
            if (this.failuresPage < this.failuresTotalPages) {
                this.failuresPage++;
                this.loadFailures();
            }
        },

        goToPage(p) {
            if (p >= 1 && p <= this.failuresTotalPages) {
                this.failuresPage = p;
                this.loadFailures();
            }
        }
    }));
});

