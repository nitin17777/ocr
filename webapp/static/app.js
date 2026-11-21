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
});
