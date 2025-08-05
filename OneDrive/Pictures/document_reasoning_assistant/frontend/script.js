// Document Reasoning Assistant Frontend JavaScript

class DocumentReasoningApp {
    constructor() {
        this.sessionId = null;
        this.currentDocument = null;
        this.queryHistory = [];
        
        this.initializeApp();
    }
    
    async initializeApp() {
        this.bindEvents();
        await this.checkSystemHealth();
        await this.createSession();
        this.loadQueryHistory();
    }
    
    bindEvents() {
        // File upload events
        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('file-input');
        
        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
        uploadArea.addEventListener('drop', this.handleDrop.bind(this));
        
        fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        
        // Query form events
        const queryForm = document.getElementById('query-form');
        queryForm.addEventListener('submit', this.handleQuerySubmit.bind(this));
        
        // Details toggle event
        const detailsToggle = document.getElementById('details-toggle');
        detailsToggle.addEventListener('click', this.toggleDetails.bind(this));
        
        // History events
        const clearHistoryBtn = document.getElementById('clear-history');
        clearHistoryBtn.addEventListener('click', this.clearHistory.bind(this));
        
        // Keyboard shortcuts
        document.addEventListener('keydown', this.handleKeydown.bind(this));
    }
    
    async checkSystemHealth() {
        try {
            const response = await fetch('/health');
            const health = await response.json();
            
            this.updateStatusIndicator(health.status === 'healthy', health.status);
            
            if (health.status !== 'healthy') {
                this.showError('System health check failed. Some components may not be working properly.');
            }
        } catch (error) {
            console.error('Health check failed:', error);
            this.updateStatusIndicator(false, 'error');
            this.showError('Unable to connect to the server. Please check your connection.');
        }
    }
    
    async createSession() {
        try {
            const response = await fetch('/session', { method: 'POST' });
            const result = await response.json();
            this.sessionId = result.session_id;
        } catch (error) {
            console.error('Failed to create session:', error);
        }
    }
    
    updateStatusIndicator(healthy, status) {
        const indicator = document.getElementById('status-indicator');
        const dot = indicator.querySelector('div');
        const text = indicator.querySelector('span');
        
        if (healthy) {
            dot.className = 'w-2 h-2 bg-green-400 rounded-full';
            text.textContent = 'System Online';
            text.className = 'text-sm text-green-600';
        } else {
            dot.className = 'w-2 h-2 bg-red-400 rounded-full status-pulse';
            text.textContent = `System ${status || 'Offline'}`;
            text.className = 'text-sm text-red-600';
        }
    }
    
    // File upload handlers
    handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        document.getElementById('upload-area').classList.add('drag-over');
    }
    
    handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        document.getElementById('upload-area').classList.remove('drag-over');
    }
    
    handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        document.getElementById('upload-area').classList.remove('drag-over');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.uploadFile(files[0]);
        }
    }
    
    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.uploadFile(file);
        }
    }
    
    async uploadFile(file) {
        // Validate file type
        const allowedTypes = ['.pdf', '.docx', '.txt'];
        const fileExt = '.' + file.name.split('.').pop().toLowerCase();
        
        if (!allowedTypes.includes(fileExt)) {
            this.showError(`Unsupported file type: ${fileExt}. Please use PDF, DOCX, or TXT files.`);
            return;
        }
        
        // Show upload progress
        this.showUploadProgress(true);
        
        try {
            const formData = new FormData();
            formData.append('file', file);
            if (this.sessionId) {
                formData.append('session_id', this.sessionId);
            }
            
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`Upload failed: ${response.statusText}`);
            }
            
            const result = await response.json();
            this.currentDocument = result;
            
            this.showUploadSuccess(result);
            this.enableQuerying();
            
            this.showSuccess(`Document "${result.filename}" uploaded successfully! ${result.chunks_count} chunks processed.`);
            
        } catch (error) {
            console.error('Upload error:', error);
            this.showError(`Upload failed: ${error.message}`);
        } finally {
            this.showUploadProgress(false);
        }
    }
    
    showUploadProgress(show) {
        const progressDiv = document.getElementById('upload-progress');
        const uploadArea = document.getElementById('upload-area');
        
        if (show) {
            progressDiv.classList.remove('hidden');
            uploadArea.style.opacity = '0.5';
            uploadArea.style.pointerEvents = 'none';
        } else {
            progressDiv.classList.add('hidden');
            uploadArea.style.opacity = '1';
            uploadArea.style.pointerEvents = 'auto';
        }
    }
    
    showUploadSuccess(documentInfo) {
        const infoDiv = document.getElementById('document-info');
        const detailsDiv = document.getElementById('doc-details');
        
        detailsDiv.innerHTML = `
            <p><strong>File:</strong> ${documentInfo.filename}</p>
            <p><strong>Chunks:</strong> ${documentInfo.chunks_count}</p>
            <p><strong>Type:</strong> ${documentInfo.metadata.file_type?.toUpperCase() || 'Unknown'}</p>
            ${documentInfo.metadata.total_pages ? `<p><strong>Pages:</strong> ${documentInfo.metadata.total_pages}</p>` : ''}
        `;
        
        infoDiv.classList.remove('hidden');
    }
    
    enableQuerying() {
        const queryInput = document.getElementById('query-input');
        const askButton = document.getElementById('ask-button');
        const helpText = document.querySelector('.text-gray-500');
        
        queryInput.disabled = false;
        askButton.disabled = false;
        helpText.textContent = 'Enter your question about the document';
        
        queryInput.focus();
    }
    
    // Query handling
    async handleQuerySubmit(e) {
        e.preventDefault();
        
        const queryInput = document.getElementById('query-input');
        const query = queryInput.value.trim();
        
        if (!query) {
            this.showError('Please enter a question.');
            return;
        }
        
        if (!this.currentDocument) {
            this.showError('Please upload a document first.');
            return;
        }
        
        this.setQueryLoading(true);
        
        try {
            const response = await fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query: query,
                    session_id: this.sessionId
                })
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Query failed');
            }
            
            const result = await response.json();
            this.displayQueryResult(query, result);
            this.addToHistory(query, result);
            
            // Clear input and prompt for next question
            queryInput.value = '';
            this.promptForNextQuestion();
            
        } catch (error) {
            console.error('Query error:', error);
            this.showError(`Query failed: ${error.message}`);
        } finally {
            this.setQueryLoading(false);
        }
    }
    
    setQueryLoading(loading) {
        const askButton = document.getElementById('ask-button');
        const buttonText = document.getElementById('ask-button-text');
        const spinner = document.getElementById('ask-button-spinner');
        
        askButton.disabled = loading;
        
        if (loading) {
            buttonText.textContent = 'Processing...';
            spinner.classList.remove('hidden');
        } else {
            buttonText.textContent = 'Ask Question';
            spinner.classList.add('hidden');
        }
    }
    
    displayQueryResult(query, result) {
        const responseSection = document.getElementById('response-section');
        const directAnswer = document.getElementById('direct-answer');
        const decisionBadge = document.getElementById('decision-badge');
        const processingTime = document.getElementById('processing-time');
        const additionalInfoSection = document.getElementById('additional-info-section');
        const additionalInfo = document.getElementById('additional-info');
        const justification = document.getElementById('justification');
        const referencedClauses = document.getElementById('referenced-clauses');
        const detailsContent = document.getElementById('details-content');
        
        // Show response section
        responseSection.classList.remove('hidden');
        
        // Set direct answer (prominently displayed)
        directAnswer.textContent = result.direct_answer || "Unable to provide a direct answer.";
        
        // Set decision badge
        const decisionClass = `decision-${result.decision.toLowerCase()}`;
        decisionBadge.className = `px-3 py-1 rounded-full text-sm font-medium ${decisionClass}`;
        decisionBadge.textContent = result.decision;
        
        // Set processing time
        processingTime.textContent = `Processed in ${result.processing_time.toFixed(2)}s`;
        
        // Set additional info if available
        if (result.additional_info && result.additional_info.trim()) {
            additionalInfo.textContent = result.additional_info;
            additionalInfoSection.classList.remove('hidden');
        } else {
            additionalInfoSection.classList.add('hidden');
        }
        
        // Set justification
        justification.textContent = result.justification;
        
        // Set referenced clauses
        if (result.referenced_clauses && result.referenced_clauses.length > 0) {
            referencedClauses.innerHTML = result.referenced_clauses.map(clause => `
                <div class="clause-card bg-white border rounded-lg p-4">
                    <div class="flex justify-between items-start mb-2">
                        <h4 class="font-medium text-gray-900">${clause.clause_id}</h4>
                        <span class="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">Referenced</span>
                    </div>
                    <p class="text-sm text-gray-600 mb-2 leading-relaxed">${clause.text}</p>
                    <p class="text-xs text-blue-600 italic">${clause.reasoning}</p>
                </div>
            `).join('');
        } else {
            referencedClauses.innerHTML = '<p class="text-sm text-gray-500">No specific clauses referenced.</p>';
        }
        
        // Ensure details are collapsed by default
        detailsContent.classList.add('hidden');
        const chevron = document.getElementById('details-chevron');
        chevron.classList.remove('rotate-180');
        
        // Scroll to response
        responseSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    
    toggleDetails() {
        const detailsContent = document.getElementById('details-content');
        const chevron = document.getElementById('details-chevron');
        
        if (detailsContent.classList.contains('hidden')) {
            detailsContent.classList.remove('hidden');
            chevron.classList.add('rotate-180');
        } else {
            detailsContent.classList.add('hidden');
            chevron.classList.remove('rotate-180');
        }
    }
    
    promptForNextQuestion() {
        const queryInput = document.getElementById('query-input');
        const helpText = document.querySelector('p.text-sm.text-gray-500');
        
        // Update help text with encouraging message
        const encouragingMessages = [
            "Great! Got another question about the document?",
            "What else would you like to know?",
            "Feel free to ask another question!",
            "Any other questions about the document?",
            "Ready for your next question!",
            "What's your next question?"
        ];
        
        const randomMessage = encouragingMessages[Math.floor(Math.random() * encouragingMessages.length)];
        helpText.textContent = randomMessage;
        
        // Focus on the input field for user convenience
        setTimeout(() => {
            queryInput.focus();
            queryInput.placeholder = "Ask another question...";
        }, 500);
        
        // Reset help text after a few seconds if user doesn't interact
        setTimeout(() => {
            if (queryInput.value === '' && document.activeElement !== queryInput) {
                helpText.textContent = 'Enter your question about the document';
                queryInput.placeholder = 'e.g., Can I claim air ambulance if the hospital is 200 km away?';
            }
        }, 8000);
        
        // Show a subtle animation or highlight to draw attention
        queryInput.classList.add('ring-2', 'ring-blue-300', 'ring-opacity-50');
        setTimeout(() => {
            queryInput.classList.remove('ring-2', 'ring-blue-300', 'ring-opacity-50');
        }, 2000);
    }
    
    // History management
    addToHistory(query, result) {
        const historyItem = {
            query,
            decision: result.decision,
            timestamp: new Date(),
            processing_time: result.processing_time
        };
        
        this.queryHistory.unshift(historyItem);
        this.updateHistoryDisplay();
        this.saveHistoryToStorage();
    }
    
    updateHistoryDisplay() {
        const historyContainer = document.getElementById('history-container');
        
        if (this.queryHistory.length === 0) {
            historyContainer.innerHTML = '<p class="text-sm text-gray-500 text-center py-8">No queries yet</p>';
            return;
        }
        
        historyContainer.innerHTML = this.queryHistory.map(item => {
            const timeAgo = this.getTimeAgo(item.timestamp);
            const decisionClass = item.decision.toLowerCase();
            
            return `
                <div class="history-item ${decisionClass} bg-white border rounded-lg p-4 cursor-pointer" 
                     onclick="app.repeatQuery('${item.query.replace(/'/g, "\\'")}')"
                >
                    <div class="flex justify-between items-start mb-2">
                        <span class="text-sm font-medium text-gray-900 text-truncate flex-1 mr-2">
                            ${item.query}
                        </span>
                        <span class="text-xs text-gray-500 whitespace-nowrap">${timeAgo}</span>
                    </div>
                    <div class="flex justify-between items-center">
                        <span class="text-xs px-2 py-1 rounded-full decision-${decisionClass}">
                            ${item.decision}
                        </span>
                        <span class="text-xs text-gray-400">
                            ${item.processing_time.toFixed(2)}s
                        </span>
                    </div>
                </div>
            `;
        }).join('');
    }
    
    repeatQuery(query) {
        const queryInput = document.getElementById('query-input');
        queryInput.value = query;
        queryInput.focus();
    }
    
    clearHistory() {
        this.queryHistory = [];
        this.updateHistoryDisplay();
        this.saveHistoryToStorage();
        this.showSuccess('Query history cleared.');
    }
    
    loadQueryHistory() {
        try {
            const saved = localStorage.getItem('documentReasoningHistory');
            if (saved) {
                this.queryHistory = JSON.parse(saved).map(item => ({
                    ...item,
                    timestamp: new Date(item.timestamp)
                }));
                this.updateHistoryDisplay();
            }
        } catch (error) {
            console.error('Failed to load history:', error);
        }
    }
    
    saveHistoryToStorage() {
        try {
            localStorage.setItem('documentReasoningHistory', JSON.stringify(this.queryHistory.slice(0, 50))); // Keep last 50
        } catch (error) {
            console.error('Failed to save history:', error);
        }
    }
    
    // Utility functions
    getTimeAgo(date) {
        const now = new Date();
        const diff = now - date;
        const minutes = Math.floor(diff / 60000);
        const hours = Math.floor(diff / 3600000);
        const days = Math.floor(diff / 86400000);
        
        if (minutes < 1) return 'Just now';
        if (minutes < 60) return `${minutes}m ago`;
        if (hours < 24) return `${hours}h ago`;
        return `${days}d ago`;
    }
    
    handleKeydown(e) {
        // Ctrl/Cmd + Enter to submit query
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            const queryForm = document.getElementById('query-form');
            queryForm.dispatchEvent(new Event('submit'));
        }
        
        // Escape to clear focus
        if (e.key === 'Escape') {
            document.activeElement.blur();
        }
    }
    
    // Toast notifications
    showError(message) {
        this.showToast('error', message);
    }
    
    showSuccess(message) {
        this.showToast('success', message);
    }
    
    showToast(type, message) {
        const toastId = type === 'error' ? 'error-toast' : 'success-toast';
        const messageId = type === 'error' ? 'error-message' : 'success-message';
        
        const toast = document.getElementById(toastId);
        const messageEl = document.getElementById(messageId);
        
        messageEl.textContent = message;
        toast.classList.remove('hidden');
        toast.classList.add('toast-enter-active');
        
        setTimeout(() => {
            toast.classList.add('toast-exit-active');
            setTimeout(() => {
                toast.classList.add('hidden');
                toast.classList.remove('toast-enter-active', 'toast-exit-active');
            }, 300);
        }, 3000);
    }
}

// Initialize app when DOM is loaded
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new DocumentReasoningApp();
});

// Export for global access
window.DocumentReasoningApp = DocumentReasoningApp;
