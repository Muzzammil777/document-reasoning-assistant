// Analysis Page JavaScript

class AnalysisPage {
    constructor() {
        this.currentAnalysis = null;
        this.initializePage();
    }
    
    initializePage() {
        this.bindEvents();
        this.loadAnalysisData();
    }
    
    bindEvents() {
        // Back button navigation
        const backButton = document.getElementById('back-button');
        backButton.addEventListener('click', this.goBack.bind(this));
        
        // Handle browser back button
        window.addEventListener('popstate', this.goBack.bind(this));
    }
    
    loadAnalysisData() {
        try {
            // Get analysis data from sessionStorage (passed from main page)
            const analysisData = sessionStorage.getItem('currentAnalysis');
            
            if (!analysisData) {
                this.showError('No analysis data found. Redirecting to main page...');
                setTimeout(() => this.goBack(), 2000);
                return;
            }
            
            this.currentAnalysis = JSON.parse(analysisData);
            this.displayAnalysis();
            
        } catch (error) {
            console.error('Error loading analysis data:', error);
            this.showError('Error loading analysis data. Redirecting to main page...');
            setTimeout(() => this.goBack(), 2000);
        }
    }
    
    displayAnalysis() {
        const { query, result } = this.currentAnalysis;
        
        // Display original query
        const originalQuery = document.getElementById('original-query');
        originalQuery.textContent = query;
        
        // Display direct answer
        const directAnswer = document.getElementById('analysis-direct-answer');
        directAnswer.textContent = result.direct_answer || "Unable to provide a direct answer.";
        
        // Display decision badge
        const decisionBadge = document.getElementById('analysis-decision-badge');
        const decisionClass = `decision-${result.decision.toLowerCase()}`;
        decisionBadge.className = `px-3 py-1 rounded-full text-sm font-medium ${decisionClass}`;
        decisionBadge.textContent = result.decision;
        
        // Display processing time
        const processingTime = document.getElementById('analysis-processing-time');
        processingTime.textContent = `Processed in ${result.processing_time.toFixed(2)}s`;
        
        // Display additional info if available
        if (result.additional_info && result.additional_info.trim()) {
            const additionalInfoSection = document.getElementById('analysis-additional-info-section');
            const additionalInfo = document.getElementById('analysis-additional-info');
            additionalInfo.textContent = result.additional_info;
            additionalInfoSection.classList.remove('hidden');
        }
        
        // Display justification
        const justification = document.getElementById('analysis-justification');
        justification.textContent = result.justification;
        
        // Display referenced clauses
        const referencedClauses = document.getElementById('analysis-referenced-clauses');
        if (result.referenced_clauses && result.referenced_clauses.length > 0) {
            referencedClauses.innerHTML = result.referenced_clauses.map(clause => `
                <div class="clause-card bg-white border rounded-lg p-6 shadow-sm">
                    <div class="flex justify-between items-start mb-3">
                        <h4 class="font-semibold text-gray-900 text-lg">${clause.clause_id}</h4>
                        <span class="text-xs text-gray-500 bg-gray-100 px-3 py-1 rounded-full">Referenced</span>
                    </div>
                    <p class="text-gray-700 mb-4 leading-relaxed">${clause.text}</p>
                    <div class="border-t pt-3">
                        <p class="text-sm text-blue-600 italic font-medium">Reasoning:</p>
                        <p class="text-sm text-blue-700 mt-1">${clause.reasoning}</p>
                    </div>
                </div>
            `).join('');
        } else {
            referencedClauses.innerHTML = '<p class="text-gray-500 italic text-center py-8">No specific clauses referenced in this analysis.</p>';
        }
    }
    
    goBack() {
        // Clear the analysis data from session storage
        sessionStorage.removeItem('currentAnalysis');
        
        // Navigate back to main page
        window.location.href = 'index.html';
    }
    
    showError(message) {
        const toast = document.getElementById('error-toast');
        const messageEl = document.getElementById('error-message');
        
        messageEl.textContent = message;
        toast.classList.remove('hidden');
        
        setTimeout(() => {
            toast.classList.add('hidden');
        }, 5000);
    }
}

// Initialize the analysis page when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new AnalysisPage();
});
