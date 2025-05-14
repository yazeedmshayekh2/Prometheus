console.log('app.js starting to load');

class InsuranceAssistant {
    constructor() {
        console.log('InsuranceAssistant constructor starting');
        
        // Get all elements and log their existence
        this.nationalIdInput = document.getElementById('nationalId');
        this.questionInput = document.getElementById('questionInput');
        this.submitBtn = document.getElementById('submitBtn');
        this.loadingIndicator = document.getElementById('loadingIndicator');
        this.responseContent = document.getElementById('responseContent');
        this.answerSection = document.getElementById('answer');
        this.suggestedQuestions = document.getElementById('suggestedQuestions');

        console.log('DOM Elements found:', {
            nationalIdInput: !!this.nationalIdInput,
            questionInput: !!this.questionInput,
            submitBtn: !!this.submitBtn,
            loadingIndicator: !!this.loadingIndicator,
            responseContent: !!this.responseContent,
            answerSection: !!this.answerSection,
            suggestedQuestions: !!this.suggestedQuestions
        });

        this.setupEventListeners();
        console.log('Event listeners set up');

        const initialNationalId = this.nationalIdInput.value.trim();
        if (initialNationalId) {
            console.log('Initial National ID found:', initialNationalId);
            this.handleNationalIdChange();
        }
    }

    setupEventListeners() {
        this.submitBtn.addEventListener('click', () => this.handleSubmit());
        
        // Enable/disable submit button based on input
        const validateInputs = () => {
            const nationalId = this.nationalIdInput.value.trim();
            const question = this.questionInput.value.trim();
            this.submitBtn.disabled = !nationalId || !question;
        };

        this.nationalIdInput.addEventListener('input', validateInputs);
        this.questionInput.addEventListener('input', validateInputs);
        
        // Handle National ID input - modify these event listeners
        this.nationalIdInput.addEventListener('input', () => {
            const nationalId = this.nationalIdInput.value.trim();
            if (nationalId.length >= 8) { // Only trigger after reasonable length
                this.handleNationalIdChange();
            } else {
                this.hideSuggestedQuestions();
            }
        });
        
        // Initialize button state
        validateInputs();
    }

    async handleNationalIdChange() {
        const nationalId = this.nationalIdInput.value.trim();
        console.log('Handling National ID change:', nationalId); // Debug log

        if (nationalId) {
            try {
                this.showLoading(true);
                console.log('Fetching suggestions...'); // Debug log
                const suggestions = await this.getSuggestedQuestions(nationalId);
                console.log('Received suggestions:', suggestions); // Debug log
                this.displaySuggestedQuestions(suggestions);
                this.showLoading(false);
            } catch (error) {
                console.error('Error getting suggestions:', error);
                this.showLoading(false);
                this.hideSuggestedQuestions();
            }
        } else {
            this.hideSuggestedQuestions();
        }
    }

    async getSuggestedQuestions(nationalId) {
        try {
            console.log('Making suggestions API call...'); // Debug log
            const response = await fetch('/api/suggestions', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                },
                body: JSON.stringify({ national_id: nationalId })
            });

            console.log('API Response status:', response.status); // Debug log

            if (!response.ok) {
                const errorData = await response.json().catch(() => null);
                console.error('API Error response:', errorData); // Debug log
                throw new Error(errorData?.detail || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            console.log('API Response data:', data); // Debug log
            return data;
        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    }

    displaySuggestedQuestions(suggestions) {
        console.log('displaySuggestedQuestions starting, answerSection exists:', !!this.answerSection);
        console.log('displaySuggestedQuestions called with:', suggestions);
        
        // Log element states at start of display
        console.log('Element states in displaySuggestedQuestions:', {
            suggestedQuestions: !!this.suggestedQuestions,
            questionInput: !!this.questionInput,
            submitBtn: !!this.submitBtn,
            answerSection: !!this.answerSection
        });

        if (!this.suggestedQuestions) {
            console.error('Suggested questions container not found');
            return;
        }

        const bubblesContainer = this.suggestedQuestions.querySelector('.question-bubbles');
        if (!bubblesContainer) {
            console.error('Question bubbles container not found');
            return;
        }

        if (!suggestions || !suggestions.questions || suggestions.questions.length === 0) {
            console.log('No suggestions to display'); // Debug log
            this.hideSuggestedQuestions();
            return;
        }

        bubblesContainer.innerHTML = '';

        suggestions.questions.forEach((question, index) => {
            console.log(`Creating bubble for question ${index + 1}:`, question); // Debug log
            const bubble = document.createElement('div');
            bubble.className = 'question-bubble';
            bubble.textContent = question;
            bubble.addEventListener('click', async (e) => {
                e.preventDefault();
                console.log(`Clicked question ${index + 1}:`, question); // Debug log
                
                // Ensure response structure exists
                if (!this.answerSection) {
                    this.createResponseStructure();
                }
                
                if (this.questionInput) {
                    this.questionInput.value = question;
                    // Add a small delay to ensure the value is set and structure is created
                    await new Promise(resolve => setTimeout(resolve, 50));
                    if (this.submitBtn && !this.submitBtn.disabled) {
                        await this.handleSubmit();
                    }
                }
            });
            bubblesContainer.appendChild(bubble);
        });

        this.suggestedQuestions.classList.remove('hidden');
        console.log('Suggestions displayed, total bubbles:', bubblesContainer.children.length); // Debug log
    }

    hideSuggestedQuestions() {
        console.log('Hiding suggestions'); // Debug log
        this.suggestedQuestions.classList.add('hidden');
    }

    async handleSubmit() {
        console.log('handleSubmit starting, answerSection exists:', !!this.answerSection);
        
        if (!this.answerSection) {
            console.warn('Answer section is null, attempting to find it');
            this.answerSection = document.getElementById('answer');
            console.log('After retry, answerSection exists:', !!this.answerSection);
        }

        const nationalId = this.nationalIdInput?.value?.trim();
        const question = this.questionInput?.value?.trim();

        if (!nationalId || !question) {
            console.log('Missing input - nationalId:', !!nationalId, 'question:', !!question);
            this.showError('Please enter both National ID and your question.');
            return;
        }

        try {
            this.showLoading(true);
            const response = await this.queryAPI(nationalId, question);
            this.showLoading(false);
            this.displayResponse(response);
        } catch (error) {
            console.error('Error in handleSubmit:', error);
            this.showLoading(false);
            this.showError('An error occurred while processing your request.');
        }
    }

    async queryAPI(nationalId, question) {
        try {
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                },
                body: JSON.stringify({
                    national_id: nationalId,
                    question: question
                })
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => null);
                throw new Error(errorData?.detail || `HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    }

    showLoading(show) {
        this.loadingIndicator.classList.toggle('hidden', !show);
        this.responseContent.classList.toggle('hidden', show);
        this.submitBtn.disabled = show;
    }

    displayResponse(response) {
        if (!this.answerSection) {
            console.error('Answer section element not found in displayResponse');
            return;
        }

        // Display the answer with improved markdown formatting
        if (response.answer) {
            try {
                this.answerSection.innerHTML = `
                    <div class="markdown">
                        ${this.markdownToHtml(response.answer)}
                    </div>
                `;
            } catch (error) {
                console.error('Error setting answer HTML:', error);
            }
        } else {
            try {
                this.answerSection.innerHTML = '<div class="error-message">No answer received</div>';
            } catch (error) {
                console.error('Error setting no-answer HTML:', error);
            }
        }
    }

    showError(message) {
        console.log('showError starting, message:', message);
        console.log('answerSection exists:', !!this.answerSection);
        
        if (!this.answerSection) {
            console.warn('Answer section missing in showError, attempting to find it');
            this.answerSection = document.getElementById('answer');
            
            if (!this.answerSection) {
                console.error('Still cannot find answer section, creating fallback');
                const fallbackError = document.createElement('div');
                fallbackError.id = 'fallback-error';
                fallbackError.className = 'error-message';
                fallbackError.textContent = message;
                
                // Try to insert after question input
                if (this.questionInput?.parentNode) {
                    this.questionInput.parentNode.insertBefore(fallbackError, this.questionInput.nextSibling);
                } else {
                    document.body.appendChild(fallbackError);
                }
                return;
            }
        }

        try {
            this.answerSection.innerHTML = `
                <div class="error-message">
                    ${this.escapeHtml(message)}
                </div>
            `;
        } catch (error) {
            console.error('Error setting error message:', error);
            alert(message); // Fallback to alert if all else fails
        }
    }

    markdownToHtml(markdown) {
        if (!markdown) return '';
        
        // Process the markdown in multiple steps
        let html = markdown
            // Headers
            .replace(/^### (.*?)$/gm, '<h3>$1</h3>')
            .replace(/^## (.*?)$/gm, '<h2>$1</h2>')
            
            // Bold text (handle both ** and __ syntax)
            .replace(/(\*\*|__)(.*?)\1/g, '<strong>$2</strong>')
            
            // Horizontal rules
            .replace(/^---+$/gm, '<hr>')
            
            // Lists - first convert all forms of bullets to a common format
            .replace(/^\s*\* (.+)$/gm, '<li>$1</li>') // Asterisk bullets
            .replace(/^\s*- (.+)$/gm, '<li>$1</li>')  // Hyphen bullets 
            .replace(/^\s*• (.+)$/gm, '<li>$1</li>')  // Bullet character
            .replace(/^\s*\d+\.\s+(.+)$/gm, '<li>$1</li>') // Numbered lists
            
            // Currency formatting
            .replace(/QR\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)/g, '<strong>QR $1</strong>')
            
            // Percentage formatting
            .replace(/(\d+)%/g, '<strong>$1%</strong>')
            
            // Paragraphs
            .replace(/\n{2,}/g, '</p><p>')
            .replace(/^(.+?)$/gm, function(match) {
                if (!/^<[h|p|ul|li|hr]/.test(match)) {
                    return '<p>' + match + '</p>';
                }
                return match;
            });
            
        // Properly wrap lists in <ul> tags - this is more complex and needs a separate step
        // Find consecutive <li> elements and wrap them in <ul> tags
        let inList = false;
        let lines = html.split('\n');
        let result = [];
        
        for (let i = 0; i < lines.length; i++) {
            let line = lines[i];
            
            if (line.startsWith('<li>')) {
                if (!inList) {
                    result.push('<ul>');
                    inList = true;
                }
                result.push(line);
            } else {
                if (inList) {
                    result.push('</ul>');
                    inList = false;
                }
                result.push(line);
            }
        }
        
        if (inList) {
            result.push('</ul>');
        }
        
        html = result.join('\n')
            // Clean up empty paragraphs and fix nested paragraphs
            .replace(/<p>\s*<\/p>/g, '')
            .replace(/<p>(\s*<(?:ul|li|h[2-3]|hr)>)/g, '$1')
            .replace(/(<\/(?:ul|li|h[2-3]|hr)>\s*)<\/p>/g, '$1')
            
            // Fix line breaks
            .replace(/\n/g, '<br>');
            
        return html;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    createResponseStructure() {
        console.log('createResponseStructure called');
        console.log('Initial element states:', {
            responseContent: !!this.responseContent,
            answerSection: !!this.answerSection
        });

        // Only create if elements don't exist
        if (!this.responseContent || !this.answerSection) {
            const responseContainer = document.querySelector('.response-container');
            console.log('Response container found:', !!responseContainer);
            if (!responseContainer) {
                console.error('Response container not found, creating structure');
                const main = document.querySelector('main');
                if (main) {
                    const container = document.createElement('div');
                    container.className = 'response-container';
                    
                    const loadingIndicator = document.createElement('div');
                    loadingIndicator.id = 'loadingIndicator';
                    loadingIndicator.className = 'loading-indicator hidden';
                    loadingIndicator.innerHTML = `
                        <div class="spinner"></div>
                        <p>Processing user's policies...</p>
                    `;
                    
                    const responseContent = document.createElement('div');
                    responseContent.id = 'responseContent';
                    responseContent.className = 'response-content';
                    
                    const answerSection = document.createElement('div');
                    answerSection.id = 'answer';
                    answerSection.className = 'answer-section';
                    
                    responseContent.appendChild(answerSection);
                    container.appendChild(loadingIndicator);
                    container.appendChild(responseContent);
                    main.appendChild(container);
                    
                    // Update the references
                    this.responseContent = responseContent;
                    this.loadingIndicator = loadingIndicator;
                    this.answerSection = answerSection;
                }
            } else {
                // If container exists but elements don't, create them
                if (!this.loadingIndicator) {
                    this.loadingIndicator = responseContainer.querySelector('#loadingIndicator');
                }
                if (!this.responseContent) {
                    this.responseContent = responseContainer.querySelector('#responseContent');
                }
                if (!this.answerSection) {
                    this.answerSection = responseContainer.querySelector('#answer');
                }
            }
        }

        console.log('Final element states:', {
            responseContent: !!this.responseContent,
            answerSection: !!this.answerSection,
            loadingIndicator: !!this.loadingIndicator
        });
    }
}

console.log('About to initialize InsuranceAssistant');

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOMContentLoaded event fired');
    try {
        window.insuranceAssistant = new InsuranceAssistant();
        console.log('InsuranceAssistant initialized successfully');
    } catch (error) {
        console.error('Error initializing InsuranceAssistant:', error);
    }
});

console.log('app.js finished loading'); 