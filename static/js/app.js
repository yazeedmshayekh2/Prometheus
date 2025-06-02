console.log('app.js starting to load');

class InsuranceAssistant {
    constructor() {
        console.log('InsuranceAssistant constructor starting');
        
        // Get all elements and log their existence
        this.nationalIdInput = document.getElementById('nationalId');
        this.nationalIdValidation = document.getElementById('nationalIdValidation');
        this.confirmNationalIdBtn = document.getElementById('confirmNationalId');
        this.questionInput = document.getElementById('questionInput');
        this.submitBtn = document.getElementById('submitBtn');
        this.loadingIndicator = document.getElementById('loadingIndicator');
        this.responseContent = document.getElementById('responseContent');
        this.answerSection = document.getElementById('answer');
        this.suggestedQuestions = document.getElementById('suggestedQuestions');
        this.themeToggle = document.getElementById('themeToggle');
        
        // PDF viewer elements
        this.pdfViewer = document.getElementById('pdfViewer');
        this.pdfFrame = document.getElementById('pdfFrame');
        this.pdfCompany = document.getElementById('pdfCompany');
        this.pdfPlaceholder = document.getElementById('pdfPlaceholder');
        
        // Track PDF state
        this.isPdfLoaded = false;
        this.currentPdfLink = null;
        
        // Track National ID state
        this.isNationalIdConfirmed = false;
        this.currentNationalId = '';

        // User info elements
        this.contractorName = document.getElementById('contractorName');
        this.expiryDate = document.getElementById('expiryDate');
        this.beneficiaryCount = document.getElementById('beneficiaryCount');

        // Add chat history for multi-turn conversation
        this.chatHistory = [];

        console.log('DOM Elements found:', {
            nationalIdInput: !!this.nationalIdInput,
            nationalIdValidation: !!this.nationalIdValidation,
            confirmNationalIdBtn: !!this.confirmNationalIdBtn,
            questionInput: !!this.questionInput,
            submitBtn: !!this.submitBtn,
            loadingIndicator: !!this.loadingIndicator,
            responseContent: !!this.responseContent,
            answerSection: !!this.answerSection,
            suggestedQuestions: !!this.suggestedQuestions,
            themeToggle: !!this.themeToggle,
            pdfViewer: !!this.pdfViewer,
            pdfFrame: !!this.pdfFrame,
            pdfCompany: !!this.pdfCompany,
            pdfPlaceholder: !!this.pdfPlaceholder
        });

        // Initialize theme
        this.initializeTheme();

        this.setupEventListeners();
        console.log('Event listeners set up');

        const initialNationalId = this.nationalIdInput.value.trim();
        if (initialNationalId) {
            console.log('Initial National ID found:', initialNationalId);
            this.validateNationalIdInput(initialNationalId);
        }
    }

    setupEventListeners() {
        // Theme toggle
        if (this.themeToggle) {
            this.themeToggle.addEventListener('click', () => this.toggleTheme());
        }
        
        // National ID input validation
        if (this.nationalIdInput) {
            this.nationalIdInput.addEventListener('input', (e) => {
                const numbersOnly = e.target.value.replace(/\D/g, '');
                e.target.value = numbersOnly;
                this.validateNationalIdInput(numbersOnly);
            });
            
            this.nationalIdInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    this.handleNationalIdConfirm();
                }
            });
        }
        
        // Confirm National ID button
        if (this.confirmNationalIdBtn) {
            this.confirmNationalIdBtn.addEventListener('click', () => this.handleNationalIdConfirm());
        }

        // Question input handlers
        if (this.questionInput) {
            // Add input event listener to update button state
            this.questionInput.addEventListener('input', () => {
                this.updateSubmitButtonState();
            });
            
            this.questionInput.addEventListener('keypress', async (e) => {
                if (e.key === 'Enter' && !e.shiftKey && !this.isLoading) {
                    e.preventDefault();
                    await this.handleQuestionSubmit();
                }
            });
        }

        if (this.submitBtn) {
            this.submitBtn.addEventListener('click', async (e) => {
                e.preventDefault();
                if (!this.isLoading) {
                    await this.handleQuestionSubmit();
                }
            });
        }

        // Initialize button state
        this.updateSubmitButtonState();
    }

    updateSubmitButtonState() {
        if (!this.submitBtn) return;

        const hasQuestion = this.questionInput && this.questionInput.value.trim().length > 0;
        const canSubmit = this.isNationalIdConfirmed && hasQuestion && !this.isLoading;

        this.submitBtn.disabled = !canSubmit;
        
        // Update button appearance
        if (canSubmit) {
            this.submitBtn.style.backgroundColor = '#0d6efd';
            this.submitBtn.style.color = '#fff';
            this.submitBtn.style.cursor = 'pointer';
        } else {
            this.submitBtn.style.backgroundColor = '#ccc';
            this.submitBtn.style.color = '#212529';
            this.submitBtn.style.cursor = 'not-allowed';
        }
    }

    validateNationalIdInput(value) {
        const length = value.length;
        
        // Clear previous validation state
        this.nationalIdInput.classList.remove('valid', 'invalid');
        this.nationalIdValidation.classList.remove('valid', 'invalid', 'progress', 'hidden');
        this.confirmNationalIdBtn.classList.remove('pulse');
        this.confirmNationalIdBtn.classList.add('hidden');
        
        // Check if ID has changed
        if (this.currentNationalId !== value) {
            this.isNationalIdConfirmed = false;
            this.clearUserInfo();
            
            // Clear all results including response container
            this.clearResults(false, '', true);
            
            // Reset policy document view to default state
            if (this.pdfPlaceholder) {
                this.pdfPlaceholder.classList.remove('hidden');
                this.pdfPlaceholder.innerHTML = `
                    <div class="pdf-placeholder-icon">📄</div>
                    <div class="pdf-placeholder-text">
                        <strong>Policy Document</strong><br>
                        Enter your National ID to view your policy documents
                    </div>
                `;
            }
            if (this.pdfViewer) {
                this.pdfViewer.classList.add('hidden');
            }
            if (this.pdfFrame) {
                this.pdfFrame.src = '';
            }
            if (this.suggestedQuestions) {
                this.suggestedQuestions.classList.add('hidden');
            }
            if (this.answerSection) {
                this.answerSection.innerHTML = '';
            }
            // Hide family information section
            const familyInfoSection = document.getElementById('familyInfoSection');
            if (familyInfoSection) {
                familyInfoSection.classList.add('hidden');
            }
        }
        
        if (length === 0) {
            // Empty input - just hide validation message
            this.nationalIdValidation.classList.add('hidden');
            this.isNationalIdConfirmed = false;
            return;
        }
        
        if (length < 11) {
            // Still typing - show neutral progress message
            this.nationalIdValidation.classList.add('progress');
            this.nationalIdValidation.innerHTML = `<span>⏳</span> Enter ${11 - length} more digit${11 - length === 1 ? '' : 's'} (${length}/11)`;
            this.isNationalIdConfirmed = false;
        } else if (length === 11) {
            // Exactly 11 digits - show enter button
            this.nationalIdInput.classList.add('valid');
            this.nationalIdValidation.classList.add('valid');
            this.nationalIdValidation.innerHTML = '<span>✅</span> 11 digits entered. Press Enter to confirm.';
            this.confirmNationalIdBtn.classList.remove('hidden');
            this.confirmNationalIdBtn.classList.add('pulse');
        } else {
            // More than 11 digits - show error
            this.nationalIdInput.classList.add('invalid');
            this.nationalIdValidation.classList.add('invalid');
            this.nationalIdValidation.innerHTML = '<span>❌</span> National ID must be exactly 11 digits';
            this.isNationalIdConfirmed = false;
        }
    }
    
    async handleNationalIdConfirm() {
        const nationalId = this.nationalIdInput.value.trim();
        
        if (nationalId.length !== 11) {
            this.showError('Please enter a valid 11-digit National ID.');
            return;
        }

        try {
            // First get suggestions which includes family data
            const suggestionsResponse = await fetch('/api/suggestions', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    national_id: nationalId
                })
            });

            if (!suggestionsResponse.ok) {
                throw new Error('Failed to fetch user information');
            }

            const data = await suggestionsResponse.json();
            console.log('API Response:', data);
            
            // Mark as confirmed first
            this.isNationalIdConfirmed = true;
            this.currentNationalId = nationalId;
            
            // Show success message
            this.nationalIdValidation.innerHTML = '<span>✅</span> ID confirmed successfully';
            this.confirmNationalIdBtn.classList.remove('pulse');
            this.confirmNationalIdBtn.classList.add('hidden');

            // Update button state after confirmation
            this.updateSubmitButtonState();

            // Update the user info display
            if (data && data.family_data && data.family_data.members) {
                const members = data.family_data.members;
                console.log('Family members:', members);

                // Find the principal member (the one with the ID we entered)
                const principal = members.find(m => m.national_id === nationalId) || members[0];
                
                if (principal) {
                    // Update the display with principal's information
                    this.contractorName.textContent = principal.name || '-';
                    this.expiryDate.textContent = this.formatDate(principal.end_date) || '-';
                    this.beneficiaryCount.textContent = data.family_data.total_members.toString() || '-';
                } else {
                    this.clearUserInfo();
                }
            } else {
                this.clearUserInfo();
            }

            // Display suggested questions if available
            if (data.questions && data.questions.length > 0) {
                let suggestedQuestionsContainer = document.getElementById('suggestedQuestionsContainer');
                if (!suggestedQuestionsContainer) {
                    suggestedQuestionsContainer = document.createElement('div');
                    suggestedQuestionsContainer.id = 'suggestedQuestionsContainer';
                    suggestedQuestionsContainer.className = 'suggested-questions-container';
                    
                    // Insert after the user info section
                    const userInfoSection = document.querySelector('.user-info-section');
                    if (userInfoSection) {
                        userInfoSection.parentNode.insertBefore(suggestedQuestionsContainer, userInfoSection.nextSibling);
                    }
                }

                // Clear previous questions
                suggestedQuestionsContainer.innerHTML = `
                    <div class="suggested-questions-header">
                        <h3>Suggested Questions</h3>
                        <p>Click on a question to ask it</p>
                    </div>
                    <div class="suggested-questions-list"></div>
                `;

                const questionsList = suggestedQuestionsContainer.querySelector('.suggested-questions-list');
                
                // Add each question as a clickable button
                data.questions.forEach(question => {
                    // Clean the question text:
                    // 1. Remove [] brackets and their contents
                    // 2. Remove "Question #:" prefix
                    // 3. Remove any markdown formatting
                    // 4. Remove extra whitespace
                    let cleanQuestion = question
                        .replace(/\[\d*\]/g, '') // Remove [1], [2], etc.
                        .replace(/\*\*Question\s*(?:#?\d+|):\*\*\s*/i, '') // Remove "Question #:" or "Question:"
                        .replace(/\*\*/g, '') // Remove any remaining **bold** markdown
                        .replace(/^#+\s*/, '') // Remove markdown headers
                        .trim(); // Remove extra whitespace

                    const questionButton = document.createElement('button');
                    questionButton.className = 'suggested-question-btn';
                    questionButton.textContent = cleanQuestion;
                    questionButton.addEventListener('click', () => {
                        if (this.questionInput) {
                            this.questionInput.value = cleanQuestion;
                            this.questionInput.focus();
                        }
                    });
                    questionsList.appendChild(questionButton);
                });

                // Show the container
                suggestedQuestionsContainer.style.display = 'block';
            }

            // Enable/disable submit button based on question input
            this.submitBtn.disabled = !this.questionInput.value.trim();
            
        } catch (error) {
            console.error('Error:', error);
            this.showError('Failed to verify National ID. Please try again.');
            this.isNationalIdConfirmed = false;
            this.clearUserInfo();
        }
    }

    async handleQuestionSubmit() {
        const question = this.questionInput.value.trim();
        if (!question || !this.isNationalIdConfirmed || !this.currentNationalId) {
            console.log('Cannot submit: question empty or ID not confirmed');
            return;
        }

        try {
            console.log('Submitting question:', question);
            this.setLoading(true);
            
            // Create or ensure chat container exists
            if (!this.chatContainer) {
                const responseContainer = document.querySelector('.response-container');
                if (!responseContainer) {
                    console.error('Response container not found');
                    return;
                }
                this.chatContainer = responseContainer.querySelector('.chatcontainer');
                if (!this.chatContainer) {
                    this.chatContainer = document.createElement('div');
                    this.chatContainer.className = 'chatcontainer';
                    responseContainer.appendChild(this.chatContainer);
                }
            }

            // Show the response container
            const responseContainer = document.querySelector('.response-container');
            if (responseContainer) {
                responseContainer.classList.remove('hidden');
                responseContainer.classList.add('show');
            }
            
            // Add user message to chat
            this.addMessageToChat('user', question);
            
            // Clear input and update button state
            this.questionInput.value = '';
            this.updateSubmitButtonState();
            
            // Send to API
            console.log('Sending request to API:', {
                national_id: this.currentNationalId,
                question: question,
                chat_history: this.chatHistory
            });

            const response = await fetch('/api/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    national_id: this.currentNationalId,
                    question: question,
                    chat_history: this.chatHistory
                })
            });

            console.log('API Response status:', response.status);
            
            if (!response.ok) {
                const errorData = await response.json();
                console.error('API Error:', errorData);
                throw new Error(errorData.detail?.message || errorData.detail || `Error: ${response.status}`);
            }

            const data = await response.json();
            console.log('API Response data:', data);
            
            if (data?.answer) {
                // Add assistant message to chat
                this.addMessageToChat('assistant', data.answer);
            } else {
                console.error('No answer in response:', data);
                throw new Error('No answer received from server');
            }
        } catch (error) {
            console.error('Error submitting question:', error);
            this.showError(error.message || 'Failed to get response from server');
        } finally {
            this.setLoading(false);
            this.updateSubmitButtonState();
        }
    }

    addMessageToChat(role, content) {
        if (!content) {
            console.error('No content provided for chat message');
            return;
        }

        console.log(`Adding ${role} message:`, content);
        
        // Add to chat history
        this.chatHistory.push({ role, content });
        
        // Create message element
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat ${role}-message`;
        
        // Style based on role
        if (role === 'user') {
            messageDiv.style.cssText = `
                float: right;
                background-color: #e9e8e8;
                clear: both;
                margin-left: auto;
                max-width: 70%;
                margin-bottom: 10px;
                padding: 10px 15px;
                border-radius: 15px;
            `;
        } else {
            messageDiv.style.cssText = `
                float: left;
                background-color: #f2f2f2;
                clear: both;
                margin-right: auto;
                max-width: 70%;
                margin-bottom: 10px;
                padding: 10px 15px;
                border-radius: 15px;
            `;
        }

        // Set content with proper formatting
        messageDiv.innerHTML = role === 'assistant' 
            ? this.markdownToHtml(content)
            : this.escapeHtml(content);

        // Ensure chat container exists
        if (!this.chatContainer) {
            const responseContainer = document.querySelector('.response-container');
            if (!responseContainer) {
                console.error('Response container not found');
                return;
            }
            this.chatContainer = responseContainer.querySelector('.chatcontainer');
            if (!this.chatContainer) {
                this.chatContainer = document.createElement('div');
                this.chatContainer.className = 'chatcontainer';
                responseContainer.appendChild(this.chatContainer);
            }
        }
        
        // Add message to container
        this.chatContainer.appendChild(messageDiv);
        
        // Scroll to bottom
        this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
        
        console.log('Message added to chat');
    }

    setLoading(isLoading) {
        this.isLoading = isLoading;
        if (this.submitBtn) {
            this.submitBtn.disabled = isLoading;
            this.submitBtn.style.backgroundColor = isLoading ? '#ccc' : '#0d6efd';
            this.submitBtn.style.cursor = isLoading ? 'not-allowed' : 'pointer';
        }
        if (this.loadingIndicator) {
            this.loadingIndicator.classList.toggle('hidden', !isLoading);
        }
    }

    async handleNationalIdChange() {
        const nationalId = this.currentNationalId;
        console.log('Handling National ID change:', nationalId);

        if (!nationalId || !this.isNationalIdConfirmed) {
            // Clear previous results and show default state
            this.clearResults(false, '', true);
            return;
        }

        try {
            this.showNationalIdLoading(true);
            console.log('Fetching suggestions...');
            const suggestions = await this.getSuggestedQuestions(nationalId);
            console.log('Received suggestions:', suggestions);
            
            // Test family endpoint directly
            console.log('Testing family endpoint directly...');
            try {
                const familyTestResponse = await fetch('/api/test-family', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ national_id: nationalId })
                });
                const familyTestData = await familyTestResponse.json();
                console.log('Family test data:', familyTestData);
            } catch (err) {
                console.error('Family test error:', err);
            }
            
            // Clear any previous error states
            if (this.pdfPlaceholder) {
                this.pdfPlaceholder.classList.remove('error');
            }
            
            this.displaySuggestedQuestions(suggestions);
            
            // Display family information if available
            console.log('Checking for family data:', suggestions.family_data);
            if (suggestions.family_data && suggestions.family_data.members && suggestions.family_data.members.length > 0) {
                console.log('Displaying family information for', suggestions.family_data.members.length, 'members');
                this.displayFamilyInformation(suggestions.family_data);
            } else {
                console.log('No family data to display');
            }
            
            // Display PDF if available in suggestions response
            if (suggestions.pdf_info && suggestions.pdf_info.pdf_link) {
                await this.displayPDF(suggestions.pdf_info);
            } else {
                this.clearResults(false, '', true);
            }
            
        } catch (error) {
            console.error('Error getting suggestions:', error);
            this.clearResults(true, error.message);
        } finally {
            this.showNationalIdLoading(false);
        }
    }

    async getSuggestedQuestions(nationalId) {
        try {
            console.log('Making suggestions API call for ID:', nationalId);
            const response = await fetch('/api/suggestions', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                },
                body: JSON.stringify({ national_id: nationalId })
            });

            console.log('API Response status:', response.status);
            const data = await response.json();
            
            if (!response.ok) {
                console.error('API Error:', data);
                // Show error in the PDF placeholder
                if (this.pdfPlaceholder) {
                    this.pdfPlaceholder.classList.remove('hidden');
                    this.pdfPlaceholder.innerHTML = `
                        <div class="pdf-placeholder-icon">⚠️</div>
                        <div class="pdf-placeholder-text error">
                            <strong>Policy Document Unavailable</strong><br>
                            ${data.detail || 'Unable to access policy document'}
                        </div>
                    `;
                }
                if (this.pdfViewer) {
                    this.pdfViewer.classList.add('hidden');
                }
                throw new Error(data.detail || `Error: ${response.status} - ${response.statusText}`);
            }

            // Log the full response data
            console.log('API Response data:', {
                status: response.status,
                statusText: response.statusText,
                headers: Object.fromEntries(response.headers.entries()),
                data: data
            });

            // Show policy statistics if available
            if (data.total_policies > 0) {
                console.log(`Found ${data.total_policies} policies, ${data.valid_pdfs} with valid PDFs`);
                if (data.valid_pdfs === 0 && this.pdfPlaceholder) {
                    this.pdfPlaceholder.classList.remove('hidden');
                    this.pdfPlaceholder.innerHTML = `
                        <div class="pdf-placeholder-icon">📄</div>
                        <div class="pdf-placeholder-text">
                            <strong>Policy Found</strong><br>
                            Your policy is active but the document is not available in digital format.
                            Please contact support if you need a copy.
                        </div>
                    `;
                    if (this.pdfViewer) {
                        this.pdfViewer.classList.add('hidden');
                    }
                }
            }

            if (data.pdf_info) {
                console.log('PDF Info:', {
                    link: data.pdf_info.pdf_link,
                    company: data.pdf_info.company_name,
                    policyNumber: data.pdf_info.policy_number,
                    policyType: data.pdf_info.policy_type
                });
            } else {
                console.log('No PDF information available');
                if (this.pdfPlaceholder) {
                    this.pdfPlaceholder.classList.remove('hidden');
                    this.pdfPlaceholder.innerHTML = `
                        <div class="pdf-placeholder-icon">📄</div>
                        <div class="pdf-placeholder-text">
                            <strong>No PDF Available</strong><br>
                            The policy document is not available in digital format.
                            Please contact support if you need a copy.
                        </div>
                    `;
                }
                if (this.pdfViewer) {
                    this.pdfViewer.classList.add('hidden');
                }
            }

            return data;
        } catch (error) {
            console.error('API Error:', error);
            // Show error in the PDF placeholder if not already shown
            if (this.pdfPlaceholder && !this.pdfPlaceholder.querySelector('.error')) {
                this.pdfPlaceholder.classList.remove('hidden');
                this.pdfPlaceholder.innerHTML = `
                    <div class="pdf-placeholder-icon">⚠️</div>
                    <div class="pdf-placeholder-text error">
                        <strong>Error Loading Policy</strong><br>
                        ${error.message}
                    </div>
                `;
            }
            if (this.pdfViewer) {
                this.pdfViewer.classList.add('hidden');
            }
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
            // Remove [], the "Question #N:" prefix, any other bold markdown, and "### " prefixes for display
            bubble.textContent = question.replace(/[\[\]]/g, '')
                                       .replace(/\*\*Question\s*(#\d+|\d+):\*\*\s*/, '')
                                       .replace(/\*\*(.*?)\*\*/g, '$1')
                                       .replace(/^#+\s*/, ''); // Remove one or more "#" followed by space
            bubble.addEventListener('click', (e) => {
                e.preventDefault();
                console.log(`Clicked question ${index + 1}:`, question); // Debug log
                
                // Ensure response structure exists
                if (!this.answerSection) {
                    this.createResponseStructure();
                }
                
                if (this.questionInput) {
                    // MODIFIED: Apply full cleaning to the question text for the input field
                    const cleanedQuestionForInput = question.replace(/[\\[\\]]/g, '')
                                                         .replace(/\*\*Question\s*(#\d+|\d+):\*\*\s*/, '')
                                                         .replace(/\*\*(.*?)\*\*/g, '$1')
                                                         .replace(/^#+\s*/, ''); // Remove one or more "#" followed by space
                    this.questionInput.value = cleanedQuestionForInput;
                    // MODIFIED: Update button state to enable it if appropriate
                    this.questionInput.focus(); 
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
                    question: question,
                    chat_history: this.chatHistory
                })
            });

            const responseData = await response.json();

            if (!response.ok) {
                // Handle specific content filtering errors from the API
                if (responseData.detail && typeof responseData.detail === 'object' && responseData.detail.error === 'inappropriate_content_blocked') {
                    this.showContentWarning(responseData.detail.message, responseData.detail.suggestion, true /* isBlocked */);
                    return null; // Blocked content, stop further processing
                }
                // For other errors, throw a generic message or the specific detail
                throw new Error(responseData.detail?.message || responseData.detail || `HTTP error! status: ${response.status}`);
            }

            return responseData;
        } catch (error) {
            console.error('API Error:', error);
            // If it's a manually thrown error with a message, use that, otherwise generic
            this.showError(error.message || 'Failed to get a response from the server.');
            return null; // Ensure we return null on error to stop processing
        }
    }

    showLoading(show) {
        // Show response container during loading, but only show the loading indicator
        const responseContainer = document.querySelector('.response-container');
        if (responseContainer && show) {
            console.log('Showing response container for loading');
            responseContainer.classList.remove('hidden');
            responseContainer.classList.add('show');
        }
        
        this.loadingIndicator.classList.toggle('hidden', !show);
        this.responseContent.classList.toggle('hidden', show);
        this.submitBtn.disabled = show;
    }

    // Separate loading method for National ID processing that doesn't show response container
    showNationalIdLoading(show) {
        // Only disable submit button, don't show response container
        this.submitBtn.disabled = show;
        
        // You could add a separate loading indicator here if needed
        // For now, just disable the button to indicate processing
    }

    displayResponse(response) {
        if (!response) {
            const responseContainer = document.querySelector('.response-container');
            if (responseContainer) {
                console.log('No response received, hiding response container');
                responseContainer.classList.remove('show');
                responseContainer.classList.add('hidden');
            }
            return;
        }

        if (!this.answerSection) {
            console.error('Answer section element not found in displayResponse');
            return;
        }

        // Show the response container
        const responseContainer = document.querySelector('.response-container');
        if (responseContainer) {
            console.log('Response container found, showing it');
            responseContainer.classList.remove('hidden');
            responseContainer.classList.add('show');
        } else {
            console.error('Response container not found!');
        }

        // Create or get the chat container
        let chatContainer = this.answerSection.querySelector('.chatcontainer');
        if (!chatContainer) {
            chatContainer = document.createElement('div');
            chatContainer.className = 'chatcontainer';
            this.answerSection.appendChild(chatContainer);
        }

        // Display the entire conversation history
        chatContainer.innerHTML = '';
        this.chatHistory.forEach((message, index) => {
            const messageDiv = document.createElement('div');
            messageDiv.className = 'chat';
            
            // Add specific styling based on the role
            if (message.role === 'user') {
                messageDiv.style.cssText = `
                    float: right;
                    background-color: #e9e8e8;
                    clear: both;
                `;
            } else {
                messageDiv.style.cssText = `
                    float: left;
                    background-color: #f2f2f2;
                    clear: both;
                `;
            }

            // Convert markdown to HTML for assistant messages
            const content = message.role === 'assistant' 
                ? this.markdownToHtml(message.content)
                : this.escapeHtml(message.content);
            
            messageDiv.innerHTML = content;
            chatContainer.appendChild(messageDiv);
        });

        // Scroll to the bottom of the chat container
        chatContainer.scrollTop = chatContainer.scrollHeight;

        // Handle PDF display if needed
        if (response.pdf_info && response.pdf_info.pdf_link) {
            if (!this.isPdfLoaded || this.currentPdfLink !== response.pdf_info.pdf_link) {
                this.displayPDF(response.pdf_info);
            }
        }
    }

    async displayPDF(pdfInfo) {
        if (!this.pdfViewer || !this.pdfFrame || !this.pdfCompany) {
            console.error('PDF viewer elements not found');
            return;
        }

        try {
            // Update company name
            this.pdfCompany.textContent = pdfInfo.company_name || 'Unknown Company';

            // Create a blob URL for the PDF
            const response = await fetch('/api/pdf', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ pdf_link: pdfInfo.pdf_link })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `Failed to fetch PDF: ${response.status}`);
            }

            const pdfBlob = await response.blob();
            const pdfUrl = URL.createObjectURL(pdfBlob);

            // Set the PDF in the iframe
            this.pdfFrame.src = pdfUrl;

            // Hide placeholder and show PDF viewer
            if (this.pdfPlaceholder) {
                this.pdfPlaceholder.classList.add('hidden');
            }
            this.pdfViewer.classList.remove('hidden');

            // Clean up the blob URL after a delay to ensure it loads
            setTimeout(() => {
                URL.revokeObjectURL(pdfUrl);
            }, 5000);

            // Update PDF state
            this.isPdfLoaded = true;
            this.currentPdfLink = pdfInfo.pdf_link;

        } catch (error) {
            console.error('Error displaying PDF:', error);
            this.hidePDF();
            
            // Show error message in the placeholder
            if (this.pdfPlaceholder) {
                this.pdfPlaceholder.classList.remove('hidden');
                this.pdfPlaceholder.innerHTML = `
                    <div class="pdf-placeholder-icon">⚠️</div>
                    <div class="pdf-placeholder-text error">
                        <strong>Error Loading PDF</strong><br>
                        ${error.message}
                    </div>
                `;
            }
        }
    }

    hidePDF() {
        if (this.pdfViewer) {
            this.pdfViewer.classList.add('hidden');
        }
        if (this.pdfFrame) {
            this.pdfFrame.src = '';
        }
        if (this.pdfPlaceholder) {
            this.pdfPlaceholder.classList.add('hidden');
        }
        
        // Reset PDF state
        this.isPdfLoaded = false;
        this.currentPdfLink = null;
    }

    showError(message) {
        console.log('showError starting, message:', message);
        console.log('answerSection exists:', !!this.answerSection);
        
        // Show the response container
        const responseContainer = document.querySelector('.response-container');
        if (responseContainer) {
            console.log('Response container found in showError, showing it');
            responseContainer.classList.remove('hidden');
            responseContainer.classList.add('show');
        } else {
            console.error('Response container not found in showError!');
        }
        
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

    initializeTheme() {
        // Get saved theme from localStorage or default to light
        const savedTheme = localStorage.getItem('theme') || 'light';
        this.setTheme(savedTheme);
    }

    setTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('theme', theme);
        
        if (this.themeToggle) {
            const themeIcon = this.themeToggle.querySelector('.theme-icon');
            const themeText = this.themeToggle.querySelector('.theme-text');
            
            if (theme === 'dark') {
                themeIcon.textContent = '☀️';
                themeText.textContent = 'Light';
            } else {
                themeIcon.textContent = '🌙';
                themeText.textContent = 'Dark';
            }
        }
    }

    toggleTheme() {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        this.setTheme(newTheme);
    }

    displayFamilyInformation(familyData) {
        console.log('displayFamilyInformation called with:', familyData);
        
        const familyInfoSection = document.getElementById('familyInfoSection');
        const familyGrid = document.getElementById('familyGrid');
        const familyEmpty = document.getElementById('familyEmpty');
        const totalMembersElement = document.getElementById('totalMembers');
        const activePoliciesElement = document.getElementById('activePolicies');

        console.log('DOM elements found:', {
            familyInfoSection: !!familyInfoSection,
            familyGrid: !!familyGrid,
            familyEmpty: !!familyEmpty,
            totalMembersElement: !!totalMembersElement,
            activePoliciesElement: !!activePoliciesElement
        });

        if (!familyData || !familyData.members || familyData.members.length === 0) {
            if (familyInfoSection) familyInfoSection.classList.add('hidden');
            if (familyGrid) familyGrid.classList.add('hidden');
            if (familyEmpty) familyEmpty.classList.remove('hidden');
            if (totalMembersElement) totalMembersElement.textContent = '0';
            if (activePoliciesElement) activePoliciesElement.textContent = '0';
            return;
        }

        // Show the family section
        if (familyInfoSection) {
            familyInfoSection.classList.remove('hidden');
            
            // Set up toggle functionality if not already done
            const toggleBtn = document.getElementById('toggleFamily');
            const familyContent = document.getElementById('familyContent');
            
            if (toggleBtn && familyContent && !toggleBtn.hasAttribute('data-initialized')) {
                toggleBtn.setAttribute('data-initialized', 'true');
                toggleBtn.addEventListener('click', () => {
                    const isCollapsed = familyContent.classList.contains('collapsed');
                    if (isCollapsed) {
                        familyContent.classList.remove('collapsed');
                        toggleBtn.classList.remove('collapsed');
                    } else {
                        familyContent.classList.add('collapsed');
                        toggleBtn.classList.add('collapsed');
                    }
                });
            }
        }

        // Update statistics
        if (totalMembersElement) totalMembersElement.textContent = familyData.total_members.toString();
        if (activePoliciesElement) activePoliciesElement.textContent = '1'; // We know there's at least 1 policy

        // Hide empty state and show tree
        if (familyEmpty) familyEmpty.classList.add('hidden');
        if (familyGrid) {
            familyGrid.classList.remove('hidden');
            
            // Clear existing content completely
            familyGrid.innerHTML = '';
            
            // Remove any existing CSS classes that might interfere
            familyGrid.className = 'family-grid';
            
            // Set inline styles to ensure proper layout
            familyGrid.style.cssText = `
                display: flex;
                justify-content: center;
                align-items: flex-start;
                margin-top: 1.5rem;
                padding: 2rem;
                min-height: 400px;
                width: 100%;
            `;

            console.log('Creating family tree with members:', familyData.members);
            
            // Create family tree structure
            const familyTree = this.createFamilyTree(familyData.members);
            familyGrid.appendChild(familyTree);
            
            console.log('Family tree appended to grid. Grid contents:', familyGrid.innerHTML);
        }
    }

    createFamilyTree(members) {
        console.log('createFamilyTree called with members:', members);
        
        // Organize members by relation and level
        const principal = members.find(m => m.relation === 'PRINCIPAL' || m.relation_order === 3);
        const spouse = members.find(m => m.relation === 'SPOUSE' || m.relation_order === 1);
        
        // Group children by their parent's ID
        const childrenByParent = {};
        members.filter(m => m.relation === 'CHILD' || m.relation_order === 2).forEach(child => {
            const parentId = child.parent_id || 'root';
            if (!childrenByParent[parentId]) {
                childrenByParent[parentId] = [];
            }
            childrenByParent[parentId].push(child);
        });

        console.log('Organized members:', {
            principal: principal,
            spouse: spouse,
            childrenByParent: childrenByParent
        });

        // Create main tree container
        const treeContainer = document.createElement('div');
        treeContainer.className = 'family-tree';

        // Create parent level container (for principal and spouse)
        const parentLevel = document.createElement('div');
        parentLevel.style.cssText = 
            'display: flex;' +
            'justify-content: center;' +
            'align-items: center;' +
            'gap: 4rem;' +
            'position: relative;' +
            'width: 100%;';

        // Add principal and spouse nodes
        if (principal) {
            const principalNode = this.createSimpleTreeNode(principal, 'PRINCIPAL');
            parentLevel.appendChild(principalNode);
        }

        if (spouse) {
            const spouseNode = this.createSimpleTreeNode(spouse, 'SPOUSE');
            parentLevel.appendChild(spouseNode);
        }

        // Add horizontal line between spouses if both exist
        if (principal && spouse) {
            const spouseConnection = document.createElement('div');
            spouseConnection.className = 'tree-connection horizontal spouse-line';
            spouseConnection.style.cssText = 
                'position: absolute;' +
                'top: 50%;' +
                'left: 50%;' +
                'transform: translate(-50%, -50%);' +
                'width: 80px;' +
                'height: 2px;' +
                'background: linear-gradient(90deg, #c2e3da, #9dd4b8);' +
                'z-index: 1;';
            parentLevel.appendChild(spouseConnection);
        }

        treeContainer.appendChild(parentLevel);

        // Function to recursively create child nodes
        const createChildrenNodes = (parentId, level = 0) => {
            const children = childrenByParent[parentId];
            if (!children || children.length === 0) return null;

            // Create container for this level of children
            const childrenContainer = document.createElement('div');
            childrenContainer.className = 'children-level level-' + level;
            childrenContainer.style.cssText = `
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 1rem; /* Adjusted gap for rows */
                margin-top: ${level === 0 ? '2rem' : '1rem'};
                position: relative;
            `;

            // Add vertical line from parent
            const verticalLine = document.createElement('div');
            verticalLine.style.cssText = `
                width: 2px;
                height: 40px;
                background: linear-gradient(180deg, #c2e3da, #9dd4b8);
                margin-bottom: 1rem;
            `;
            childrenContainer.appendChild(verticalLine);

            // Process children in pairs
            for (let i = 0; i < children.length; i += 2) {
                const childrenPair = children.slice(i, i + 2);

                // Create row for this pair of children
                const childrenRow = document.createElement('div');
                childrenRow.style.cssText = `
                    display: flex;
                    justify-content: center;
                    gap: 3rem;
                    position: relative;
                    width: 100%; /* Ensure row takes full width */
                    margin-bottom: 1rem; /* Add margin between rows */
                `;

                // Add horizontal line connecting children if more than one in the pair
                if (childrenPair.length > 1) {
                    const horizontalLine = document.createElement('div');
                    horizontalLine.style.cssText = `
                        position: absolute;
                        top: 50%;
                        left: 25%; /* Adjusted for two children */
                        right: 25%; /* Adjusted for two children */
                        height: 2px;
                        background: linear-gradient(90deg, #c2e3da, #9dd4b8);
                        z-index: 1;
                    `;
                    childrenRow.appendChild(horizontalLine);
                }

                // Add child nodes in the pair
                childrenPair.forEach(child => {
                    const childWrapper = document.createElement('div');
                    childWrapper.style.cssText = `
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        position: relative;
                        z-index: 2;
                    `;

                    const childNode = this.createSimpleTreeNode(child, 'CHILD');
                    childWrapper.appendChild(childNode);

                    // Recursively add next level of children
                    const nextLevel = createChildrenNodes(child.national_id, level + 1);
                    if (nextLevel) {
                        childWrapper.appendChild(nextLevel);
                    }

                    childrenRow.appendChild(childWrapper);
                });
                childrenContainer.appendChild(childrenRow);
            }
            return childrenContainer;
        };

        // Add root level children
        const rootChildren = createChildrenNodes('root');
        if (rootChildren) {
            treeContainer.appendChild(rootChildren);
        }

        console.log('Tree container created:', treeContainer);
        return treeContainer;
    }

    createSimpleTreeNode(member, nodeType) {
        console.log(`Creating simple tree node for ${member.name} as ${nodeType}`);
        
        const node = document.createElement('div');
        node.className = 'family-tree-node';
        node.setAttribute('data-relation', nodeType);
        
        const initials = this.getInitials(member.name);
        const relationIcon = this.getRelationIcon(member.relation);
        
        node.innerHTML = `
            <div class="family-tree-node-content">
                <div class="family-tree-node-initials">${initials}</div>
                <div class="family-tree-node-name">${this.escapeHtml(member.name)}</div>
                <div class="family-tree-node-relation">${relationIcon} ${this.escapeHtml(nodeType)}</div>
                <div class="family-tree-node-details">
                ${member.national_id ? `
                        <div class="family-tree-node-detail">
                            <span class="family-tree-node-detail-label">ID</span>
                            <span class="family-tree-node-detail-value">${this.escapeHtml(member.national_id)}</span>
                    </div>
                ` : ''}
                    <div class="family-tree-node-detail">
                        <span class="family-tree-node-detail-label">DOB</span>
                        <span class="family-tree-node-detail-value">${this.formatDate(member.date_of_birth)}</span>
                    </div>
                </div>
            </div>
        `;

        // Add click event to show member's policy details
        node.addEventListener('click', () => {
            this.showMemberPolicyDetails(member);
        });

        console.log(`Created node for ${member.name}`);
        return node;
    }

    getInitials(name) {
        if (!name) return '?';
        return name
            .split(' ')
            .map(word => word[0])
            .join('')
            .toUpperCase()
            .slice(0, 2);
    }

    formatDate(dateString) {
        if (!dateString) return '-';
        try {
            // Handle different date formats
            const date = new Date(dateString);
            if (isNaN(date.getTime())) {
                // If direct parsing fails, try to handle different formats
                const parts = dateString.split(/[-/]/);
                if (parts.length === 3) {
                    // Try different date part arrangements
                    date = new Date(parts[2], parts[1] - 1, parts[0]); // DD/MM/YYYY
                    if (isNaN(date.getTime())) {
                        date = new Date(parts[2], parts[0] - 1, parts[1]); // MM/DD/YYYY
                    }
                }
            }
            
            if (!isNaN(date.getTime())) {
                return date.toLocaleDateString('en-GB'); // Format as DD/MM/YYYY
            }
            return dateString; // Return original string if parsing fails
        } catch (e) {
            console.error('Error formatting date:', e);
            return dateString || '-';
        }
    }

    showMemberPolicyDetails(member) {
        // Create and show a modal with detailed policy information
        const modal = document.createElement('div');
        modal.className = 'modal';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3>${this.escapeHtml(member.name)}'s Policy Details</h3>
                    <button class="modal-close">&times;</button>
                </div>
                <div class="modal-body">
                    <div class="policy-details-grid">
                        <div class="detail-item">
                            <div class="detail-label">Full Name</div>
                            <div class="detail-value">${this.escapeHtml(member.name)}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">National ID</div>
                            <div class="detail-value">${this.escapeHtml(member.national_id || 'N/A')}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Relation</div>
                            <div class="detail-value">${this.getRelationIcon(member.relation)} ${this.escapeHtml(member.relation)}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Date of Birth</div>
                            <div class="detail-value">${this.formatDate(member.date_of_birth)}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Policy Number</div>
                            <div class="detail-value">${this.escapeHtml(member.policy_number || 'N/A')}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Company Name</div>
                            <div class="detail-value">${this.escapeHtml(member.company_name || 'N/A')}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Contract ID</div>
                            <div class="detail-value">${this.escapeHtml(member.contract_id || 'N/A')}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Coverage Period</div>
                            <div class="detail-value">
                                ${this.formatDate(member.start_date)} - ${this.formatDate(member.end_date)}
                            </div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Annual Limit</div>
                            <div class="detail-value">${this.escapeHtml(member.annual_limit || 'N/A')}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Coverage Area</div>
                            <div class="detail-value">${this.escapeHtml(member.area_of_cover || 'N/A')}</div>
                        </div>
                        <div class="detail-item full-width">
                            <div class="detail-label">Emergency Treatment</div>
                            <div class="detail-value">${this.escapeHtml(member.emergency_treatment || 'N/A')}</div>
                        </div>
                        ${member.pdf_link ? `
                            <div class="detail-item full-width">
                                <div class="detail-label">Policy Document</div>
                                <div class="detail-value">
                                    <a href="${this.escapeHtml(member.pdf_link)}" target="_blank" style="color: var(--primary-color); text-decoration: none;">
                                        📄 View Policy Document
                                    </a>
                                </div>
                            </div>
                        ` : ''}
                    </div>
                </div>
            </div>
        `;

        // Add modal close functionality
        const closeBtn = modal.querySelector('.modal-close');
        closeBtn.addEventListener('click', () => {
            modal.remove();
        });

        // Close modal when clicking outside
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.remove();
            }
        });

        document.body.appendChild(modal);
    }

    formatBenefits(benefits) {
        if (!benefits) return 'No benefits information available';
        try {
            const benefitsObj = typeof benefits === 'string' ? JSON.parse(benefits) : benefits;
            return Object.entries(benefitsObj)
                .map(([key, value]) => `
                    <div class="benefit-item">
                        <strong>${this.escapeHtml(key)}:</strong> ${this.escapeHtml(value)}
                    </div>
                `)
                .join('');
        } catch (e) {
            return this.escapeHtml(benefits);
        }
    }

    getRelationIcon(relation) {
        if (!relation) return '👤';
        const relationLower = relation.toLowerCase();
        if (relationLower.includes('spouse') || relationLower.includes('wife') || relationLower.includes('husband')) {
            return '💑';
        } else if (relationLower.includes('child') || relationLower.includes('son') || relationLower.includes('daughter')) {
            return '👶';
        } else if (relationLower.includes('father') || relationLower.includes('mother') || relationLower.includes('parent')) {
            return '👴';
        }
        return '👤';
    }

    clearResults(showError = false, errorMessage = '', showDefault = false) {
        // Hide response container
        const responseContainer = document.querySelector('.response-container');
        if (responseContainer) {
            console.log('Hiding response container in clearResults');
            responseContainer.classList.remove('show');
            responseContainer.classList.add('hidden');
        }

        // Hide suggestions
        if (this.suggestedQuestions) {
            this.suggestedQuestions.classList.add('hidden');
        }

        // Hide family information
        const familyInfoSection = document.getElementById('familyInfoSection');
        if (familyInfoSection) {
            familyInfoSection.classList.add('hidden');
        }

        // Hide PDF viewer
        if (this.pdfViewer) {
            this.pdfViewer.classList.add('hidden');
        }

        // Clear PDF frame
        if (this.pdfFrame) {
            this.pdfFrame.src = '';
        }

        // Clear or show error/default state in PDF placeholder
        if (this.pdfPlaceholder) {
            if (showError) {
                this.pdfPlaceholder.classList.remove('hidden');
                this.pdfPlaceholder.innerHTML = `
                    <div class="pdf-placeholder-icon error">⚠️</div>
                    <div class="pdf-placeholder-text error">
                        <strong>Error</strong><br>
                        ${this.escapeHtml(errorMessage)}
                    </div>
                `;
            } else if (showDefault) {
                this.pdfPlaceholder.classList.remove('hidden');
                this.pdfPlaceholder.innerHTML = `
                    <div class="pdf-placeholder-icon">📄</div>
                    <div class="pdf-placeholder-text">
                        <strong>Policy Document</strong><br>
                        Enter your National ID to view your policy document
                    </div>
                `;
            } else {
                this.pdfPlaceholder.classList.add('hidden');
            }
        }

        // Clear answer section
        if (this.answerSection) {
            this.answerSection.innerHTML = '';
        }

        // Reset PDF state
        this.isPdfLoaded = false;
        this.currentPdfLink = null;
    }

    // New method to display content warnings
    showContentWarning(message, suggestion, isBlocked = false) {
        if (!this.answerSection) {
            this.createResponseStructure(); // Ensure response area exists
        }
        
        // Show the response container
        const responseContainer = document.querySelector('.response-container');
        if (responseContainer) {
            console.log('Response container found in showContentWarning, showing it');
            responseContainer.classList.remove('hidden');
            responseContainer.classList.add('show');
        } else {
            console.error('Response container not found in showContentWarning!');
        }
        
        let warningHtml = `
            <div class="content-warning ${isBlocked ? 'blocked' : 'sanitized'}">
                <div class="warning-icon">${isBlocked ? '🚫' : '⚠️'}</div>
                <div class="warning-text">
                    <p class="warning-message">${this.escapeHtml(message)}</p>
        `;
        if (suggestion) {
            warningHtml += `<p class="warning-suggestion">${this.escapeHtml(suggestion)}</p>`;
        }
        warningHtml += `</div></div>`;
        
        this.answerSection.innerHTML = warningHtml;
        this.showLoading(false); // Ensure loading indicator is hidden
    }

    updateUserInfo(data) {
        if (data && data.answer) {
            try {
                // Extract information from the answer
                const answer = data.answer;
                let info = {};

                // Try to parse if it's a JSON string
                if (typeof answer === 'string') {
                    try {
                        info = JSON.parse(answer);
                    } catch {
                        // If not JSON, parse the text response
                        const nameMatch = answer.match(/name(?:\s+is)?:\s*([^\n,]+)/i);
                        const expiryMatch = answer.match(/expiry(?:\s+date)?:\s*([^\n,]+)/i);
                        const familyMatch = answer.match(/family members?:\s*(\d+)/i);
                        
                        info = {
                            name: nameMatch ? nameMatch[1].trim() : null,
                            expiry_date: expiryMatch ? expiryMatch[1].trim() : null,
                            family_count: familyMatch ? familyMatch[1].trim() : null
                        };
                    }
                } else {
                    info = answer;
                }

                // Update the display fields
                this.contractorName.textContent = info.name || info.contractor_name || info.username || '-';
                this.expiryDate.textContent = info.expiry_date || info.policy_expiry || '-';
                this.beneficiaryCount.textContent = info.family_count || info.beneficiary_count || info.members_count || '-';

            } catch (e) {
                console.error('Error parsing user info:', e);
                this.clearUserInfo();
            }
        } else {
            this.clearUserInfo();
        }
    }

    clearUserInfo() {
        this.contractorName.textContent = '-';
        this.expiryDate.textContent = '-';
        this.beneficiaryCount.textContent = '-';
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