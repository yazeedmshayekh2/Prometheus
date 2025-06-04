console.log('app.js starting to load');

class InsuranceAssistant {
    constructor() {
        // Check authentication first
        const token = localStorage.getItem('authToken');
        if (!token) {
            window.location.href = '/login.html';
            return;
        }

        console.log('InsuranceAssistant constructor starting');
        
        // Chat elements
        this.chatContainer = document.querySelector('.chatcontainer');
        this.textArea = document.querySelector('.textChat textarea');
        this.sendButton = document.querySelector('.btnSend');
        this.loadingIndicator = document.getElementById('loadingIndicator');

        // User info elements
        this.contractorName = document.getElementById('contractorName');
        this.expiryDate = document.getElementById('expiryDate');
        this.beneficiaryCount = document.getElementById('beneficiaryCount');

        // Modals
        this.tobModal = document.getElementById('TOBModal');
        this.memsModal = document.getElementById('MemsModal');
        
        // Suggestions
        this.suggestContainer = document.querySelector('.suggest');
        
        // State tracking
        this.isNationalIdConfirmed = false;
        this.currentNationalId = '';
        this.isLoading = false;
        this.chatHistory = [];

        // Add conversation history management
        this.currentConversationId = null;
        this.setupSidebar();
        this.loadConversations();

        // Initialize
        this.setupEventListeners();
        this.initializeChat();
        
        console.log('InsuranceAssistant initialized with elements:', {
            chatContainer: !!this.chatContainer,
            textArea: !!this.textArea,
            sendButton: !!this.sendButton
        });
    }

    setupEventListeners() {
        // Text input handler
        if (this.textArea) {
            this.textArea.addEventListener('input', (e) => {
                if (!this.isNationalIdConfirmed) {
                    // Only allow numbers and limit to 11 digits for National ID
                    const numbersOnly = e.target.value.replace(/\D/g, '');
                    if (numbersOnly !== e.target.value) {
                e.target.value = numbersOnly;
            }
                    if (numbersOnly.length > 11) {
                        e.target.value = numbersOnly.slice(0, 11);
                }
                }
                this.updateSendButtonState();
        });
        
            this.textArea.addEventListener('keypress', async (e) => {
                if (e.key === 'Enter' && !e.shiftKey && !this.isLoading) {
                e.preventDefault();
                    await this.handleSendMessage();
                }
            });
        }

        // Send button handler
        if (this.sendButton) {
            this.sendButton.addEventListener('click', async (e) => {
                e.preventDefault();
                if (!this.isLoading) {
                    await this.handleSendMessage();
                }
            });
        }

        // Initialize button state
        this.updateSendButtonState();
    }

    initializeChat() {
        // Clear any existing messages except the welcome message
        if (this.chatContainer) {
            const welcomeMessage = this.chatContainer.querySelector('.chat:first-child');
            if (welcomeMessage) {
                this.chatContainer.innerHTML = '';
                this.chatContainer.appendChild(welcomeMessage);
            }
        }
    }

    updateSendButtonState() {
        if (!this.sendButton || !this.textArea) return;

        const inputValue = this.textArea.value.trim();
        let canSubmit = false;

        if (!this.isNationalIdConfirmed) {
            // For ID validation phase
            canSubmit = inputValue.replace(/\D/g, '').length === 11;
        } else {
            // For question phase
            canSubmit = inputValue.length > 0;
        }

        this.sendButton.disabled = !canSubmit || this.isLoading;
        this.sendButton.style.opacity = canSubmit && !this.isLoading ? '1' : '0.5';
        this.sendButton.style.cursor = canSubmit && !this.isLoading ? 'pointer' : 'not-allowed';
    }

    async handleSendMessage() {
        if (this.isLoading || !this.textArea) return;

        const message = this.textArea.value.trim();
        if (!message) return;

        if (!this.isNationalIdConfirmed) {
            // Handle National ID validation
            const numbersOnly = message.replace(/\D/g, '');
            if (numbersOnly.length === 11) {
                await this.handleNationalIdConfirm(numbersOnly);
            }
        } else {
            // Handle regular chat message
            await this.handleQuestionSubmit(message);
        }

        // Clear input after sending
        this.textArea.value = '';
        this.updateSendButtonState();
    }

    async handleNationalIdConfirm(nationalId) {
        try {
            // Add user's message to chat
            this.addMessageToChat('user', nationalId);
            
            // Show loading state
            this.addLoadingIndicator();
            this.setLoading(true);
            
            const response = await fetch('/api/suggestions', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    national_id: nationalId
                })
            });

            if (!response.ok) {
                throw new Error('Failed to fetch user information');
            }

            const data = await response.json();
            console.log('API Response:', data);
        
            // Remove loading indicator before adding success message
            this.removeLoadingIndicator();
        
        // Mark as confirmed
        this.isNationalIdConfirmed = true;
        this.currentNationalId = nationalId;
        
            // Add confirmation message to chat
            this.addMessageToChat('assistant', '✅ ID verified successfully. How can I help you with your policy today?');
            
            // Update textarea placeholder
            if (this.textArea) {
                this.textArea.placeholder = 'Ask anything about your policy...';
            }

            // Update user info display and handle PDF
            if (data?.family_data?.members) {
                const members = data.family_data.members;
                const principal = members.find(m => m.national_id === nationalId) || members[0];
                
                if (principal) {
                    // Set contractor name as company name
                    this.contractorName.textContent = this.capitalizeWords(principal.company_name || '-');
                    
                    // Set individual name in the appropriate field
                    const individualNameElement = document.querySelector('.name.col-md-7');
                    if (individualNameElement) {
                        individualNameElement.textContent = this.capitalizeWords(principal.name || '-');
                    }
                    
                    this.expiryDate.textContent = this.formatDate(principal.end_date) || '-';
                    this.beneficiaryCount.textContent = data.family_data.total_members.toString() || '-';

                    // Try to get PDF info from different possible sources
                    let pdfInfo = null;

                    // First try the direct pdf_info from the response
                    if (data.pdf_info?.pdf_link) {
                        pdfInfo = {
                            pdf_link: data.pdf_info.pdf_link,
                            company_name: data.pdf_info.company_name || principal.company_name || 'DIG'
                        };
                    }
                    // Then try the principal member's PDF link
                    else if (principal.pdf_link) {
                        pdfInfo = {
                            pdf_link: principal.pdf_link,
                            company_name: principal.company_name || 'DIG'
                        };
                    }
                    // Finally, look through other family members for a PDF link
                    else {
                        const memberWithPdf = members.find(m => m.pdf_link);
                        if (memberWithPdf) {
                            pdfInfo = {
                                pdf_link: memberWithPdf.pdf_link,
                                company_name: memberWithPdf.company_name || principal.company_name || 'DIG'
                            };
                        }
                    }

                    // Display PDF if we found any PDF information
                    if (pdfInfo) {
                        console.log('Displaying PDF with info:', pdfInfo);
                        try {
                            await this.displayPDF(pdfInfo);
                        } catch (error) {
                            console.error('Error displaying PDF:', error);
                            this.showErrorMessage('Failed to display PDF. Please try again later.');
                        }
                    } else {
                        console.log('No PDF link available in the response');
                    }
                }
            }

            // Display suggested questions if available
            if (data.questions && data.questions.length > 0) {
                this.displaySuggestedQuestions(data.questions);
            }

        } catch (error) {
            console.error('Error:', error);
            this.removeLoadingIndicator();
            this.addMessageToChat('assistant', '❌ Failed to verify ID. Please try again.');
            this.isNationalIdConfirmed = false;
            this.clearUserInfo();
        } finally {
            this.setLoading(false);
        }
    }

    capitalizeWords(str) {
        if (!str || str === '-') return str;
        return str.split(' ')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
            .join(' ');
    }

    async handleQuestionSubmit(question) {
        if (!question || !this.isNationalIdConfirmed || !this.currentNationalId) {
            console.log('Cannot submit: question empty or ID not confirmed');
            return;
        }

        try {
            // Add user's message to chat
            this.addMessageToChat('user', question);
            
            // Show loading state
            this.addLoadingIndicator();
            this.setLoading(true);

            // Query the API
            const response = await this.queryAPI(this.currentNationalId, question);
            
            // Remove loading before showing response
            this.removeLoadingIndicator();

            if (response) {
                // Add assistant's response to chat
                this.addMessageToChat('assistant', response.answer);

                // Update suggested questions if available
                if (response.suggested_questions) {
                    this.displaySuggestedQuestions(response.suggested_questions);
                }
            }

        } catch (error) {
            console.error('Error submitting question:', error);
            this.removeLoadingIndicator();
            this.addMessageToChat('assistant', 'Sorry, I encountered an error processing your request. Please try again.');
        } finally {
            this.setLoading(false);
        }
    }
    
    addMessageToChat(role, content, shouldSave = true) {
        if (!content) {
            console.error('No content provided for chat message');
            return;
        }

        console.log(`Adding ${role} message:`, content);
        
        // Add to chat history
        this.chatHistory.push({ role, content });
        
        // Create message element
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat ${role === 'user' ? 'user' : 'assistant'}`;

        // Set content with proper formatting
        messageDiv.innerHTML = role === 'assistant' 
            ? this.markdownToHtml(content)
            : this.escapeHtml(content);
        
        // Add message to container
        if (this.chatContainer) {
            this.chatContainer.appendChild(messageDiv);
            // Scroll to bottom
            this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
        }
        
        console.log('Message added to chat');

        // Save conversation only if shouldSave is true
        if (shouldSave) {
            this.saveConversation();
        }
    }

    setLoading(isLoading) {
        this.isLoading = isLoading;
        this.showLoading(isLoading);
        this.updateSendButtonState();
    }

    showSuccessMessage() {
        // Implement the logic to show a success message to the user
        console.log('Success message logic not implemented');
    }

    showErrorMessage(message) {
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
                if (this.textArea?.parentNode) {
                    this.textArea.parentNode.insertBefore(fallbackError, this.textArea.nextSibling);
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

    displaySuggestedQuestions(questions) {
        if (!this.suggestContainer) return;
        
        const suggestbtn = this.suggestContainer.querySelector('.suggestbtn');
        if (!suggestbtn) return;
        
        suggestbtn.innerHTML = ''; // Clear previous buttons
        
        if (!questions || !questions.length) {
            console.log('No suggestions to display');
            this.suggestContainer.style.display = 'none';
            return;
        }

        // Take only the first 3 questions
        const limitedQuestions = questions.slice(0, 3);

        limitedQuestions.forEach(question => {
            let cleanQuestion = question
                .replace(/\[\d*\]/g, '')
                .replace(/\*\*Question\s*(?:#?\d+|):\*\*\s*/i, '')
                .replace(/\*\*/g, '')
                .replace(/^#+\s*/, '')
                .trim();
            
            const button = document.createElement('button');
            button.textContent = cleanQuestion;
            button.addEventListener('click', () => {
                if (this.textArea) {
                    this.textArea.value = cleanQuestion;
                    this.textArea.focus();
                    this.updateSendButtonState();
                }
            });
            
            suggestbtn.appendChild(button);
        });
        
        this.suggestContainer.style.display = 'block';

        
    }   
    

    hideSuggestedQuestions() {
        console.log('Hiding suggestions');
        const suggestContainer = document.querySelector('.suggest');
        if (suggestContainer) {
            suggestContainer.style.display = 'none';
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
            this.showErrorMessage(error.message || 'Failed to get a response from the server.');
            return null; // Ensure we return null on error to stop processing
        }
    }

    showLoading(show) {
        if (!this.loadingIndicator) {
            this.loadingIndicator = document.getElementById('loadingIndicator');
        }
        
        if (this.loadingIndicator) {
            if (show) {
                this.loadingIndicator.classList.remove('hidden');
            } else {
                this.loadingIndicator.classList.add('hidden');
            }
        }
        
        // Disable input while loading
        if (this.textArea) {
            this.textArea.disabled = show;
        }
        if (this.sendButton) {
            this.sendButton.disabled = show;
            this.sendButton.style.opacity = show ? '0.5' : '1';
            this.sendButton.style.cursor = show ? 'not-allowed' : 'pointer';
        }
    }

    // Separate loading method for National ID processing that doesn't show response container
    showNationalIdLoading(show) {
        // Only disable submit button, don't show response container
        this.sendButton.disabled = show;
        
        // You could add a separate loading indicator here if needed
        // For now, just disable the button to indicate processing
    }

    async displayPDF(pdfInfo) {
        if (!pdfInfo || !pdfInfo.pdf_link) {
            console.error('No PDF info provided');
            return;
        }

        try {
            // Get modal elements
            const modal = document.querySelector('#TOBModal');
            const modalTitle = document.querySelector('#TOBModalLabel');
            const pdfFrame = document.querySelector('#pdfFrame');
            const loadingIndicator = document.querySelector('#pdfLoadingIndicator');
            const errorMessage = document.querySelector('#pdfErrorMessage');

            // Show loading indicator
            if (loadingIndicator) {
                loadingIndicator.style.display = 'block';
            }
            if (errorMessage) {
                errorMessage.style.display = 'none';
            }
            if (pdfFrame) {
                pdfFrame.style.display = 'none';
            }

            // Update modal title
            if (modalTitle) {
                modalTitle.textContent = `${pdfInfo.company_name || 'Company'} TOB`;
            }

            // Show modal
            if (modal && !modal.classList.contains('show')) {
                const bootstrapModal = new bootstrap.Modal(modal);
            }
            
            // Create a blob URL for the PDF
            const response = await fetch('/api/pdf', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ pdf_link: pdfInfo.pdf_link })
            });

            if (!response.ok) {
                throw new Error(`Failed to fetch PDF: ${response.status}`);
            }

            const pdfBlob = await response.blob();
            const pdfUrl = URL.createObjectURL(pdfBlob);

            // Update iframe source and show it
            if (pdfFrame) {
                pdfFrame.onload = () => {
                    // Hide loading indicator and show PDF
                    if (loadingIndicator) {
                        loadingIndicator.style.display = 'none';
                    }
                    pdfFrame.style.display = 'block';
                };

                pdfFrame.onerror = () => {
                    throw new Error('Failed to load PDF in iframe');
                };

                pdfFrame.src = pdfUrl;
            }

            // Clean up the blob URL after a delay to ensure it loads
            setTimeout(() => {
                URL.revokeObjectURL(pdfUrl);
            }, 5000);

        } catch (error) {
            console.error('Error displaying PDF:', error);
            const loadingIndicator = document.querySelector('#pdfLoadingIndicator');
            const errorMessage = document.querySelector('#pdfErrorMessage');
            const pdfFrame = document.querySelector('#pdfFrame');

            if (loadingIndicator) {
                loadingIndicator.style.display = 'none';
            }
            if (errorMessage) {
                errorMessage.style.display = 'block';
                errorMessage.textContent = 'Failed to load PDF. Please try again later.';
            }
            if (pdfFrame) {
                pdfFrame.style.display = 'none';
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
    }

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
                        const companyMatch = answer.match(/company(?:\s+name)?:\s*([^\n,]+)/i);
                        const expiryMatch = answer.match(/expiry(?:\s+date)?:\s*([^\n,]+)/i);
                        const familyMatch = answer.match(/family members?:\s*(\d+)/i);
                        
                        info = {
                            name: nameMatch ? nameMatch[1].trim() : null,
                            company_name: companyMatch ? companyMatch[1].trim() : null,
                            expiry_date: expiryMatch ? expiryMatch[1].trim() : null,
                            family_count: familyMatch ? familyMatch[1].trim() : null
                        };
                    }
                } else {
                    info = answer;
                }

                // Update the display fields
                // Set contractor name as company name
                this.contractorName.textContent = this.capitalizeWords(info.company_name || '-');
                
                // Set individual name
                const individualNameElement = document.querySelector('.name.col-md-7');
                if (individualNameElement) {
                    individualNameElement.textContent = this.capitalizeWords(info.name || '-');
                }
                
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
        const individualNameElement = document.querySelector('.name.col-md-7');
        if (individualNameElement) {
            individualNameElement.textContent = '-';
        }
        this.expiryDate.textContent = '-';
        this.beneficiaryCount.textContent = '-';
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
        if (!familyData || !familyData.members || familyData.members.length === 0) {
            console.log('No family data to display');
            return;
        }

        const modalBody = document.querySelector('.modal-body.familymem');
        if (!modalBody) {
            console.error('Family modal body not found');
            return;
        }

        // Clear existing content
        modalBody.innerHTML = '';

        // Sort members by relation
        const principal = familyData.members.find(m => m.relation === 'PRINCIPAL');
        const spouse = familyData.members.find(m => m.relation === 'SPOUSE');
        const children = familyData.members.filter(m => m.relation === 'CHILD');

        // Add principal
        if (principal) {
            const principalDiv = document.createElement('div');
            principalDiv.className = 'Parent';
            principalDiv.innerHTML = `
                <span>${this.escapeHtml(principal.name)}</span>
                <span>QID: ${this.escapeHtml(principal.national_id)}</span>
                ${principal.contract_id ? `<span>Indv. ID: ${this.escapeHtml(principal.contract_id)}</span>` : ''}
            `;
            modalBody.appendChild(principalDiv);
        }

        // Add spouse
        if (spouse) {
            const spouseDiv = document.createElement('div');
            spouseDiv.className = 'spouse';
            spouseDiv.innerHTML = `
                <span>${this.escapeHtml(spouse.name)}</span>
                <span>QID: ${this.escapeHtml(spouse.national_id)}</span>
                ${spouse.contract_id ? `<span>Indv. ID: ${this.escapeHtml(spouse.contract_id)}</span>` : ''}
            `;
            modalBody.appendChild(spouseDiv);
        }

        // Add children
        children.forEach(child => {
            const childDiv = document.createElement('div');
            childDiv.className = 'child';
            childDiv.innerHTML = `
                <span>${this.escapeHtml(child.name)}</span>
                <span>QID: ${this.escapeHtml(child.national_id)}</span>
                ${child.contract_id ? `<span>Indv. ID: ${this.escapeHtml(child.contract_id)}</span>` : ''}
            `;
            modalBody.appendChild(childDiv);
        });
    }

    formatDate(dateString) {
        if (!dateString) return '-';
        try {
            const date = new Date(dateString);
            if (isNaN(date.getTime())) {
                // Try different date formats
                const parts = dateString.split(/[-/]/);
                if (parts.length === 3) {
                    // Try DD/MM/YYYY
                    const newDate = new Date(parts[2], parts[1] - 1, parts[0]);
                    if (!isNaN(newDate.getTime())) {
                        return newDate.toLocaleDateString('en-GB');
                    }
                    // Try MM/DD/YYYY
                    const altDate = new Date(parts[2], parts[0] - 1, parts[1]);
                    if (!isNaN(altDate.getTime())) {
                        return altDate.toLocaleDateString('en-GB');
                    }
                }
                return dateString;
            }
            return date.toLocaleDateString('en-GB');
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
        // Never hide response container
        const responseContainer = document.querySelector('.response-container');
        if (responseContainer) {
            responseContainer.classList.add('show');
            responseContainer.classList.remove('hidden');
        }

        // Hide suggestions
        if (this.suggestContainer) {
            this.suggestContainer.classList.add('hidden');
        }

        // Hide family information
        const familyInfoSection = document.getElementById('familyInfoSection');
        if (familyInfoSection) {
            familyInfoSection.classList.add('hidden');
        }

        // Keep chat container visible but clear messages except the welcome message
        if (this.chatContainer) {
            const welcomeMessage = this.chatContainer.querySelector('.chat.assistant-message:first-child');
            if (welcomeMessage) {
                this.chatContainer.innerHTML = '';
                this.chatContainer.appendChild(welcomeMessage);
            }
        }

        // Reset chat history but keep welcome message
        this.chatHistory = [];

        // Update send button state
        this.updateSendButtonState();
    }

    addLoadingIndicator() {
        if (!this.chatContainer) return;

        // Remove any existing loading indicator first
        this.removeLoadingIndicator();

        // Create new loading indicator
        const loadingDiv = document.createElement('div');
        loadingDiv.id = 'loadingIndicator';
        loadingDiv.className = 'chat assistant loading-indicator';
        loadingDiv.innerHTML = `
            <span class="loading-text">Processing your request</span>
            <span class="loading-dots"><span>.</span><span>.</span><span>.</span></span>
        `;
        
        this.chatContainer.appendChild(loadingDiv);
        this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
    }

    removeLoadingIndicator() {
        if (!this.chatContainer) return;
        
        const loadingIndicator = this.chatContainer.querySelector('#loadingIndicator');
        if (loadingIndicator) {
            loadingIndicator.remove();
        }
    }

    checkPDFLoadStatus(iframe) {
        return new Promise((resolve) => {
            if (!iframe) {
                resolve({ success: false, error: 'No iframe found' });
                return;
            }

            const timeoutId = setTimeout(() => {
                resolve({ success: false, error: 'Loading timeout' });
            }, 10000); // 10 second timeout

            iframe.onload = () => {
                clearTimeout(timeoutId);
                resolve({ success: true });
            };

            iframe.onerror = (error) => {
                clearTimeout(timeoutId);
                resolve({ success: false, error: error });
            };
        });
    }

    setupSidebar() {
        // Setup sidebar toggle
        const sidebarToggle = document.querySelector('.sidebar-toggle');
        const sidebar = document.querySelector('.sidebar');
        const sidebarHoverArea = document.querySelector('.sidebar-hover-area');
        
        if (sidebarToggle && sidebar) {
            // Click handler for the toggle button
            sidebarToggle.addEventListener('click', () => {
                sidebar.classList.toggle('open');
            });

            // Close sidebar when clicking outside
            document.addEventListener('click', (e) => {
                // If click is outside sidebar and toggle button, and sidebar is open
                if (!sidebar.contains(e.target) && 
                    !sidebarToggle.contains(e.target) && 
                    !sidebarHoverArea.contains(e.target) &&
                    sidebar.classList.contains('open')) {
                    sidebar.classList.remove('open');
                }
            });

            // Handle mouse leave for the entire sidebar area
            sidebar.addEventListener('mouseleave', () => {
                // Only close if it wasn't opened by clicking
                if (!sidebar.classList.contains('open')) {
                    sidebar.style.left = '-250px';
                    sidebarToggle.style.opacity = '1';
                }
            });

            // Handle hover functionality
            sidebarHoverArea.addEventListener('mouseenter', () => {
                if (!sidebar.classList.contains('open')) {
                    sidebar.style.left = '0';
                    sidebarToggle.style.opacity = '0';
                }
            });

            sidebarHoverArea.addEventListener('mouseleave', (e) => {
                // Check if the mouse moved to the sidebar
                const toElement = e.relatedTarget;
                if (!sidebar.contains(toElement) && !sidebar.classList.contains('open')) {
                    sidebar.style.left = '-250px';
                    sidebarToggle.style.opacity = '1';
                }
            });
        }

        // Set user name in sidebar
        const userName = localStorage.getItem('userName');
        const userNameElement = document.querySelector('.user-info .user-name');
        if (userNameElement && userName) {
            userNameElement.textContent = userName;
        }
    }

    async loadConversations() {
        try {
            const token = localStorage.getItem('authToken');
            if (!token) return;

            const response = await fetch('/api/conversations', {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });

            if (response.ok) {
                const conversations = await response.json();
                this.displayConversations(conversations);
            }
        } catch (error) {
            console.error('Error loading conversations:', error);
        }
    }

    displayConversations(conversations) {
        const conversationList = document.querySelector('.conversation-list');
        if (!conversationList) return;

        conversationList.innerHTML = '';
        
        conversations.sort((a, b) => new Date(b.updated_at) - new Date(a.updated_at));
        
        conversations.forEach(conv => {
            const item = document.createElement('div');
            item.className = 'conversation-item';
            if (conv.id === this.currentConversationId) {
                item.classList.add('active');
            }
            
            // Get the first message or use a default title
            const title = conv.messages[0]?.content || 'New Conversation';
            item.textContent = title.substring(0, 30) + (title.length > 30 ? '...' : '');
            
            item.onclick = () => this.loadConversation(conv.id);
            conversationList.appendChild(item);
        });
    }

    async startNewConversation() {
        // Reset conversation state
        this.currentConversationId = null;
        this.chatHistory = [];
        
        // Reset national ID state
        this.isNationalIdConfirmed = false;
        this.currentNationalId = '';
        
        // Reset user info
        this.clearUserInfo();
        
        // Reset textarea placeholder
        if (this.textArea) {
            this.textArea.value = '';
            this.textArea.placeholder = 'Enter your 11-digit National ID';
        }
        
        // Hide suggested questions
        if (this.suggestContainer) {
            this.suggestContainer.style.display = 'none';
        }
        
        // Clear chat container except welcome message
        if (this.chatContainer) {
            this.chatContainer.innerHTML = '';
            const welcomeMessage = document.createElement('div');
            welcomeMessage.className = 'chat assistant';
            welcomeMessage.innerHTML = '👋 Hi I\'m a chatbot that can help with your medical group policy; Please Enter Your Insured Qatari ID';
            this.chatContainer.appendChild(welcomeMessage);
        }
        
        // Remove active class from all conversations
        document.querySelectorAll('.conversation-item').forEach(item => {
            item.classList.remove('active');
        });

        // Update send button state
        this.updateSendButtonState();
    }

    async loadConversation(conversationId) {
        try {
            const token = localStorage.getItem('authToken');
            if (!token) return;

            const response = await fetch(`/api/conversations/${conversationId}`, {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });

            if (response.ok) {
                const conversation = await response.json();
                
                // Reset all state first (except conversation ID and messages)
                this.isNationalIdConfirmed = false;
                this.currentNationalId = '';
                this.clearUserInfo();
                
                // Reset textarea placeholder
                if (this.textArea) {
                    this.textArea.value = '';
                    this.textArea.placeholder = 'Enter your 11-digit National ID';
                }
                
                // Hide suggested questions
                if (this.suggestContainer) {
                    this.suggestContainer.style.display = 'none';
                }
                
                // Update conversation ID
                this.currentConversationId = conversationId;
                
                // Clear chat container first
                if (this.chatContainer) {
                    this.chatContainer.innerHTML = '';
                }
                
                // Restore chat history
                this.chatHistory = conversation.messages;
                this.chatHistory.forEach(msg => {
                    this.addMessageToChat(msg.role, msg.content, false); // false means don't save to DB
                });

                // Update active state in sidebar
                document.querySelectorAll('.conversation-item').forEach(item => {
                    item.classList.remove('active');
                    if (item.dataset.id === conversationId) {
                        item.classList.add('active');
                    }
                });

                // Update send button state
                this.updateSendButtonState();
            }
        } catch (error) {
            console.error('Error loading conversation:', error);
        }
    }

    async saveConversation() {
        try {
            const token = localStorage.getItem('authToken');
            if (!token || !this.chatHistory.length) return;

            // Create a complete state object
            const conversationState = {
                messages: this.chatHistory,
                userInfo: {
                    contractorName: this.contractorName?.textContent || '-',
                    expiryDate: this.expiryDate?.textContent || '-',
                    beneficiaryCount: this.beneficiaryCount?.textContent || '-',
                    nationalId: this.currentNationalId || ''
                },
                suggestedQuestions: this.suggestContainer?.querySelector('.suggestbtn')?.innerHTML || '',
                isNationalIdConfirmed: this.isNationalIdConfirmed
            };

            const method = this.currentConversationId ? 'PUT' : 'POST';
            const url = this.currentConversationId 
                ? `/api/conversations/${this.currentConversationId}`
                : '/api/conversations';

            const response = await fetch(url, {
                method,
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(conversationState)
            });

            if (response.ok) {
                const conversation = await response.json();
                this.currentConversationId = conversation.id;
                await this.loadConversations(); // Refresh the conversation list
            }
        } catch (error) {
            console.error('Error saving conversation:', error);
        }
    }
}

console.log('About to initialize InsuranceAssistant');

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, checking auth before initializing InsuranceAssistant');
    // Check if user is authenticated
    const token = localStorage.getItem('authToken');
    if (!token) {
        window.location.href = '/login.html';
        return;
    }
    window.insuranceAssistant = new InsuranceAssistant();
});

console.log('app.js finished loading'); 