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
        
        // Arabic text configuration
        this.arabicConfig = {
            fontSize: '11pt',
            fontFamily: localStorage.getItem('arabicFont') || 'Arial' // Default font, can be changed
        };
        
        // TTS Configuration
        this.ttsConfig = {
            provider: localStorage.getItem('ttsProvider') || 'kokoro',
            voice: localStorage.getItem('ttsVoice') || 'sarah',
            availableVoices: null
        };
        
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
        this.familyData = null; // Store family data for later use

        // Add conversation history management
        this.currentConversationId = null;
        this.setupSidebar();
        this.loadConversations();

        // Audio properties
        this.currentAudio = null;
        this.currentAudioButton = null;
        this.audioSpeed = 1.0; // Default speed

        // Initialize
        this.setupEventListeners();
        this.initializeChat();
        this.loadTTSVoices(); // Load available TTS voices
        
        console.log('InsuranceAssistant initialized with elements:', {
            chatContainer: !!this.chatContainer,
            textArea: !!this.textArea,
            sendButton: !!this.sendButton
        });

        // Load jsPDF library
        this.loadJsPDF();
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

        // Voice settings button handler
        const voiceSettingsBtn = document.getElementById('voiceSettingsBtn');
        if (voiceSettingsBtn) {
            voiceSettingsBtn.addEventListener('click', () => {
                this.toggleVoiceSelector();
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
                console.log(data.family_data);
                const members = data.family_data.members;
                const principal = members.find(m => m.national_id === nationalId) || members[0];
                
                // Store family data for later use
                this.familyData = data.family_data;
                
                // Debug logging
                console.log('Family data received:', data.family_data);
                console.log('Total members from API:', data.family_data.total_members);
                console.log('Members array length:', members.length);
                
                if (principal) {
                    // Set contractor name as company name
                    this.contractorName.textContent = this.capitalizeWords(principal.company_name || '-');
                    
                    // Set individual name in the appropriate field
                    const individualNameElement = document.querySelector('.name.col-md-7');
                    if (individualNameElement) {
                        individualNameElement.textContent = this.capitalizeWords(principal.name || '-');
                    }
                    
                    this.expiryDate.textContent = this.formatDate(principal.end_date) || '-';
                    
                    // Fix beneficiary count - use the actual count
                    const beneficiaryCount = data.family_data.total_members;
                    this.beneficiaryCount.textContent = beneficiaryCount.toString();
                    console.log('Setting beneficiary count to:', beneficiaryCount);

                    // Make beneficiary count clickable
                    this.setupBeneficiaryCountClick();

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

    setupBeneficiaryCountClick() {
        const beneficiaryCount = document.getElementById('beneficiaryCount');
        if (beneficiaryCount) {
            beneficiaryCount.style.cursor = 'pointer';
            beneficiaryCount.style.color = '#007bff';
            beneficiaryCount.addEventListener('click', () => {
                console.log('Beneficiary count clicked');
                // Initialize and show the modal
                const memsModal = new bootstrap.Modal(document.getElementById('MemsModal'));
                this.showFamilyMemberCards();
                memsModal.show();
            });
        }
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

        // Create message content wrapper
        const contentWrapper = document.createElement('div');
        contentWrapper.className = 'message-content';

        // Set content with proper formatting
        const formattedContent = role === 'assistant' 
            ? this.basicMarkdownToHtml(content)
            : this.escapeHtml(content);

        // Check if content is Arabic for user messages
        if (role === 'user' && this.isArabicText(content)) {
            contentWrapper.style.textAlign = 'right';
            contentWrapper.style.direction = 'rtl';
            contentWrapper.style.fontSize = this.arabicConfig.fontSize;
            contentWrapper.style.fontFamily = this.arabicConfig.fontFamily;
        }

        contentWrapper.innerHTML = formattedContent;

        // Add content to message div
        messageDiv.appendChild(contentWrapper);
        
        // Add message to container
        if (this.chatContainer) {
        this.chatContainer.appendChild(messageDiv);
            
            // Add action buttons only for assistant messages
            if (role === 'assistant') {
                // Create a wrapper for assistant messages that includes the message and actions
                const messageWrapper = document.createElement('div');
                messageWrapper.className = 'assistant-message-wrapper';
                
                // Add the message to the wrapper
                messageWrapper.appendChild(messageDiv);
                
                // Create and add action buttons
                const actionBar = document.createElement('div');
                actionBar.className = 'message-actions';
                
                // Copy button
                const copyButton = document.createElement('button');
                copyButton.className = 'action-button copy-button';
                copyButton.innerHTML = '<img src="Content/img/copy-2.png" alt="copy" style="width: 20px; height: 20px;">';
                copyButton.title = 'Copy message';
                copyButton.onclick = (e) => {
                    e.stopPropagation();
                    this.addClickAnimation(copyButton);
                    // Strip HTML tags and decode entities for clean text copying
                    const cleanText = this.stripHtmlAndDecode(content);
                    console.log('Copying text:', cleanText.substring(0, 100) + '...');
                    this.copyMessageText(cleanText);
                };
                
                // Audio button
                const audioButton = document.createElement('button');
                audioButton.className = 'action-button audio-button';
                audioButton.innerHTML = '<img src="Content/img/volume.png" alt="audio" style="width: 20px; height: 20px;">'; // Using emoji for now, can be replaced with an icon
                audioButton.title = 'Listen to message';
                audioButton.onclick = (e) => {
                    e.stopPropagation();
                    this.toggleAudio(audioButton, content);
                };
                
                // Like button
                const likeButton = document.createElement('button');
                likeButton.className = 'action-button like-button';
                likeButton.innerHTML = '<img src="Content/img/like-unfilled.png" alt="like" style="width: 20px; height: 20px;">';
                likeButton.title = 'Like this response';
                likeButton.onclick = (e) => {
                    e.stopPropagation();
                    this.likeMessage(likeButton, 'like');
                };
                
                // Dislike button
                const dislikeButton = document.createElement('button');
                dislikeButton.className = 'action-button dislike-button';
                dislikeButton.innerHTML = '<img src="Content/img/dislike-unfilled.png" alt="dislike" style="width: 20px; height: 20px;">';
                dislikeButton.title = 'Dislike this response';
                dislikeButton.onclick = (e) => {
                    e.stopPropagation();
                    this.likeMessage(dislikeButton, 'dislike');
                };
                
                // Add buttons to action bar
                actionBar.appendChild(copyButton);
                actionBar.appendChild(audioButton);
                actionBar.appendChild(likeButton);
                actionBar.appendChild(dislikeButton);

                // Add voice settings button
                const voiceButton = document.createElement('button');
                voiceButton.className = 'btn btn-outline-secondary btn-sm';
                voiceButton.title = 'Voice Settings';
                voiceButton.style.cssText = `
                    background-color: #f8f9fa;
                    border: 1px solid #dee2e6;
                    color: #495057;
                    font-size: 8pt;
                    padding: 8px 12px;
                    border-radius: 1rem;
                    transition: all 0.2s ease;
                    white-space: nowrap;
                    min-width: 80px;
                `;
                voiceButton.innerHTML = '🎵 Voice';
                voiceButton.onclick = (e) => {
                    e.stopPropagation();
                    this.toggleVoiceSelector();
                };
                actionBar.appendChild(voiceButton);
                
                // Add action bar to the wrapper
                messageWrapper.appendChild(actionBar);
                
                // Add the complete wrapper to chat container
                this.chatContainer.appendChild(messageWrapper);
            } else {
                // For user messages, just add the message directly
                this.chatContainer.appendChild(messageDiv);
            }
            
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

    cleanQuotedText(text) {
        if (typeof text === 'string') {
            // Remove leading and trailing quotes
            return text.replace(/^["']|["']$/g, '');
        }
        return text;
    }
    

    basicMarkdownToHtml(markdown) {
        if (!markdown) return '';
    
        // Step 1: Detect if text is Arabic
        const isArabic = this.isArabicText(markdown);
        
        // Step 2: Escape HTML
        let html = markdown
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;')
            .replace(/ - /g, '<li>')
    
        // Step 3: Headers (h2 and h3 only, as per system design)
        html = html
            .replace(/^### (.*?)$/gm, '<h3>$1</h3>')
            .replace(/^## (.*?)$/gm, '<h2>$1</h2>');
    
        // Step 4: Bold formatting
        html = html.replace(/(\*\*|__)(.*?)\1/g, '<strong>$2</strong>');
    
        // Step 5: Currency and percentage formatting
        html = html
            .replace(/QR\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:\/-)?)/g, '<strong>QR $1</strong>')
            .replace(/(\d+(?:\.\d+)?)%/g, '<strong>$1%</strong>')
            // Add break before explanations with strong tags
            .replace(/(<strong>Explanation:<\/strong>)/g, '<br>$1<br>')
            .replace(/\*([^\*\n]+)\*/g, '<strong>$1</strong>');
            
    
        // Step 6: Numbered lists inside sentences
        if (!isArabic) {
            html = html.replace(/([ :.!?])\s*(\d+\.)\s*/g, '$1<br><li class="numbered">');
        } else {
            html = html.replace(/([ :.!?])\s*(\d+\.)\s*/g, '$1<li class="numbered">');
        }
        
        // Step 7: Bullet points inside sentences
        if (!isArabic) {
            html = html.replace(/([ :.!?])\s*[•|*|✓]\s*/g, '$1<br><li>');
        } else {
            html = html.replace(/([ :.!?])\s*[•|*|✓]\s*/g, '$1<li>');
        }

        // Step 8: Regular numbered lists at the start of lines
        html = html
            .replace(/^(\d+\.)\s+(.+)$/gm, '<li class="numbered">$2.</li>');

        // Step 9: Regular bullet points at the start of lines
        html = html
            .replace(/^•\s+(.+)$/gm, '<li>$1</li>')
            .replace(/^\*\s+(.+)$/gm, '<li>$1</li>')
            .replace(/^-\s+(.+)$/gm, '<li>$1</li>');

        // Step 10: Remove any remaining bullet points that might be inside li elements
        html = html.replace(/<li>•\s*/g, '<li>');

        // Step 11: Wrap <li> elements in <ul>
        html = this.wrapListItems(html);

        // Step 12: Line breaks - handle differently for Arabic
        if (!isArabic) {
            html = html.replace(/\n/g, '<br>');
        } else {
            // For Arabic, only add line breaks between paragraphs (double newlines)
            html = html.replace(/\n\n+/g, '<br>').replace(/\n/g, ' ');
        }

        // Step 13: Add text alignment and styling based on language
        if (isArabic) {
            html = `<div style="text-align: right; direction: rtl; font-size: ${this.arabicConfig.fontSize}; font-family: ${this.arabicConfig.fontFamily};">${html}</div>`;
        }

        return html;
    }

    isArabicText(text) {
        // Arabic Unicode range: [\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]
        const arabicPattern = /[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]/;
        
        // Check if the text contains Arabic characters
        // Only consider it Arabic if more than 30% of the non-whitespace characters are Arabic
        const nonWhitespace = text.replace(/\s/g, '');
        const arabicMatches = nonWhitespace.match(new RegExp(arabicPattern, 'g'));
        
        if (!arabicMatches) return false;
        
        return (arabicMatches.length / nonWhitespace.length) > 0.3;
    }

    wrapListItems(html) {
        // Split by <br> to process line by line
        let lines = html.split('<br>');
        let result = [];
        let inList = false;
        let inNumberedList = false;
        
        for (let i = 0; i < lines.length; i++) {
            let line = lines[i].trim();
            
            if (line.startsWith('<li class="numbered">')) {
                // Handle numbered lists
                if (!inNumberedList) {
                    if (inList) {
                        result.push('</ul>');
                        inList = false;
                    }
                    result.push('<ol>');
                    inNumberedList = true;
                }
                result.push(line);
            } else if (line.startsWith('<li>')) {
                // Handle regular bullet lists
                if (!inList) {
                    if (inNumberedList) {
                        result.push('</ol>');
                        inNumberedList = false;
                    }
                    result.push('<ul>');
                    inList = true;
                }
                result.push(line);
            } else {
                // Close any open lists
                if (inList) {
                    result.push('</ul>');
                    inList = false;
                }
                if (inNumberedList) {
                    result.push('</ol>');
                    inNumberedList = false;
                }
                result.push(line);
            }
        }
        
        // Close any open lists
        if (inList) {
            result.push('</ul>');
        }
        if (inNumberedList) {
            result.push('</ol>');
        }
        
        // Join with <br> tags, but don't add <br> inside lists
        let finalHtml = '';
        for (let i = 0; i < result.length; i++) {
            finalHtml += result[i];
            
            // Add <br> between elements, except inside lists
            if (i < result.length - 1) {
                const current = result[i];
                const next = result[i + 1];
                
                // Don't add <br> between list items or list tags
                if (!current.startsWith('<li>') && !next.startsWith('<li>') && 
                    !current.startsWith('<li class="numbered">') && !next.startsWith('<li class="numbered">') &&
                    current !== '<ul>' && current !== '</ul>' &&
                    current !== '<ol>' && current !== '</ol>' &&
                    next !== '<ul>' && next !== '</ul>' &&
                    next !== '<ol>' && next !== '</ol>') {
                    finalHtml += '<br>';
                }
            }
        }
        
        return finalHtml;
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

        // Remove any remaining bullet points that might be inside li elements
        html = html.replace(/<li>•\s*/g, '<li>');
            
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
            .replace(/<p>(\s*<(?:ul|li|h[1-6]|hr)>)/g, '$1')
            .replace(/(<\/(?:ul|li|h[1-6]|hr)>\s*)<\/p>/g, '$1')
            
            // Fix line breaks
            .replace(/\n/g, '<br>');
            
        return html;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    stripHtmlAndDecode(html) {
        if (!html) return '';
        
        // Create a temporary div to convert HTML to text
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = html;
        
        // Get the text content (this removes HTML tags and decodes entities)
        let text = tempDiv.textContent || tempDiv.innerText || '';
        
        // Clean up extra whitespace
        text = text.replace(/\s+/g, ' ').trim();
        
        return text;
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

            console.log('Raw data:', responseData); // Check if quotes are here

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
                
                // Only update beneficiary count if we don't have family data already set
                if (!this.familyData) {
                this.beneficiaryCount.textContent = info.family_count || info.beneficiary_count || info.members_count || '-';
                }

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
        // Only clear beneficiary count if we don't have family data
        if (!this.familyData) {
        this.beneficiaryCount.textContent = '-';
        }
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
                
                // Only update beneficiary count if we don't have family data already set
                if (!this.familyData) {
                this.beneficiaryCount.textContent = info.family_count || info.beneficiary_count || info.members_count || '-';
                }

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
        // Only clear beneficiary count if we don't have family data
        if (!this.familyData) {
        this.beneficiaryCount.textContent = '-';
        }
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

    showFamilyMemberCards() {
        if (!this.familyData) {
            console.error('No family data available');
            return;
        }

        console.log('Showing family member cards with data:', this.familyData);

        const modalBody = document.querySelector('#MemsModal .modal-body.familymem');
        if (!modalBody) {
            console.error('Modal body element not found');
            return;
        }

        const container = document.createElement('div');
        container.className = 'family-cards-container';

        // Handle both array and object structures
        const allMembers = Array.isArray(this.familyData) 
            ? this.familyData 
            : Object.entries(this.familyData).reduce((acc, [relation, members]) => {
                if (Array.isArray(members)) {
                    return acc.concat(members.map(m => ({...m, relation})));
                }
                return acc;
            }, []);

        console.log('Processed members:', allMembers);

        // Sort members by relation priority: Principal first, then Spouse, then Children
        const sortedMembers = allMembers.sort((a, b) => {
            const getRelationPriority = (member) => {
                const relation = (member.relation_order || '');
                if (relation === 3) return 1;
                if (relation === 1) return 2;
                if (relation === 2) return 3;
                return 4; // Any other relation types
            };
            
            const priorityA = getRelationPriority(a);
            const priorityB = getRelationPriority(b);
            
            // If same priority (e.g., multiple children), sort by name
            if (priorityA === priorityB) {
                const nameA = (a.name || '').toLowerCase();
                const nameB = (b.name || '').toLowerCase();
                return nameA.localeCompare(nameB);
            }
            
            return priorityA - priorityB;
        });

        console.log('Sorted members:', sortedMembers);

        sortedMembers.forEach(member => {
            const card = document.createElement('div');
            card.className = 'family-member-card';
            
            const content = document.createElement('div');
            content.className = 'card-content';

            // Top section with name
            const topSection = document.createElement('div');
            const name = document.createElement('div');
            name.className = 'member-name';
            const fullName = member.name || member.individual_name || 'N/A';
            const nameParts = fullName.split(' ');
            const firstName = (nameParts[0] || '').toUpperCase();
            const lastName = (nameParts[nameParts.length - 1] || '').toUpperCase();
            name.textContent = `${firstName} ${lastName}`;
            topSection.appendChild(name);

            // Details section
            const details = document.createElement('div');
            details.className = 'member-details';

            console.log(member);

            // Add member details
            const detailsData = [
                { value: member.company_name || member.individual_name || 'N/A' },
                { value: (member.id || member.individual_id || member.national_id || 'N/A') + ' - ' + member.contract_id || 'N/A' },
                { label: 'C.I.D No.', value: member.national_id || 'N/A' },
                { label: 'Relation', value: member.relation_order === 1 ? 'Spouse' : (member.relation_order === 2 ? 'Child' : 'Principal') },
                { label: 'DOB', value: this.formatDate(member.date_of_birth) || 'N/A' },
                { label: 'Coverage Period', value: this.formatDate(new Date()) + ' - ' + this.formatDate(member.end_date) || 'N/A' },
            ];

            detailsData.forEach(detail => {
                const detailRow = document.createElement('div');
                detailRow.className = 'detail-row';

                const label = document.createElement('div');
                label.className = 'detail-label';
                label.textContent = detail.label;

                const value = document.createElement('div');
                value.className = 'detail-value';
                value.textContent = detail.value;

                detailRow.appendChild(label);
                detailRow.appendChild(value);
                details.appendChild(detailRow);
            });

            // Assemble card
            content.appendChild(topSection);
            content.appendChild(details);
            card.appendChild(content);

            // Add click handler for showing more details
            card.addEventListener('click', () => {
                console.log('Card clicked:', member);
                this.showMemberPolicyDetails(member);
            });

            container.appendChild(card);
        });

        modalBody.innerHTML = '';
        modalBody.appendChild(container);

        // Initialize and show the modal
        const memsModal = new bootstrap.Modal(document.getElementById('MemsModal'));
        
        // Add hidden event listener to clean up backdrop
        const modalElement = document.getElementById('MemsModal');
        modalElement.addEventListener('hidden.bs.modal', () => {
            const backdrop = document.querySelector('.modal-backdrop');
            if (backdrop) {
                backdrop.remove();
            }
            document.body.classList.remove('modal-open');
        });
        
        memsModal.show();
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
                // No need to manually rotate arrow, CSS will handle it
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

            // Handle hover functionality
            sidebarHoverArea.addEventListener('mouseenter', () => {
                if (!sidebar.classList.contains('open')) {
                    sidebar.style.left = '0';
                }
            });

            sidebarHoverArea.addEventListener('mouseleave', (e) => {
                // Check if the mouse moved to the sidebar
                const toElement = e.relatedTarget;
                if (!sidebar.contains(toElement) && !sidebar.classList.contains('open')) {
                    sidebar.style.left = '-250px';
                }
            });

            // Handle sidebar hover
            sidebar.addEventListener('mouseleave', () => {
                // Only close if it wasn't opened by clicking
                if (!sidebar.classList.contains('open')) {
                    sidebar.style.left = '-250px';
                }
            });
        }

        // Set user name in sidebar
        const userName = localStorage.getItem('userName');
        const userNameElement = document.querySelector('.user-info .user-name');
        if (userNameElement && userName) {
            userNameElement.textContent = userName;
        }

        // Add download button to sidebar
        const sidebarActions = document.createElement('div');
        sidebarActions.className = 'sidebar-actions';
        
        const downloadButton = document.createElement('button');
        downloadButton.className = 'download-conversation';
        downloadButton.innerHTML = 'Download Conversation';
        downloadButton.onclick = () => this.downloadConversationAsPDF();
        
        sidebarActions.appendChild(downloadButton);
        
        if (sidebar) {
            sidebar.appendChild(sidebarActions);
        }
    }

    async loadConversations(includeArchived = false) {
        try {
            const token = localStorage.getItem('authToken');
            if (!token) return;

            const response = await fetch(`/api/conversations?include_archived=${includeArchived}`, {
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
        
        conversations.forEach((conv) => {
            const item = document.createElement('div');
            item.className = 'conversation-item';
            item.dataset.id = conv.id;
            
            if (conv.id === this.currentConversationId) {
                item.classList.add('active');
            }
            
            // Add archived class if conversation is archived
            if (conv.archived) {
                item.classList.add('archived');
            }
            
            // Generate title from conversation content
            const title = this.generateConversationTitle(conv.messages);
            
            // Format the date
            const date = new Date(conv.updated_at);
            const formattedDate = date.toLocaleDateString('en-GB', {
                day: '2-digit',
                month: 'short'
            });
            
            // Create main content container
            const contentContainer = document.createElement('div');
            contentContainer.className = 'conversation-content';
            
            // Create title element
            const titleSpan = document.createElement('span');
            titleSpan.className = 'conversation-title';
            titleSpan.textContent = title;
            
            // Create date element
            const dateSpan = document.createElement('span');
            dateSpan.className = 'conversation-date';
            dateSpan.textContent = formattedDate;
            
            // Add archived indicator
            if (conv.archived) {
                const archivedIndicator = document.createElement('span');
                archivedIndicator.className = 'archived-indicator';
                archivedIndicator.textContent = '📥';
                archivedIndicator.title = 'Archived';
                contentContainer.appendChild(archivedIndicator);
            }
            
            // Add elements to content container
            contentContainer.appendChild(titleSpan);
            contentContainer.appendChild(dateSpan);
            
            // Create action buttons container
            const actionsContainer = document.createElement('div');
            actionsContainer.className = 'conversation-actions';
            
            // Archive button
            const archiveBtn = document.createElement('button');
            archiveBtn.className = 'conversation-action-btn archive-btn';
            archiveBtn.innerHTML = conv.archived ? '📥' : '📥';
            archiveBtn.title = conv.archived ? 'Unarchive' : 'Archive';
            archiveBtn.onclick = (e) => {
                e.stopPropagation();
                this.toggleArchiveConversation(conv.id, !conv.archived);
            };
            
            // Delete button
            const deleteBtn = document.createElement('button');
            deleteBtn.className = 'conversation-action-btn delete-btn';
            deleteBtn.innerHTML = '🗑️';
            deleteBtn.title = 'Delete';
            deleteBtn.onclick = (e) => {
                e.stopPropagation();
                this.deleteConversation(conv.id);
            };
            
            actionsContainer.appendChild(archiveBtn);
            actionsContainer.appendChild(deleteBtn);
            
            // Add content and actions to item
            item.appendChild(contentContainer);
            item.appendChild(actionsContainer);
            
            // Set click handler for the main content area
            contentContainer.onclick = () => this.loadConversation(conv.id);
            
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
        
        // Reset family data
        this.familyData = null;
        
        // Reset user info
        this.clearUserInfo();
        
        // Reset beneficiary count styling
        const beneficiaryElement = document.getElementById('beneficiaryCount');
        if (beneficiaryElement) {
            beneficiaryElement.classList.remove('clickable');
            beneficiaryElement.replaceWith(beneficiaryElement.cloneNode(true));
        }
        
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
            
            // Add initial message to chat history
            this.chatHistory.push({
                role: 'assistant',
                content: welcomeMessage.innerHTML
            });
        }
        
        // Remove active class from all conversations
        document.querySelectorAll('.conversation-item').forEach(item => {
            item.classList.remove('active');
        });

        // Update send button state
        this.updateSendButtonState();
        
        // Save the new conversation to get an ID
        await this.saveConversation();
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

                console.log(conversation);
                
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

                // Restore user information
                console.log(conversation.userInfo);
                if (conversation.userInfo) {
                    // Update contractor name
                    if (this.contractorName) {
                        this.contractorName.textContent = conversation.userInfo.contractorName || '-';
                    }

                    // Update expiry date
                    if (this.expiryDate) {
                        this.expiryDate.textContent = conversation.userInfo.expiryDate || '-';
                    }

                    // Update individual name
                    const individualNameElement = document.getElementById('individualName');
                    if (individualNameElement && conversation.userInfo.individualName && conversation.userInfo.individualName !== '-') {
                        individualNameElement.textContent = conversation.userInfo.individualName;
                    }

                    // Update beneficiary count
                    if (this.beneficiaryCount) {
                        console.log(conversation.userInfo.beneficiaryCount);
                        this.beneficiaryCount.textContent = conversation.userInfo.beneficiaryCount || '-';
                        console.log(conversation.userInfo.beneficiaryCount);
                    }

                    // Update national ID state
                    this.currentNationalId = conversation.userInfo.nationalId || '';
                    this.isNationalIdConfirmed = conversation.isNationalIdConfirmed || false;

                    // Update PDF if available
                    const memberWithPdf = conversation.pdfInfo.pdf_link;
                    let pdfInfo = null;
                        if (memberWithPdf) {
                            pdfInfo = {
                                pdf_link: memberWithPdf,
                                company_name: conversation.userInfo.contractorName || 'DIG'
                            };
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

                // Update textarea placeholder based on ID confirmation status
                if (this.textArea) {
                    this.textArea.placeholder = this.isNationalIdConfirmed ? 
                        'Ask anything about your policy...' : 
                        'Enter your 11-digit National ID';
                }

                // Restore suggested questions if available
                if (conversation.suggestedQuestions && this.suggestContainer) {
                    this.suggestContainer.style.display = 'block';
                    const suggestBtn = this.suggestContainer.querySelector('.suggestbtn');
                    if (suggestBtn) {
                        suggestBtn.innerHTML = conversation.suggestedQuestions;
                        // Reattach click handlers to suggested questions
                        suggestBtn.querySelectorAll('button').forEach(button => {
                            button.addEventListener('click', () => {
                                if (this.textArea) {
                                    this.textArea.value = button.textContent;
                                    this.textArea.focus();
                                    this.updateSendButtonState();
                                }
                            });
                        });
                    }
                }

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

            // Get all user information
            const userInfo = {
                contractorName: this.contractorName?.textContent || '-',
                expiryDate: this.expiryDate?.textContent || '-',
                individualName: document.querySelector('.name.col-md-7')?.textContent || '-',
                beneficiaryCount: this.beneficiaryCount?.textContent || '-',
                nationalId: this.currentNationalId || '',
                familyData: this.familyData || null,
                pdfInfo: null
            };

            // Get PDF information if available
            const tobModal = document.getElementById('TOBModal');
            if (tobModal) {
                const pdfFrame = tobModal.querySelector('#pdfFrame');
                if (pdfFrame && pdfFrame.src) {
                    userInfo.pdfInfo = {
                        pdf_link: pdfFrame.src
                    };
                }
            }

            // Create a complete state object
            const conversationState = {
                messages: this.chatHistory,
                userInfo: userInfo,
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

    async copyMessageText(text) {
        try {
            // First try the modern clipboard API
            if (navigator.clipboard && window.isSecureContext) {
                await navigator.clipboard.writeText(text);
                this.showToast('Text copied to clipboard!');
                return;
            }
        } catch (err) {
            console.warn('Clipboard API failed, trying fallback:', err);
        }

        // Fallback method for older browsers or non-secure contexts
        try {
            // Create a temporary textarea element
            const textArea = document.createElement('textarea');
            textArea.value = text;
            textArea.style.position = 'fixed';
            textArea.style.left = '-999999px';
            textArea.style.top = '-999999px';
            document.body.appendChild(textArea);
            textArea.focus();
            textArea.select();
            
            // Try to copy using execCommand
            const successful = document.execCommand('copy');
            document.body.removeChild(textArea);
            
            if (successful) {
                this.showToast('Text copied to clipboard!');
            } else {
                throw new Error('execCommand failed');
            }
        } catch (err) {
            console.error('All copy methods failed:', err);
            
            // Final fallback - show the text in a modal for manual copy
            this.showCopyModal(text);
        }
    }

    showCopyModal(text) {
        // Create modal for manual copy
        const modal = document.createElement('div');
        modal.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.5);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 10000;
        `;
        
        const modalContent = document.createElement('div');
        modalContent.style.cssText = `
            background: white;
            padding: 20px;
            border-radius: 8px;
            max-width: 80%;
            max-height: 80%;
            overflow: auto;
        `;
        
        modalContent.innerHTML = `
            <h3>Copy Message Text</h3>
            <p>Please select and copy the text below:</p>
            <textarea readonly style="width: 100%; height: 200px; margin: 10px 0;">${text}</textarea>
            <button onclick="this.closest('[style*=fixed]').remove()" style="background: #007bff; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer;">Close</button>
        `;
        
        modal.appendChild(modalContent);
        document.body.appendChild(modal);
        
        // Select the text automatically
        const textarea = modalContent.querySelector('textarea');
        textarea.select();
        
        this.showToast('Please manually copy the selected text');
    }

    addClickAnimation(button) {
        button.classList.add('clicked');
        setTimeout(() => {
            button.classList.remove('clicked');
        }, 500);
    }

    likeMessage(button, action) {
        // Add click animation
        button.classList.add('clicked');
        
        // Remove animation class after animation completes
        setTimeout(() => {
            button.classList.remove('clicked');
        }, 400);
        
        // Get the current state of the button
        const isActive = button.classList.contains('active');
        
        if (action === 'like') {
            if (isActive) {
                // Currently active (filled), make it inactive (unfilled)
                button.classList.remove('active');
                button.innerHTML = '<img src="Content/img/like-unfilled.png" alt="like" style="width: 20px; height: 20px;">';
                this.showToast('Like removed');
            } else {
                // Currently inactive (unfilled), make it active (filled)
                button.classList.add('active');
                button.innerHTML = '<img src="Content/img/like-filled.png" alt="like" style="width: 20px; height: 20px;">';
                this.showToast('Thank you for your feedback! 👍');
                
                // Remove dislike if it was active
                const actionBar = button.parentElement;
                const dislikeBtn = actionBar.querySelector('.dislike-button');
                if (dislikeBtn && dislikeBtn.classList.contains('active')) {
                    dislikeBtn.classList.remove('active');
                    dislikeBtn.innerHTML = '<img src="Content/img/dislike-unfilled.png" alt="dislike" style="width: 20px; height: 20px;">';
                }
            }
        } else {
            if (isActive) {
                // Currently active (filled), make it inactive (unfilled)
                button.classList.remove('active');
                button.innerHTML = '<img src="Content/img/dislike-unfilled.png" alt="dislike" style="width: 20px; height: 20px;">';
                this.showToast('Dislike removed');
            } else {
                // Currently inactive (unfilled), make it active (filled)
                button.classList.add('active');
                button.innerHTML = '<img src="Content/img/dislike-filled.png" alt="dislike" style="width: 20px; height: 20px;">';
                this.showToast('Thank you for your feedback! We\'ll work to improve 👎');
                
                // Remove like if it was active
                const actionBar = button.parentElement;
                const likeBtn = actionBar.querySelector('.like-button');
                if (likeBtn && likeBtn.classList.contains('active')) {
                    likeBtn.classList.remove('active');
                    likeBtn.innerHTML = '<img src="Content/img/like-unfilled.png" alt="like" style="width: 20px; height: 20px;">';
                }
            }
        }
        
        // Here you could send feedback to server if needed
        console.log(`User ${action}d the message, active: ${!isActive}`);
    }

    showToast(message, type = 'success') {
        const toast = document.createElement('div');
        toast.className = 'toast';
        
        // Set background color based on type
        let backgroundColor;
        let textColor = 'white';
        switch(type) {
            case 'warning':
                backgroundColor = 'rgba(209,231,221,0.8)';
                textColor = 'rgba(15,81,50,0.8)';
                break;
            case 'error':
                backgroundColor = 'rgba(244, 67, 54, 0.9)';
                break;
            default: // success
                backgroundColor = 'rgba(0, 0, 0, 0.8)';
        }
        
        toast.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 12px 24px;
            border-radius: 4px;
            background: ${backgroundColor};
            color: ${textColor};
            z-index: 1000;
            opacity: 0;
            transform: translateY(20px);
            transition: all 0.3s ease;
            font-size: 14px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        `;
        
        toast.textContent = message;
        document.body.appendChild(toast);
        
        // Trigger animation
        setTimeout(() => {
            toast.style.opacity = '1';
            toast.style.transform = 'translateY(0)';
        }, 10);
        
        // Remove after 3 seconds
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(20px)';
            setTimeout(() => {
                document.body.removeChild(toast);
            }, 300);
        }, 3000);
    }

    // Audio functionality
    async toggleAudio(button, text) {
        // If audio is currently playing, stop it
        if (this.currentAudio && !this.currentAudio.paused) {
            this.stopAudio();
            return;
        }

        try {
            // Check if text is Arabic and using Kokoro provider
            if (this.isArabicText(text) && this.ttsConfig.provider === 'kokoro') {
                this.showToast('Arabic text is not supported with the current voice. Please switch to ElevenLabs voice for Arabic support.', 'warning');
                return;
            }

            // Add loading animation
            this.addClickAnimation(button);
            const originalContent = button.innerHTML;
            button.innerHTML = '⏳'; // Loading indicator
            button.disabled = true;

            // Clean up text for better TTS
            // Add newlines before single bullet symbols (*, •, ✓) that are not followed by another one
            text = text.replace(/(\s)([*•✓])(?![*•✓])/g, '\n$2');
            // Add newlines before numbered lists
            text = text.replace(/(\s)(\d+\.)(?!\d)/g, '\n$2');

            let ttsText = text
                .split('\n')
                .map((line, i, arr) => {
                let trimmed = line.trim();
                const isSingleBullet = /^([*•✓])(?![*•✓])/.test(trimmed);
                const isNumberedItem = /^(\d+\.)(?!\d)/.test(trimmed);

                if (isSingleBullet) {
                const prevIsBullet = i > 0 && /^([*•✓])(?![*•✓])/.test(arr[i - 1].trim());
                const nextIsBullet = i < arr.length - 1 && /^([*•✓])(?![*•✓])/.test(arr[i + 1].trim());

                if (!prevIsBullet) {
                    return line.replace(/^\s*([*•✓])(?![*•✓])\s*/, 'First, ');
                } else if (!nextIsBullet) {
                    return line.replace(/^\s*([*•✓])(?![*•✓])\s*/, 'Finally, ');
                        } else {
                    return line.replace(/^\s*([*•✓])(?![*•✓])\s*/, 'Then, ');
                }
                }
                
                if (isNumberedItem) {
                const prevIsNumbered = i > 0 && /^(\d+\.)(?!\d)/.test(arr[i - 1].trim());
                const nextIsNumbered = i < arr.length - 1 && /^(\d+\.)(?!\d)/.test(arr[i + 1].trim());
                
                // Extract the number and convert to ordinal words
                const numberMatch = trimmed.match(/^(\d+)\./);
                if (numberMatch) {
                    const num = parseInt(numberMatch[1]);
                    const ordinals = ['First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth', 'Seventh', 'Eighth', 'Ninth', 'Tenth'];
                    const ordinal = ordinals[num - 1] || `Number ${num}`;
                    
                    return line.replace(/^\s*(\d+\.)(?!\d)\s*/, `${ordinal}, `);
                        }
                    }
                
                    return line;
                })
            .join(' ')
            .replace(/✅/g, 'Great news! ');

            console.log(ttsText);




            // Show different loading message based on text length
            const textLength = ttsText.length;
            let loadingMessage = 'Generating audio...';
            if (textLength > 10000) {
                loadingMessage = 'Generating audio for very long text (this will take longer, please wait)...';
            } else if (textLength > 5000) {
                loadingMessage = 'Generating audio for long text (this may take a moment)...';
            } else if (textLength > 2000) {
                loadingMessage = 'Generating audio for medium text...';
            }
            this.showToast(loadingMessage);

            // Request TTS from server
            const response = await fetch('/api/tts', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: ttsText,
                    voice: this.ttsConfig.voice,
                    provider: this.ttsConfig.provider
                })
            });

            if (!response.ok) {
                throw new Error(`TTS request failed: ${response.status}`);
            }

            // Create audio blob from response
            const audioBlob = await response.blob();
            const audioUrl = URL.createObjectURL(audioBlob);

            // Stop any currently playing audio
            this.stopAudio();

            // Create and play new audio
            this.currentAudio = new Audio(audioUrl);
            this.currentAudioButton = button;

            // Create speed control container
            const actionBar = button.parentElement;
            const speedControls = document.createElement('div');
            speedControls.className = 'speed-controls';
            speedControls.style.cssText = `
                display: inline-flex;
                align-items: center;
                background: var(--bg-color);
                border: 1px solid rgba(0, 0, 0, 0.2);
                border-radius: 15px;
                user-select: none;
                cursor: pointer;
                margin-left: 5px;
                padding: 2px 8px;
                transition: all 0.2s ease;
            `;

            // Create speed display
            const speedDisplay = document.createElement('span');
            speedDisplay.className = 'speed-display';
            speedDisplay.textContent = '1x';
            speedDisplay.style.cssText = `
                min-width: 45px;
                text-align: center;
                font-size: 14px;
                color: var(--text-color);
                font-weight: 500;
                pointer-events: none;
            `;

            // Add scroll event to control speed
            let isDragging = false;
            let startX = 0;
            let startSpeed = 1;
            const speeds = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2];
            let currentSpeedIndex = 2; // Start at 1x (index 2)

            // Add click event for easy speed cycling
            speedControls.addEventListener('click', (e) => {
                // Only trigger if not dragging
                if (!isDragging) {
                    currentSpeedIndex = (currentSpeedIndex + 1) % speeds.length;
                    const newSpeed = speeds[currentSpeedIndex];
                    this.setAudioSpeed(newSpeed);
                    speedDisplay.textContent = newSpeed + 'x';
                }
            });

            speedControls.addEventListener('mousedown', (e) => {
                isDragging = true;
                startX = e.clientX;
                startSpeed = this.audioSpeed;
                speedControls.style.cursor = 'grabbing';
                
                // Prevent click event from firing when dragging
                setTimeout(() => {
                    isDragging = false;
                }, 150);
            });

            document.addEventListener('mousemove', (e) => {
                if (!isDragging) return;
                
                const diff = e.clientX - startX;
                const speedChange = Math.floor(diff / 30); // Every 30px = 1 speed change
                let newIndex = speeds.indexOf(startSpeed) + speedChange;
                
                // Clamp the index
                newIndex = Math.max(0, Math.min(speeds.length - 1, newIndex));
                
                if (currentSpeedIndex !== newIndex) {
                    currentSpeedIndex = newIndex;
                    const newSpeed = speeds[newIndex];
                    this.setAudioSpeed(newSpeed);
                    speedDisplay.textContent = newSpeed + 'x';
                }
            });

            document.addEventListener('mouseup', () => {
                if (isDragging) {
                    isDragging = false;
                    speedControls.style.cursor = 'pointer';
                }
            });

            // Add wheel event for scroll control
            speedControls.addEventListener('wheel', (e) => {
                e.preventDefault();
                const direction = e.deltaY > 0 ? -1 : 1;
                currentSpeedIndex = Math.max(0, Math.min(speeds.length - 1, currentSpeedIndex + direction));
                const newSpeed = speeds[currentSpeedIndex];
                this.setAudioSpeed(newSpeed);
                speedDisplay.textContent = newSpeed + 'x';
            });

            // Add hover effect
            speedControls.addEventListener('mouseenter', () => {
                speedControls.style.background = 'var(--hover-color, #f0f0f0)';
                speedControls.style.borderColor = 'rgba(0, 0, 0, 0.4)';
                speedControls.style.transform = 'scale(1.05)';
            });

            speedControls.addEventListener('mouseleave', () => {
                speedControls.style.background = 'var(--bg-color)';
                speedControls.style.borderColor = 'rgba(0, 0, 0, 0.2)';
                speedControls.style.transform = 'scale(1)';
            });

            speedControls.appendChild(speedDisplay);
            actionBar.appendChild(speedControls);

            // Set up audio event listeners
            this.currentAudio.onplay = () => {
                button.innerHTML = '<img src="Content/img/stop-button.png" alt="pause" style="width: 20px; height: 20px;">'; // Stop icon
                button.title = 'Stop audio';
                this.showToast('Playing audio...');
                speedControls.style.display = 'inline-flex';
            };

            this.currentAudio.onended = () => {
                this.resetAudioButton();
                URL.revokeObjectURL(audioUrl); // Clean up
                speedControls.remove(); // Remove speed controls
            };

            this.currentAudio.onerror = () => {
                this.resetAudioButton();
                this.showToast('Error playing audio');
                URL.revokeObjectURL(audioUrl); // Clean up
                speedControls.remove(); // Remove speed controls
            };

            // Set initial playback speed
            this.currentAudio.playbackRate = this.audioSpeed;

            // Start playing
            await this.currentAudio.play();

        } catch (error) {
            console.error('Audio error:', error);
            this.showToast('Failed to generate audio');
            this.resetAudioButton();
        } finally {
            button.disabled = false;
        }
    }

    setAudioSpeed(speed) {
        if (this.currentAudio) {
            this.audioSpeed = speed;
            this.currentAudio.playbackRate = speed;
            
            // Update active state of speed buttons
            const speedButtons = document.querySelectorAll('.speed-button');
            speedButtons.forEach(btn => {
                btn.classList.remove('active');
                if (btn.innerHTML === `${speed}x`) {
                    btn.classList.add('active');
                }
            });
            
            this.showToast(`Playback speed: ${speed}x`);
        }
    }

    stopAudio() {
        if (this.currentAudio) {
            this.currentAudio.pause();
            this.currentAudio.currentTime = 0;
            this.currentAudio = null;
            
            // Remove speed controls if they exist
            const speedControls = document.querySelector('.speed-controls');
            if (speedControls) {
                speedControls.remove();
            }
        }
        this.resetAudioButton();
    }

    resetAudioButton() {
        if (this.currentAudioButton) {
            this.currentAudioButton.innerHTML = '<img src="Content/img/volume.png" alt="audio" style="width: 20px; height: 20px;">';
            this.currentAudioButton.title = 'Listen to message';
            this.currentAudioButton = null;
        }
    }

    async downloadConversationAsPDF() {
        try {
            // Create content for PDF
            let content = '';
            
            // Add user information
            content += 'Insurance Policy Information\n';
            content += '===========================\n\n';
            content += `Contractor Name: ${this.contractorName?.textContent || '-'}\n`;
            content += `Individual Name: ${document.querySelector('.name.col-md-7')?.textContent || '-'}\n`;
            content += `Contact Expiry Date: ${this.expiryDate?.textContent || '-'}\n`;
            content += `No. of Beneficiaries: ${this.beneficiaryCount?.textContent || '-'}\n\n`;
            
            content += 'Conversation History\n';
            content += '===================\n\n';
            
            // Add chat history with enhanced formatting
            this.chatHistory.forEach((msg, index) => {
                // Skip the ID verification message
                if (msg.role === 'assistant' && msg.content.includes('ID verified successfully')) {
                    return; // Skip this message
                }
                
                // Process the content with the new formatting approach
                let cleanContent = this.formatContentForPDF(msg.content);
                
                // Add the message to content with proper spacing
                content += `${msg.role === 'user' ? 'You' : 'Assistant'}:\n${cleanContent}\n\n`;
            });

            // Create PDF using jsPDF
            const { jsPDF } = window.jspdf;
            const doc = new jsPDF();

            // Set font to ensure proper character rendering
            doc.setFont('helvetica', 'normal');
            doc.setFontSize(11);

            // Split text into lines that fit the page width, preserving words
            const pageWidth = doc.internal.pageSize.getWidth();
            const margin = 15;
            const maxLineWidth = pageWidth - (margin * 2);
            
            // Split content into paragraphs first
            const paragraphs = content.split('\n');
            const lines = [];
            
            paragraphs.forEach(paragraph => {
                if (paragraph.trim() === '') {
                    lines.push(''); // Empty line
                } else {
                    // Split long paragraphs into multiple lines
                    const wrappedLines = doc.splitTextToSize(paragraph, maxLineWidth);
                    lines.push(...wrappedLines);
                }
            });
            
            // Add lines to PDF with enhanced formatting
            let yPosition = 20;
            const lineHeight = 6;
            
            lines.forEach(line => {
                // Check if we need a new page
                if (yPosition > doc.internal.pageSize.getHeight() - 20) {
                    doc.addPage();
                    yPosition = 20;
                }
                
                // Handle bullet points with proper indentation
                if (line.trim().startsWith('• ')) {
                    doc.text(line, margin + 10, yPosition); // Indent bullet points
                } else {
                doc.text(line, margin, yPosition);
                }
                yPosition += lineHeight;
            });

            // Save the PDF
            const timestamp = new Date().toISOString().slice(0, 19).replace(/[:.]/g, '-');
            doc.save(`conversation-${timestamp}.pdf`);
            
            this.showToast('Conversation downloaded as PDF!');
        } catch (error) {
            console.error('Error generating PDF:', error);
            this.showToast('Failed to download conversation');
        }
    }

    // Updated helper function to format content for PDF with new processing
    formatContentForPDF(content) {
        if (!content) return '';
        
        // First, process the content similar to the new basicMarkdownToHtml approach
        let cleanContent = content;
        
        // Step 1: Handle HTML entities and tags
        cleanContent = cleanContent
            // Convert strong tags to plain text (remove HTML but keep content)
            .replace(/<strong>(.*?)<\/strong>/g, '$1')
            
            // Convert ordered list structure to numbered points
            .replace(/<ol>/g, '')
            .replace(/<\/ol>/g, '')
            .replace(/<li class="numbered">(\d+\.)\s*(.*?)<\/li>/g, '$1 $2')
            
            // Convert unordered list structure to bullet points
            .replace(/<ul>/g, '')
            .replace(/<\/ul>/g, '')
            .replace(/<li>(.*?)<\/li>/g, '• $1')
            
            // Convert line breaks to newlines
            .replace(/<br\s*\/?>/g, '\n')
            
            // Remove any remaining HTML tags
            .replace(/<[^>]*>/g, '');
        
        // Step 2: Decode HTML entities
        const textarea = document.createElement('textarea');
        textarea.innerHTML = cleanContent;
        cleanContent = textarea.value;
        
        // Step 3: Clean up formatting to match the new approach
        cleanContent = cleanContent
            // Handle numbered lists that come after sentences
            .replace(/([.!?])\s*(\d+\.)\s*/g, '$1\n$2 ')
            
            // Handle bullet points that come after sentences
            .replace(/([.!?])\s*•\s*/g, '$1\n• ')
            
            // Normalize numbered list spacing
            .replace(/\s*(\d+\.)\s*/g, '\n$1 ')
            
            // Normalize bullet point spacing
            .replace(/\s*•\s*/g, '\n• ')
            
            // Fix multiple spaces
            .replace(/[ \t]+/g, ' ')
            
            // Clean up excessive newlines but preserve paragraph breaks
            .replace(/\n{3,}/g, '\n\n')
            
            // Ensure numbered lists are on new lines
            .replace(/([^\n])(\d+\.)/g, '$1\n$2')
            
            // Ensure bullet points are on new lines
            .replace(/([^\n])•/g, '$1\n•')
            
            // Trim whitespace
            .trim();
        
        return cleanContent;
    }

    async loadJsPDF() {
        try {
            // Create script element
            const script = document.createElement('script');
            script.src = 'https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js';
            script.async = true;
            
            // Wait for script to load
            await new Promise((resolve, reject) => {
                script.onload = resolve;
                script.onerror = reject;
                document.head.appendChild(script);
            });
            
            console.log('jsPDF loaded successfully');
        } catch (error) {
            console.error('Failed to load jsPDF:', error);
        }
    }

    // Add after constructor initialization
    generateConversationTitle(messages) {
        try {
            // Enhanced validation for messages parameter
            if (!messages) {
                console.log('generateConversationTitle: messages is null/undefined');
                return 'New Policy Inquiry';
            }
            
            // Check if messages is an array
            if (!Array.isArray(messages)) {
                console.log('generateConversationTitle: messages is not an array:', typeof messages, messages);
                return 'New Policy Inquiry';
            }
            
            // Check if array is empty
            if (messages.length === 0) {
                console.log('generateConversationTitle: messages array is empty');
                return 'New Policy Inquiry';
            }

            // Key topics and their related keywords
            const topics = {
                coverage: ['coverage', 'limit', 'benefits', 'covered', 'insurance', 'policy'],
                network: ['network', 'hospital', 'clinic', 'provider', 'facility', 'medical center'],
                claims: ['claim', 'reimbursement', 'payment', 'bill', 'invoice'],
                family: ['dependent', 'spouse', 'child', 'family', 'member', 'beneficiary'],
                maternity: ['maternity', 'pregnancy', 'delivery', 'prenatal', 'postnatal'],
                dental: ['dental', 'teeth', 'orthodontic', 'dentist'],
                optical: ['optical', 'vision', 'eye', 'glasses', 'contact lens'],
                medication: ['medicine', 'pharmacy', 'prescription', 'drug'],
                preapproval: ['approval', 'pre-approval', 'authorization', 'pre-auth'],
                general: ['information', 'details', 'explain', 'tell', 'what', 'how']
            };

            // Find the first substantive question after ID verification
            let relevantMessages = [];
            let hasIdVerification = false;
            let nationalId = '';
            
            // Safely iterate through messages
            for (let i = 0; i < messages.length; i++) {
                const msg = messages[i];
                
                // Validate message structure
                if (!msg || typeof msg !== 'object' || !msg.role || !msg.content) {
                    console.log('generateConversationTitle: Invalid message structure at index', i, msg);
                    continue;
                }
                
                // Extract National ID if present
                if (msg.role === 'user' && /^\d{11}$/.test(msg.content.trim())) {
                    nationalId = msg.content.trim();
                    continue;
                }

                // Check for ID verification
                if (msg.role === 'assistant' && msg.content.includes('ID verified successfully')) {
                    hasIdVerification = true;
                    continue;
                }

                // Collect relevant messages after ID verification
                if (hasIdVerification && msg.role === 'user') {
                    relevantMessages.push(msg.content.toLowerCase());
                }
            }

            // If we have relevant messages, analyze them for topics
            if (relevantMessages.length > 0) {
                // Count topic occurrences
                const topicCounts = {};
                for (const [topic, keywords] of Object.entries(topics)) {
                    topicCounts[topic] = 0;
                    keywords.forEach(keyword => {
                        relevantMessages.forEach(msg => {
                            if (msg.includes(keyword)) {
                                topicCounts[topic]++;
                            }
                        });
                    });
                }

                // Get the most discussed topic
                const mainTopic = Object.entries(topicCounts)
                    .sort((a, b) => b[1] - a[1])
                    .filter(([_, count]) => count > 0)[0]?.[0];

                // Generate title based on main topic and first question
                if (mainTopic) {
                    const topicTitles = {
                        coverage: 'Policy Coverage Inquiry',
                        network: 'Medical Network Discussion',
                        claims: 'Claims and Reimbursement',
                        family: 'Family Coverage Details',
                        maternity: 'Maternity Benefits',
                        dental: 'Dental Coverage Inquiry',
                        optical: 'Optical Benefits',
                        medication: 'Medication Coverage',
                        preapproval: 'Pre-approval Request',
                        general: 'General Policy Information'
                    };

                    // If we have a first question, make the title more specific
                    if (relevantMessages[0]) {
                        const questionWords = relevantMessages[0]
                            .replace(/[?!.,]/g, '')
                            .split(' ')
                            .slice(0, 3)
                            .join(' ');
                        
                        return `${topicTitles[mainTopic]} - ${questionWords}...`;
                    }

                    return topicTitles[mainTopic];
                }
            }

            // If we have a national ID but no topic identified
            if (nationalId) {
                return `Policy Inquiry - ID: ${nationalId.slice(-4)}`;
            }

            // If ID verified but no specific topic
            if (hasIdVerification) {
                return 'Medical Policy Information';
            }

            // Default meaningful title for new conversations
            return 'New conversation';
        } catch (error) {
            console.error('Error generating title:', error);
            console.error('Messages parameter:', messages);
            return 'Medical Policy Inquiry';
        }
    }

    async deleteConversation(conversationId) {
        // Show confirmation dialog
        if (!confirm('Are you sure you want to delete this conversation? This action cannot be undone.')) {
            return;
        }

        try {
            const token = localStorage.getItem('authToken');
            if (!token) return;

            const response = await fetch(`/api/conversations/${conversationId}`, {
                method: 'DELETE',
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });

            if (response.ok) {
                // If the deleted conversation was currently active, start a new one
                if (this.currentConversationId === conversationId) {
                    await this.startNewConversation();
                }
                
                // Refresh conversation list
                await this.loadConversations();
                this.showToast('Conversation deleted successfully');
            } else {
                throw new Error('Failed to delete conversation');
            }
        } catch (error) {
            console.error('Error deleting conversation:', error);
            this.showToast('Failed to delete conversation', 'error');
        }
    }

    async toggleArchiveConversation(conversationId, archived) {
        try {
            const token = localStorage.getItem('authToken');
            if (!token) return;

            const response = await fetch(`/api/conversations/${conversationId}/archive?archived=${archived}`, {
                method: 'PATCH',
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });

            if (response.ok) {
                // If the archived conversation was currently active, start a new one
                if (archived && this.currentConversationId === conversationId) {
                    await this.startNewConversation();
                }
                
                // Refresh conversation list
                await this.loadConversations();
                
                const action = archived ? 'archived' : 'unarchived';
                this.showToast(`Conversation ${action} successfully`);
            } else {
                throw new Error(`Failed to ${archived ? 'archive' : 'unarchive'} conversation`);
            }
        } catch (error) {
            console.error('Error toggling archive status:', error);
            const action = archived ? 'archive' : 'unarchive';
            this.showToast(`Failed to ${action} conversation`, 'error');
        }
    }

    // Add method to change Arabic font
    setArabicFont(fontFamily) {
        this.arabicConfig.fontFamily = fontFamily;
        localStorage.setItem('arabicFont', fontFamily);
        // Refresh existing messages if needed
        this.refreshChatMessages();
    }

    // Add method to refresh chat messages when font changes
    refreshChatMessages() {
        if (!this.chatContainer || !this.chatHistory) return;
        
        // Store the scroll position
        const scrollPos = this.chatContainer.scrollTop;
        
        // Clear and rebuild messages
        this.chatContainer.innerHTML = '';
        this.chatHistory.forEach(msg => {
            this.addMessageToChat(msg.role, msg.content, false); // false to prevent saving again
        });
        
        // Restore scroll position
        this.chatContainer.scrollTop = scrollPos;
    }

    // Add method to get available Arabic fonts
    getAvailableArabicFonts() {
        return [
            'Arial',
            'Tahoma',
            'Dubai',
            'Amiri',
            'Scheherazade',
            'Noto Naskh Arabic',
            'Traditional Arabic',
            'Simplified Arabic'
        ];
    }

    async loadTTSVoices() {
        try {
            const response = await fetch('/api/tts/voices');
            if (response.ok) {
                this.ttsConfig.availableVoices = await response.json();
                console.log('Loaded TTS voices:', this.ttsConfig.availableVoices);
            } else {
                console.error('Failed to load TTS voices');
                // Use fallback configuration
                this.ttsConfig.availableVoices = {
                    providers: {
                        kokoro: {
                            available: true,
                            voices: {
                                sarah: { name: "Sarah", description: "Natural female voice", language: "English" },
                                emma: { name: "Emma", description: "Expressive female voice", language: "English" }
                            }
                        }
                    },
                    default: { provider: "kokoro", voice: "sarah" }
                };
            }
        } catch (error) {
            console.error('Error loading TTS voices:', error);
            // Use fallback configuration
            this.ttsConfig.availableVoices = {
                providers: {
                    kokoro: {
                        available: true,
                        voices: {
                            sarah: { name: "Sarah", description: "Natural female voice", language: "English" },
                            emma: { name: "Emma", description: "Expressive female voice", language: "English" }
                        }
                    }
                },
                default: { provider: "kokoro", voice: "sarah" }
            };
        }
    }

    createVoiceSelector() {
        // Check if voice selector already exists
        if (document.querySelector('.voice-selector')) {
            return;
        }

        // Create voice selector container
        const voiceSelector = document.createElement('div');
        voiceSelector.className = 'voice-selector';
        voiceSelector.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: var(--bg-color, #ffffff);
            border: 1px solid rgba(0, 0, 0, 0.2);
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            z-index: 1000;
            min-width: 280px;
            transition: all 0.3s ease;
        `;

        // Create title
        const title = document.createElement('div');
        title.style.cssText = `
            font-weight: bold;
            margin-bottom: 12px;
            color: var(--text-color, #333333);
            font-size: 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        `;
        title.innerHTML = `
            🎵 Voice Settings
            <span style="cursor: pointer; font-size: 18px; color: #666;" onclick="this.parentElement.parentElement.style.display='none'">×</span>
        `;

        // Create voice options
        const voiceOptions = document.createElement('div');
        voiceOptions.style.cssText = `
            display: flex;
            flex-direction: column;
            gap: 8px;
        `;

        // Add voice options for each provider
        const providers = this.ttsConfig.availableVoices?.providers || {};
        
        Object.entries(providers).forEach(([providerId, providerData]) => {
            if (!providerData.available) return;
            
            Object.entries(providerData.voices).forEach(([voiceId, voiceData]) => {
                const option = document.createElement('label');
                option.style.cssText = `
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    padding: 8px 12px;
                    border-radius: 6px;
                    cursor: pointer;
                    background: var(--hover-color, #f8f9fa);
                    transition: all 0.2s ease;
                `;
                
                const isSelected = this.ttsConfig.provider === providerId && this.ttsConfig.voice === voiceId;
                
                option.innerHTML = `
                    <input type="radio" name="voice" value="${providerId}:${voiceId}" ${isSelected ? 'checked' : ''} 
                           style="margin: 0;">
                    <div>
                        <div style="font-weight: 600; color: var(--text-color, #333);">${voiceData.name}</div>
                        <div style="font-size: 12px; color: #666;">${voiceData.description}</div>
                    </div>
                `;
                
                option.addEventListener('change', (e) => {
                    if (e.target.checked) {
                        const [provider, voice] = e.target.value.split(':');
                        this.updateVoiceSettings(provider, voice);
                        this.showToast(`Voice changed to ${voiceData.name}`);
                    }
                });
                
                voiceOptions.appendChild(option);
            });
        });

        // Assemble the selector
        voiceSelector.appendChild(title);
        voiceSelector.appendChild(voiceOptions);

        // Add to page
        document.body.appendChild(voiceSelector);

        // Make it draggable (optional)
        this.makeDraggable(voiceSelector);
    }

    updateVoiceSettings(provider, voice) {
        this.ttsConfig.provider = provider;
        this.ttsConfig.voice = voice;
        
        // Save to localStorage
        localStorage.setItem('ttsProvider', provider);
        localStorage.setItem('ttsVoice', voice);
        
        console.log(`TTS settings updated: ${provider}:${voice}`);
    }

    makeDraggable(element) {
        let isDragging = false;
        let startX, startY, startLeft, startTop;

        const title = element.querySelector('div');
        title.style.cursor = 'move';

        title.addEventListener('mousedown', (e) => {
            isDragging = true;
            startX = e.clientX;
            startY = e.clientY;
            startLeft = parseInt(window.getComputedStyle(element).left, 10) || 0;
            startTop = parseInt(window.getComputedStyle(element).top, 10) || 0;
            e.preventDefault();
        });

        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            
            const newLeft = startLeft + e.clientX - startX;
            const newTop = startTop + e.clientY - startY;
            
            element.style.left = `${newLeft}px`;
            element.style.top = `${newTop}px`;
            element.style.right = 'auto';
        });

        document.addEventListener('mouseup', () => {
            isDragging = false;
        });
    }

    toggleVoiceSelector() {
        const existingSelector = document.querySelector('.voice-selector');
        if (existingSelector) {
            existingSelector.style.display = existingSelector.style.display === 'none' ? 'block' : 'none';
        } else {
            this.createVoiceSelector();
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

// Global logout function for the logout button
function handleLogout() {
    console.log('handleLogout called');
    
    try {
        // Try to use Auth.logout() first
        if (typeof Auth !== 'undefined' && Auth.logout) {
            console.log('Using Auth.logout()');
            Auth.logout();
            return;
        }
    } catch (error) {
        console.error('Error with Auth.logout():', error);
    }
    
    // Fallback logout method
    console.log('Using fallback logout method');
    try {
        // Clear all localStorage items
        localStorage.removeItem('authToken');
        localStorage.removeItem('userName');
        localStorage.removeItem('userEmail');
        localStorage.removeItem('ttsProvider');
        localStorage.removeItem('ttsVoice');
        localStorage.removeItem('arabicFont');
        
        console.log('LocalStorage cleared, redirecting to login...');
        
        // Show a brief logout message
        if (window.insuranceAssistant && window.insuranceAssistant.showToast) {
            window.insuranceAssistant.showToast('Logging out...', 'success');
        }
        
        // Redirect to login page after a brief delay
        setTimeout(() => {
            window.location.href = '/login.html';
        }, 500);
        
    } catch (error) {
        console.error('Error during fallback logout:', error);
        // Force immediate redirect as last resort
        window.location.href = '/login.html';
    }
}

// Make handleLogout globally available
window.handleLogout = handleLogout;