class Auth {
    constructor() {
        this.form = document.querySelector('form#loginForm') || 
                   document.querySelector('form#signupForm') ||
                   document.querySelector('form#forgotPasswordForm') ||
                   document.querySelector('form#resetPasswordForm');
        this.setupEventListeners();
        
        // Check token expiration periodically
        this.startTokenCheck();

        // Handle different page types
        this.handlePageSpecificLogic();
    }

    handlePageSpecificLogic() {
        const currentPath = window.location.pathname;
        
        if (currentPath.includes('reset-password.html')) {
            this.handleResetPasswordPage();
        } else if (currentPath.includes('signup.html')) {
            this.setupSignupValidation();
        }
    }

    async handleResetPasswordPage() {
        // Get token from URL
        const urlParams = new URLSearchParams(window.location.search);
        const token = urlParams.get('token');

        console.log('Reset password page loaded with token:', token ? 'present' : 'missing');

        if (!token) {
            this.showError('Invalid reset link. Please request a new password reset.');
            return;
        }

        try {
            console.log('Validating reset token...');
            // Validate token
            const response = await fetch(`/api/validate-reset-token/${token}`);
            const data = await response.json();

            console.log('Token validation response:', response.status, data);

            if (response.ok) {
                console.log('Token is valid, showing reset form');
                // Token is valid, show reset form
                document.getElementById('loadingContainer').style.display = 'none';
                document.getElementById('resetFormContainer').style.display = 'block';
                
                // Show user info
                const userInfo = document.getElementById('userInfo');
                userInfo.innerHTML = `
                    <strong>Reset password for</strong><br>
                    <span style="font-size: 11pt;">${data.user_name} (${data.email})</span>
                `;

                // Setup password validation for reset form
                this.setupPasswordValidation();
            } else {
                console.log('Token validation failed:', data.detail);
                this.showError(data.detail || 'Invalid or expired reset token.');
            }
        } catch (error) {
            console.error('Token validation error:', error);
            this.showError('An error occurred while validating your reset link.');
        }
    }

    showError(message) {
        document.getElementById('loadingContainer').style.display = 'none';
        document.getElementById('resetFormContainer').style.display = 'none';
        document.getElementById('errorContainer').style.display = 'block';
        document.getElementById('errorMessage').textContent = message;
    }

    setupEventListeners() {
        if (this.form) {
            this.form.addEventListener('submit', async (e) => {
                e.preventDefault();
                if (this.form.id === 'loginForm') {
                    await this.handleLogin();
                } else if (this.form.id === 'signupForm') {
                    await this.handleSignup();
                } else if (this.form.id === 'forgotPasswordForm') {
                    await this.handleForgotPassword();
                } else if (this.form.id === 'resetPasswordForm') {
                    await this.handleResetPasswordSubmit(e);
                }
            });
        }
    }

    startTokenCheck() {
        // Don't start token checking on auth pages
        const currentPath = window.location.pathname;
        const authPages = ['/login.html', '/signup.html', '/forgot-password.html', '/reset-password.html'];
        const isAuthPage = authPages.some(page => currentPath.includes(page));
        
        if (isAuthPage) {
            console.log('Auth page detected, skipping token validity checks');
            return;
        }
        
        // Check token every minute only for protected pages
        setInterval(() => {
            this.checkTokenValidity();
        }, 60000); // 60000 ms = 1 minute
    }

    checkTokenValidity() {
        // Don't check token validity on auth pages
        const currentPath = window.location.pathname;
        const authPages = ['/login.html', '/signup.html', '/forgot-password.html', '/reset-password.html'];
        const isAuthPage = authPages.some(page => currentPath.includes(page));
        
        if (isAuthPage) {
            return; // Skip token checks on auth pages
        }
        
        const token = localStorage.getItem('authToken');
        if (!token) {
            this.redirectToLogin();
            return;
        }

        // Check if token is expired
        try {
            const tokenData = JSON.parse(atob(token.split('.')[1]));
            if (tokenData.exp * 1000 < Date.now()) {
                // Token is expired
                this.redirectToLogin();
            }
        } catch (e) {
            // If there's any error parsing the token, redirect to login
            this.redirectToLogin();
        }
    }

    redirectToLogin() {
        localStorage.removeItem('authToken');
        localStorage.removeItem('userName');
        localStorage.removeItem('userEmail');
        window.location.href = '/login.html';
    }

    async handleLogin() {
        const email = document.getElementById('email').value;
        const password = document.getElementById('password').value;

        try {
            const response = await fetch('/api/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ email, password })
            });

            const data = await response.json();

            if (response.ok) {
                // Store the token
                localStorage.setItem('authToken', data.access_token);
                localStorage.setItem('userName', data.name);
                localStorage.setItem('userEmail', email);
                
                // Redirect to main chat page
                window.location.href = '/index.html';
            } else {
                alert(data.detail || 'Login failed. Please try again.');
            }
        } catch (error) {
            console.error('Login error:', error);
            alert('Login failed. Please try again.');
        }
    }

    async handleSignup() {
        const name = document.getElementById('name').value;
        const email = document.getElementById('email').value;
        const password = document.getElementById('password').value;
        const confirmPassword = document.getElementById('confirmPassword').value;

        // Enhanced password validation with detailed feedback
        const validation = this.validatePasswordWithDetails(password);
        if (!validation.isValid) {
            this.showPasswordRequirementsPopup(validation.failedRequirements);
            return;
        }

        // Check password match
        if (password !== confirmPassword) {
            this.showMessage('Passwords do not match. Please make sure both password fields are identical.', 'error');
            return;
        }

        const signupBtn = document.querySelector('button[type="submit"]');
        const originalText = signupBtn.textContent;
        
        // Show loading state
        signupBtn.disabled = true;
        signupBtn.textContent = 'Creating Account...';

        try {
            const response = await fetch('/api/signup', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    name: name,
                    email: email,
                    password: password
                })
            });

            const data = await response.json();

            if (response.ok) {
                this.showMessage('🎉 Account created successfully! Redirecting to login page...', 'success');
                setTimeout(() => {
                    window.location.href = '/login.html';
                }, 2000);
            } else {
                // Handle specific error messages from backend
                let errorMessage = data.detail || 'Failed to create account';
                
                if (errorMessage.includes('Password validation failed')) {
                    // Parse server-side validation errors and show popup
                    const serverErrors = errorMessage.split(': ')[1] || errorMessage;
                    const errorList = serverErrors.split(';').map(error => error.trim()).filter(error => error);
                    this.showPasswordRequirementsPopup(errorList);
                    return;
                } else if (errorMessage.includes('Email already registered')) {
                    errorMessage = '📧 An account with this email address already exists. Please use a different email or try logging in.';
                }
                
                this.showMessage(errorMessage, 'error');
            }
        } catch (error) {
            console.error('Signup error:', error);
            this.showMessage('🌐 Network error. Please check your internet connection and try again.', 'error');
        } finally {
            // Re-enable button
            signupBtn.disabled = false;
            signupBtn.textContent = originalText;
        }
    }

    async handleForgotPassword() {
        const email = document.getElementById('email').value;
        const submitBtn = document.getElementById('submitBtn');
        const messageContainer = document.getElementById('messageContainer');

        // Show loading state
        submitBtn.disabled = true;
        submitBtn.textContent = 'Sending...';

        try {
            const response = await fetch('/api/forgot-password', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ email })
            });

            const data = await response.json();

            if (response.ok) {
                messageContainer.innerHTML = `
                    <div class="message success">
                        ${data.message}
                    </div>
                `;
                // Clear the form
                document.getElementById('email').value = '';
            } else {
                messageContainer.innerHTML = `
                    <div class="message error">
                        ${data.detail || 'An error occurred. Please try again.'}
                    </div>
                `;
            }
        } catch (error) {
            console.error('Forgot password error:', error);
            messageContainer.innerHTML = `
                <div class="message error">
                    Network error. Please check your connection and try again.
                </div>
            `;
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = 'Send Reset Link';
        }
    }

    // Original password validation for real-time feedback (while typing)
    validatePassword(password) {
        const requirements = {
            length: password.length >= 8,
            uppercase: /[A-Z]/.test(password),
            lowercase: /[a-z]/.test(password),
            number: /\d/.test(password),
            special: /[!@#$%^&*(),.?":{}|<>]/.test(password),
            noSpaces: !/\s/.test(password)
        };

        // Check for weak patterns
        const weakPatterns = ['123456', 'password', 'qwerty', 'abc123', 'admin'];
        const hasWeakPattern = weakPatterns.some(pattern => 
            password.toLowerCase().includes(pattern)
        );

        return {
            isValid: Object.values(requirements).every(req => req) && !hasWeakPattern,
            requirements: requirements,
            hasWeakPattern: hasWeakPattern
        };
    }

    // Enhanced password validation with detailed feedback
    validatePasswordWithDetails(password) {
        const requirements = {
            length: {
                valid: password.length >= 8,
                message: "Password must be at least 8 characters long"
            },
            uppercase: {
                valid: /[A-Z]/.test(password),
                message: "Password must contain at least one uppercase letter (A-Z)"
            },
            lowercase: {
                valid: /[a-z]/.test(password),
                message: "Password must contain at least one lowercase letter (a-z)"
            },
            number: {
                valid: /\d/.test(password),
                message: "Password must contain at least one number (0-9)"
            },
            special: {
                valid: /[!@#$%^&*(),.?":{}|<>]/.test(password),
                message: "Password must contain at least one special character (!@#$%^&*(),.?\":{}|<>)"
            },
            noSpaces: {
                valid: !/\s/.test(password),
                message: "Password must not contain spaces"
            }
        };

        // Check for weak patterns
        const weakPatterns = [
            { pattern: '123456', message: "Password cannot contain '123456'" },
            { pattern: 'password', message: "Password cannot contain the word 'password'" },
            { pattern: 'qwerty', message: "Password cannot contain 'qwerty'" },
            { pattern: 'abc123', message: "Password cannot contain 'abc123'" },
            { pattern: 'admin', message: "Password cannot contain 'admin'" }
        ];

        const weakPattern = weakPatterns.find(wp => 
            password.toLowerCase().includes(wp.pattern)
        );

        // Collect all failed requirements
        const failedRequirements = [];
        
        Object.entries(requirements).forEach(([key, requirement]) => {
            if (!requirement.valid) {
                failedRequirements.push(requirement.message);
            }
        });

        if (weakPattern) {
            failedRequirements.push(weakPattern.message);
        }

        const isValid = failedRequirements.length === 0;

        return {
            isValid: isValid,
            requirements: Object.fromEntries(
                Object.entries(requirements).map(([key, req]) => [key, req.valid])
            ),
            failedRequirements: failedRequirements,
            hasWeakPattern: !!weakPattern
        };
    }

    // Show detailed password requirements popup
    showPasswordRequirementsPopup(failedRequirements) {
        const popupHtml = `
            <div id="passwordRequirementsPopup" style="
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: white;
                border: 2px solid #dc3545;
                border-radius: 10px;
                padding: 20px;
                max-width: 400px;
                width: 90%;
                z-index: 10000;
                box-shadow: 0 4px 20px rgba(0,0,0,0.3);
                font-family: 'Poppins', sans-serif;
            ">
                <div style="
                    display: flex;
                    align-items: center;
                    margin-bottom: 15px;
                    color: #dc3545;
                ">
                    <span style="font-size: 24px; margin-right: 10px;">⚠️</span>
                    <h3 style="margin: 0; font-size: 16px; font-weight: 600;">Password Requirements Not Met</h3>
                </div>
                
                <div style="margin-bottom: 20px; color: #666; font-size: 14px;">
                    Please fix the following issues with your password:
                </div>
                
                <ul style="
                    list-style: none;
                    padding: 0;
                    margin: 0 0 20px 0;
                    color: #dc3545;
                    font-size: 13px;
                ">
                    ${failedRequirements.map(req => `
                        <li style="
                            margin-bottom: 8px;
                            padding-left: 20px;
                            position: relative;
                        ">
                            <span style="
                                position: absolute;
                                left: 0;
                                color: #dc3545;
                                font-weight: bold;
                            ">✗</span>
                            ${req}
                        </li>
                    `).join('')}
                </ul>
                
                <button onclick="this.parentElement.remove()" style="
                    background: #dc3545;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 14px;
                    font-weight: 500;
                    width: 100%;
                    transition: background-color 0.2s ease;
                " onmouseover="this.style.backgroundColor='#c82333'" onmouseout="this.style.backgroundColor='#dc3545'">
                    Got it, I'll fix my password
                </button>
            </div>
            
            <div id="passwordRequirementsOverlay" onclick="document.getElementById('passwordRequirementsPopup').remove(); this.remove();" style="
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0,0,0,0.5);
                z-index: 9999;
            "></div>
        `;

        // Remove existing popup if any
        const existingPopup = document.getElementById('passwordRequirementsPopup');
        if (existingPopup) {
            existingPopup.remove();
        }
        const existingOverlay = document.getElementById('passwordRequirementsOverlay');
        if (existingOverlay) {
            existingOverlay.remove();
        }

        // Add popup to page
        document.body.insertAdjacentHTML('beforeend', popupHtml);

        // Auto-remove after 10 seconds
        setTimeout(() => {
            const popup = document.getElementById('passwordRequirementsPopup');
            const overlay = document.getElementById('passwordRequirementsOverlay');
            if (popup) popup.remove();
            if (overlay) overlay.remove();
        }, 10000);
    }

    // Enhanced showMessage method for consistent styling
    showMessage(message, type) {
        const messageContainer = document.getElementById('messageContainer');
        if (!messageContainer) return;

        const messageClass = type === 'error' ? 'error' : 'success';
        messageContainer.innerHTML = `
            <div class="message ${messageClass}">
                ${message}
            </div>
        `;

        // Auto-hide success messages after 5 seconds
        if (type === 'success') {
            setTimeout(() => {
                if (messageContainer.innerHTML.includes(message)) {
                    messageContainer.innerHTML = '';
                }
            }, 5000);
        }
    }

    updatePasswordRequirements(password) {
        const validation = this.validatePassword(password);
        const requirements = validation.requirements;

        // Update requirement indicators
        this.updateRequirement('req-length', requirements.length);
        this.updateRequirement('req-uppercase', requirements.uppercase);
        this.updateRequirement('req-lowercase', requirements.lowercase);
        this.updateRequirement('req-number', requirements.number);
        this.updateRequirement('req-special', requirements.special);
        this.updateRequirement('req-no-spaces', requirements.noSpaces);

        // Update input styling for both signup and reset password pages
        const passwordInput = document.getElementById('password') || document.getElementById('newPassword');
        if (passwordInput) {
            passwordInput.classList.remove('valid', 'invalid');
            if (password.length > 0) {
                passwordInput.classList.add(validation.isValid ? 'valid' : 'invalid');
            }
        }

        return validation.isValid;
    }

    updateRequirement(id, isValid) {
        const element = document.getElementById(id);
        if (element) {
            element.classList.remove('valid', 'invalid');
            element.classList.add(isValid ? 'valid' : 'invalid');
        }
    }

    checkPasswordMatch(password, confirmPassword) {
        const matchElement = document.getElementById('passwordMatch');
        const confirmInput = document.getElementById('confirmPassword');
        
        if (confirmPassword.length === 0) {
            matchElement.textContent = '';
            confirmInput.classList.remove('valid', 'invalid');
            return false;
        }

        const matches = password === confirmPassword;
        matchElement.textContent = matches ? '✓ Passwords match' : '✗ Passwords do not match';
        matchElement.className = 'password-match ' + (matches ? 'valid' : 'invalid');
        
        confirmInput.classList.remove('valid', 'invalid');
        confirmInput.classList.add(matches ? 'valid' : 'invalid');
        
        return matches;
    }

    setupPasswordValidation() {
        const newPasswordInput = document.getElementById('newPassword');
        const confirmPasswordInput = document.getElementById('confirmPassword');

        if (newPasswordInput) {
            newPasswordInput.addEventListener('input', (e) => {
                const password = e.target.value;
                this.updatePasswordRequirements(password);
                
                // Also check confirm password if it has a value
                const confirmPassword = confirmPasswordInput?.value || '';
                if (confirmPassword) {
                    this.checkPasswordMatch(password, confirmPassword);
                }
            });
        }

        if (confirmPasswordInput) {
            confirmPasswordInput.addEventListener('input', (e) => {
                const confirmPassword = e.target.value;
                const password = newPasswordInput?.value || '';
                this.checkPasswordMatch(password, confirmPassword);
            });
        }
    }

    setupSignupValidation() {
        const passwordInput = document.getElementById('password');
        const confirmPasswordInput = document.getElementById('confirmPassword');

        if (passwordInput) {
            passwordInput.addEventListener('input', (e) => {
                const password = e.target.value;
                this.updatePasswordRequirements(password);
                
                // Also check confirm password if it has a value
                const confirmPassword = confirmPasswordInput?.value || '';
                if (confirmPassword) {
                    this.checkPasswordMatch(password, confirmPassword);
                }
            });
        }

        if (confirmPasswordInput) {
            confirmPasswordInput.addEventListener('input', (e) => {
                const confirmPassword = e.target.value;
                const password = passwordInput?.value || '';
                this.checkPasswordMatch(password, confirmPassword);
            });
        }
    }

    async handleResetPasswordSubmit(e) {
        e.preventDefault();
        
        const newPassword = document.getElementById('newPassword').value;
        const confirmPassword = document.getElementById('confirmPassword').value;
        const resetBtn = document.getElementById('resetBtn');

        // Enhanced password validation with detailed feedback
        const validation = this.validatePasswordWithDetails(newPassword);
        if (!validation.isValid) {
            this.showPasswordRequirementsPopup(validation.failedRequirements);
            return;
        }

        // Check password match
        if (newPassword !== confirmPassword) {
            this.showMessage('Passwords do not match. Please make sure both password fields are identical.', 'error');
            return;
        }

        // Get token from URL
        const urlParams = new URLSearchParams(window.location.search);
        const token = urlParams.get('token');

        if (!token) {
            this.showMessage('Invalid reset token. Please request a new password reset link.', 'error');
            return;
        }

        // Show loading state
        resetBtn.disabled = true;
        resetBtn.textContent = 'Resetting Password...';

        try {
            const response = await fetch('/api/reset-password', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    token: token,
                    new_password: newPassword
                })
            });

            const data = await response.json();

            if (response.ok) {
                this.showMessage('🎉 Password has been reset successfully! Redirecting to login page...', 'success');
                setTimeout(() => {
                    window.location.href = '/login.html';
                }, 2000);
            } else {
                // Handle specific error messages from backend
                let errorMessage = data.detail || 'Failed to reset password';
                
                // Make error messages more user-friendly
                if (errorMessage.includes('Password validation failed')) {
                    // Parse server-side validation errors and show popup
                    const serverErrors = errorMessage.split(': ')[1] || errorMessage;
                    const errorList = serverErrors.split(';').map(error => error.trim()).filter(error => error);
                    this.showPasswordRequirementsPopup(errorList);
                    return;
                } else if (errorMessage.includes('same as your current password')) {
                    errorMessage = '🔄 Please choose a different password. Your new password cannot be the same as your current password.';
                } else if (errorMessage.includes('Invalid or expired')) {
                    errorMessage = '⏰ This password reset link has expired or is invalid. Please request a new password reset from the login page.';
                }
                
                this.showMessage(errorMessage, 'error');
            }
        } catch (error) {
            console.error('Reset password error:', error);
            this.showMessage('🌐 Network error. Please check your internet connection and try again.', 'error');
        } finally {
            // Re-enable button
            resetBtn.disabled = false;
            resetBtn.textContent = 'Reset Password';
        }
    }

    static checkAuth() {
        const token = localStorage.getItem('authToken');
        const currentPath = window.location.pathname;
        
        // If on root path, redirect to login
        if (currentPath === '/') {
            window.location.href = '/login.html';
            return;
        }
        
        // Define auth pages
        const authPages = ['/login.html', '/signup.html', '/forgot-password.html', '/reset-password.html'];
        const isAuthPage = authPages.some(page => currentPath.includes(page));
        
        // Special handling for reset password page - allow access regardless of token status
        if (currentPath.includes('/reset-password.html')) {
            console.log('Reset password page - allowing access');
            return; // Allow access to reset password page
        }
        
        // If no token and trying to access protected pages
        if (!token && !isAuthPage) {
            Auth.prototype.redirectToLogin();
            return;
        }
        
        // If has token and on other auth pages (login, signup, forgot-password), redirect to main app
        if (token && (currentPath.includes('/login.html') || currentPath.includes('/signup.html') || currentPath.includes('/forgot-password.html'))) {
            window.location.href = '/index.html';
            return;
        }

        // If has token and on protected pages, check its validity
        if (token && !isAuthPage) {
            try {
                const tokenData = JSON.parse(atob(token.split('.')[1]));
                if (tokenData.exp * 1000 < Date.now()) {
                    // Token is expired
                    Auth.prototype.redirectToLogin();
                }
            } catch (e) {
                // If there's any error parsing the token, redirect to login
                Auth.prototype.redirectToLogin();
            }
        }
    }

    static logout() {
        Auth.prototype.redirectToLogin();
    }
}

// Initialize auth
document.addEventListener('DOMContentLoaded', () => {
    // Check authentication status
    Auth.checkAuth();
    
    // Initialize Auth class for all auth-related pages
    if (document.querySelector('form#loginForm') || 
        document.querySelector('form#signupForm') ||
        document.querySelector('form#forgotPasswordForm') ||
        document.querySelector('form#resetPasswordForm')) {
        new Auth();
    }
}); 