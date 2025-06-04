class Auth {
    constructor() {
        this.form = document.querySelector('form#loginForm') || document.querySelector('form#signupForm');
        this.setupEventListeners();
        
        // Check token expiration periodically
        this.startTokenCheck();
    }

    setupEventListeners() {
        if (this.form) {
            this.form.addEventListener('submit', async (e) => {
                e.preventDefault();
                if (this.form.id === 'loginForm') {
                    await this.handleLogin();
                } else if (this.form.id === 'signupForm') {
                    await this.handleSignup();
                }
            });
        }
    }

    startTokenCheck() {
        // Check token every minute
        setInterval(() => {
            this.checkTokenValidity();
        }, 60000); // 60000 ms = 1 minute
    }

    checkTokenValidity() {
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

        if (password !== confirmPassword) {
            alert('Passwords do not match!');
            return;
        }

        try {
            const response = await fetch('/api/signup', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ name, email, password })
            });

            const data = await response.json();

            if (response.ok) {
                alert('Account created successfully! Please login.');
                window.location.href = '/login.html';
            } else {
                alert(data.detail || 'Signup failed. Please try again.');
            }
        } catch (error) {
            console.error('Signup error:', error);
            alert('Signup failed. Please try again.');
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
        
        // If no token and trying to access protected pages
        if (!token && currentPath !== '/login.html' && currentPath !== '/signup.html') {
            Auth.prototype.redirectToLogin();
            return;
        }
        
        // If has token and on auth pages, redirect to main app
        if (token && (currentPath === '/login.html' || currentPath === '/signup.html')) {
            window.location.href = '/index.html';
            return;
        }

        // If has token, check its validity
        if (token) {
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
    
    // Initialize Auth class if on login or signup page
    if (document.querySelector('form#loginForm') || document.querySelector('form#signupForm')) {
        new Auth();
    }
}); 