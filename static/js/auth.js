class Auth {
    constructor() {
        this.form = document.querySelector('form');
        this.setupEventListeners();
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
                
                // Redirect to main page
                window.location.href = '/';
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
        if (!token && !window.location.pathname.includes('login.html') && !window.location.pathname.includes('signup.html')) {
            window.location.href = '/login.html';
        }
    }

    static logout() {
        localStorage.removeItem('authToken');
        localStorage.removeItem('userName');
        localStorage.removeItem('userEmail');
        window.location.href = '/login.html';
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