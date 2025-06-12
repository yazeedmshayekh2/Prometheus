# Gmail SMTP Configuration for Real Email Sending

## Quick Setup Instructions

To enable real Gmail email sending instead of mock mode, you need to:

### 1. Generate Gmail App Password

**Important**: You cannot use your regular Gmail password for SMTP. You need an "App Password":

1. Go to your Google Account settings: https://myaccount.google.com/
2. Click on "Security" in the left sidebar
3. Under "Signing in to Google", make sure "2-Step Verification" is enabled
4. Scroll down and click on "App passwords"
5. Select "Mail" as the app and "Other" as the device
6. Enter "Prometheus App" as the device name
7. Click "Generate" - Google will give you a 16-character password like: `abcd efgh ijkl mnop`
8. **Copy this password and remove all spaces**: `abcdefghijklmnop`

### 2. Set Environment Variables

You can set these environment variables either:

**Option A: Export in terminal (temporary)**
```bash
export USE_MOCK_EMAIL=False
export SMTP_USERNAME=your-email@gmail.com
export SMTP_PASSWORD=your-16-char-app-password
export FROM_EMAIL=your-email@gmail.com
export FROM_NAME="Prometheus Insurance Assistant"
```

**Option B: Create .env file (recommended)**
Create a file called `.env` in your project root with:
```
USE_MOCK_EMAIL=False
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-16-char-app-password
FROM_EMAIL=your-email@gmail.com
FROM_NAME=Prometheus Insurance Assistant
```

### 3. Install python-dotenv (if using .env file)

If you choose Option B, install python-dotenv:
```bash
pip install python-dotenv
```

Then modify your app.py to load the .env file at the top:
```python
from dotenv import load_dotenv
load_dotenv()  # Add this line early in app.py
```

### 4. Test Configuration

Start your app and try the forgot password feature - emails should now be sent to real Gmail addresses!

## Troubleshooting

- **"Authentication failed"**: Double-check your App Password (no spaces, 16 characters)
- **"2-Step Verification required"**: Enable 2FA on your Google account first
- **"App passwords not available"**: Make sure 2-Step Verification is enabled
- **Still getting mock emails**: Check that `USE_MOCK_EMAIL=False` is set correctly

## Security Notes

- Never commit your actual credentials to version control
- The App Password is specific to this application
- You can revoke App Passwords anytime from Google Account settings
- Gmail SMTP is free for reasonable usage (up to 500 emails per day) 