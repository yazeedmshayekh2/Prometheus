import aiosmtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class EmailService:
    def __init__(self):
        # Email configuration - you can set these via environment variables
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.smtp_username = os.getenv('SMTP_USERNAME')
        self.smtp_password = os.getenv('SMTP_PASSWORD')
        self.from_email = os.getenv('FROM_EMAIL', self.smtp_username)
        self.from_name = os.getenv('FROM_NAME', 'Prometheus Insurance Assistant')
        
        # For development/testing, we can use a mock email service
        self.use_mock = os.getenv('USE_MOCK_EMAIL', 'True').lower() == 'true'

    async def send_password_reset_email(self, to_email: str, user_name: str, reset_token: str, base_url: str) -> bool:
        """Send password reset email"""
        try:
            if self.use_mock:
                return await self._send_mock_email(to_email, user_name, reset_token, base_url)
            else:
                return await self._send_real_email(to_email, user_name, reset_token, base_url)
        except Exception as e:
            logger.error(f"Failed to send password reset email: {e}")
            return False

    async def _send_mock_email(self, to_email: str, user_name: str, reset_token: str, base_url: str) -> bool:
        """Mock email service for development"""
        reset_link = f"{base_url}/reset-password.html?token={reset_token}"
        
        # Log the email content instead of sending
        email_content = f"""
        Password Reset Email (MOCK MODE)
        ================================
        To: {to_email}
        Subject: Reset Your Password - Prometheus Insurance Assistant
        
        Hello {user_name},
        
        You requested a password reset for your Prometheus Insurance Assistant account.
        
        Please click the following link to reset your password:
        {reset_link}
        
        This link will expire in 7 days for security reasons.
        
        If you didn't request this reset, please ignore this email.
        
        Best regards,
        Prometheus Insurance Team
        """
        
        print(email_content)
        logger.info(f"Mock email sent to {to_email} with reset token: {reset_token}")
        return True

    async def _send_real_email(self, to_email: str, user_name: str, reset_token: str, base_url: str) -> bool:
        """Send real email using SMTP"""
        if not all([self.smtp_username, self.smtp_password]):
            logger.error("SMTP credentials not configured")
            return False

        reset_link = f"{base_url}/reset-password.html?token={reset_token}"
        
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = 'Reset Your Password - Prometheus Insurance Assistant'
        msg['From'] = f"{self.from_name} <{self.from_email}>"
        msg['To'] = to_email

        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Password Reset</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                .button {{ display: inline-block; background: #007bff; color: white; padding: 12px 25px; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
                .button:hover {{ background: #0056b3; }}
                .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Password Reset Request</h1>
                    <p>Prometheus Insurance Assistant</p>
                </div>
                <div class="content">
                    <p>Hello {user_name},</p>
                    
                    <p>You requested a password reset for your Prometheus Insurance Assistant account.</p>
                    
                    <p>Please click the button below to reset your password:</p>
                    
                    <div style="text-align: center;">
                        <a href="{reset_link}" class="button">Reset My Password</a>
                    </div>
                    
                    <p>Or copy and paste this link into your browser:</p>
                    <p style="word-break: break-all; background: #e9ecef; padding: 10px; border-radius: 4px;">{reset_link}</p>
                    
                    <p><strong>Important:</strong> This link will expire in 7 days for security reasons.</p>
                    
                    <p>If you didn't request this password reset, please ignore this email. Your password will remain unchanged.</p>
                    
                    <div class="footer">
                        <p>Best regards,<br>
                        The Prometheus Insurance Team</p>
                        <p>This is an automated message. Please do not reply to this email.</p>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """

        # Create text content (fallback)
        text_content = f"""
        Hello {user_name},

        You requested a password reset for your Prometheus Insurance Assistant account.

        Please click the following link to reset your password:
        {reset_link}

        This link will expire in 7 days for security reasons.

        If you didn't request this reset, please ignore this email.

        Best regards,
        The Prometheus Insurance Team
        """

        # Attach parts
        text_part = MIMEText(text_content, 'plain')
        html_part = MIMEText(html_content, 'html')
        
        msg.attach(text_part)
        msg.attach(html_part)

        # Send email
        try:
            await aiosmtplib.send(
                msg,
                hostname=self.smtp_server,
                port=self.smtp_port,
                start_tls=True,
                username=self.smtp_username,
                password=self.smtp_password,
            )
            logger.info(f"Password reset email sent to {to_email}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email via SMTP: {e}")
            return False

# Global instance
email_service = EmailService() 