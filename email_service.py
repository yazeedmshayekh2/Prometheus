import aiosmtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
import logging
from email.mime.image import MIMEImage

logger = logging.getLogger(__name__)

class EmailService:
    def __init__(self):
        # Email configuration - you can set these via environment variables
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.smtp_username = os.getenv('SMTP_USERNAME')
        self.smtp_password = os.getenv('SMTP_PASSWORD')
        self.from_email = os.getenv('FROM_EMAIL', self.smtp_username)
        self.from_name = os.getenv('FROM_NAME', 'DIG Medical Assistant')
        
        # For development/testing, we can use a mock email service
        self.use_mock = os.getenv('USE_MOCK_EMAIL', 'True').lower() == 'true'
        
        print("📧 EmailService initialized:")
        print(f"   use_mock: {self.use_mock}")
        print(f"   smtp_server: {self.smtp_server}")
        print(f"   smtp_port: {self.smtp_port}")
        print(f"   smtp_username: {self.smtp_username}")
        print(f"   from_email: {self.from_email}")
        print(f"   from_name: {self.from_name}")

    async def send_password_reset_email(self, to_email: str, user_name: str, reset_token: str, base_url: str) -> bool:
        """Send password reset email"""
        print(f"🔄 send_password_reset_email called for {to_email}")
        print(f"   Mock mode: {self.use_mock}")
        
        try:
            if self.use_mock:
                print("   → Using mock email service")
                return await self._send_mock_email(to_email, user_name, reset_token, base_url)
            else:
                print("   → Using real email service")
                return await self._send_real_email(to_email, user_name, reset_token, base_url)
        except Exception as e:
            print(f"❌ Email sending failed with exception: {e}")
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
        Subject: Reset Your Password - DIG Medical Assistant
        
        Hello {user_name},
        
        You requested a password reset for your DIG Medical Assistant account.
        
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
        print(f"📧 _send_real_email called with:")
        print(f"   to_email: {to_email}")
        print(f"   user_name: {user_name}")
        print(f"   base_url: {base_url}")
        print(f"   SMTP config: {self.smtp_server}:{self.smtp_port}")
        print(f"   Username: {self.smtp_username}")
        print(f"   Password set: {bool(self.smtp_password)}")
        
        if not all([self.smtp_username, self.smtp_password]):
            print("❌ SMTP credentials not configured")
            logger.error("SMTP credentials not configured")
            return False

        reset_link = f"{base_url}/reset-password.html?token={reset_token}"
        print(f"   Reset link: {reset_link}")
        
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = 'Reset Your Password - DIG Medical Assistant'
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
                body {{ 
                    font-family: 'Poppins', Arial, sans-serif; 
                    line-height: 1.6; 
                    color: #333; 
                    margin: 0; 
                    padding: 0;
                    background-color: #f8f9fa;
                }}
                .container {{ 
                    max-width: 600px; 
                    margin: 0 auto; 
                    padding: 20px; 
                }}
                .header {{ 
                    background: linear-gradient(135deg, #b99f70 0%, #8b7a5a 100%); 
                    color: white; 
                    padding: 30px; 
                    text-align: center; 
                    border-radius: 15px 15px 0 0;
                    position: relative;
                }}
                .logo {{
                    margin-bottom: 15px;
                    border-radius: 50%;
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    padding: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.2);
                }}
                .logo img {{
                    width: 40px;
                    height: 40px;
                    display: block;
                }}
                .company-name {{
                    font-size: 24px;
                    font-weight: 700;
                    margin-bottom: 5px;
                    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
                }}
                .tagline {{
                    font-size: 14px;
                    opacity: 0.9;
                    font-weight: 300;
                }}
                .content {{ 
                    background: white; 
                    padding: 40px; 
                    border-radius: 0 0 15px 15px;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                }}
                .greeting {{
                    font-size: 18px;
                    color: #1b1b1b;
                    font-weight: 600;
                    margin-bottom: 20px;
                }}
                .button {{ 
                    display: inline-block; 
                    background: linear-gradient(135deg, #b99f70 0%, #a8926a 100%);
                    color: white !important; 
                    padding: 15px 30px; 
                    text-decoration: none; 
                    border-radius: 25px; 
                    margin: 25px 0;
                    font-weight: 600;
                    font-size: 16px;
                    box-shadow: 0 4px 15px rgba(185, 159, 112, 0.3);
                    transition: all 0.3s ease;
                }}
                .button:hover {{ 
                    background: linear-gradient(135deg, #a8926a 0%, #8b7a5a 100%);
                    transform: translateY(-2px);
                    box-shadow: 0 6px 20px rgba(185, 159, 112, 0.4);
                }}
                .link-box {{
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                    border-left: 4px solid #b99f70;
                    margin: 20px 0;
                    word-break: break-all;
                    font-family: monospace;
                    font-size: 12px;
                    color: #555;
                }}
                .important-note {{
                    background: rgba(185, 159, 112, 0.1);
                    border: 1px solid #b99f70;
                    border-radius: 8px;
                    padding: 15px;
                    margin: 20px 0;
                }}
                .footer {{ 
                    margin-top: 30px; 
                    padding-top: 25px; 
                    border-top: 2px solid #e9e8e8; 
                    font-size: 12px; 
                    color: #6c757d;
                    text-align: center;
                }}
                .footer-logo {{
                    width: 24px;
                    height: 24px;
                    display: inline-block;
                    vertical-align: middle;
                    margin-right: 8px;
                    background: #1b1b1b;
                    border-radius: 4px;
                    padding: 4px;
                }}
                .security-notice {{
                    background: #e9ecef;
                    border-radius: 5px;
                    padding: 10px;
                    font-size: 11px;
                    color: #6c757d;
                    margin-top: 15px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div class="logo">
                        <img src="cid:dig-sm-logo-w.png" alt="Doha Insurance Group Logo">
                    </div>
                    <div class="company-name">Doha Insurance Group</div>
                    <div class="tagline">💉 Medical Assistant - DIG</div>
                </div>
                <div class="content">
                    <div class="greeting">Hello {user_name},</div>
                    
                    <p>You requested a password reset for your <strong>DIG Medical Assistant</strong> account.</p>
                    
                    <p>Click the button below to securely reset your password:</p>
                    
                    <div style="text-align: center;">
                        <a href="{reset_link}" class="button">🔐 Reset My Password</a>
                    </div>
                    
                    <p style="color: #6c757d; font-size: 14px;">If the button above doesn't work, copy and paste this link into your browser:</p>
                    <div class="link-box">{reset_link}</div>
                    
                    <div class="important-note">
                        <strong>⏰ Important:</strong> This secure link will expire in <strong>7 days</strong> for your security. If you need a new reset link after this time, please request another password reset.
                    </div>
                    
                    <p style="color: #6c757d;">If you didn't request this password reset, you can safely ignore this email. Your password will remain unchanged and your account will stay secure.</p>
                    
                    <div class="footer">
                        <div>
                            <img src="cid:dig-sm-logo-w.png" alt="DIG" style="width: 20px; height: 20px; vertical-align: middle; margin-right: 8px; background: #1b1b1b; border-radius: 4px; padding: 2px;">
                            <strong>Doha Insurance Group</strong> - DIG Medical Assistant
                        </div>
                        <div style="margin-top: 10px;">This is an automated security message. Please do not reply to this email.</div>
                        <div class="security-notice">
                            🛡️ <strong>Security Notice:</strong> DIG Medical Assistant will never ask for your password via email. 
                            Only use the reset link above to change your password.
                        </div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """

        # Create text content (fallback)
        text_content = f"""
        ============================================
        DOHA INSURANCE GROUP (DIG)
        💉 Medical Assistant - DIG Medical Assistant
        ============================================

        Hello {user_name},

        You requested a password reset for your DIG Medical Assistant account.

        Please click the following link to reset your password:
        {reset_link}

        ⏰ IMPORTANT: This secure link will expire in 7 days for your security.

        If you didn't request this password reset, you can safely ignore this email. 
        Your password will remain unchanged and your account will stay secure.

        --------------------------------------------
        Best regards,
        Doha Insurance Group (DIG)

        🛡️ Security Notice: DIG Medical Assistant will never ask for 
        your password via email. Only use the reset link above to change your password.

        This is an automated security message. Please do not reply to this email.
        ============================================
        """

        # Attach parts
        text_part = MIMEText(text_content, 'plain')
        html_part = MIMEText(html_content, 'html')
        
        msg.attach(text_part)
        msg.attach(html_part)
        
        # Embed the DIG logo image
        try:
            # Path to the logo file
            logo_path = os.path.join(os.path.dirname(__file__), 'static', 'Content', 'img', 'dig-sm-logo-w.png')
            
            # If the file exists, embed it
            if os.path.exists(logo_path):
                with open(logo_path, 'rb') as f:
                    logo_data = f.read()
                
                logo_image = MIMEImage(logo_data)
                logo_image.add_header('Content-ID', '<dig-sm-logo-w.png>')
                logo_image.add_header('Content-Disposition', 'inline', filename='dig-sm-logo-w.png')
                msg.attach(logo_image)
                print(f"✅ Logo embedded from: {logo_path}")
            else:
                print(f"⚠️ Logo file not found at: {logo_path}")
                
        except Exception as e:
            print(f"⚠️ Could not embed logo: {e}")

        # Send email
        try:
            print("🔄 Attempting to send email via SMTP...")
            await aiosmtplib.send(
                msg,
                hostname=self.smtp_server,
                port=self.smtp_port,
                start_tls=True,
                username=self.smtp_username,
                password=self.smtp_password,
            )
            print(f"✅ Password reset email sent successfully to {to_email}")
            logger.info(f"Password reset email sent to {to_email}")
            return True
        except Exception as e:
            print(f"❌ Failed to send email: {e}")
            logger.error(f"Failed to send email via SMTP: {e}")
            return False

# Global instance
email_service = EmailService() 