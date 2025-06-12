#!/usr/bin/env python3
"""
Test script to verify Gmail SMTP email sending functionality
"""
import asyncio
import aiosmtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

async def test_gmail_smtp():
    """Test Gmail SMTP connection and email sending"""
    
    # Get configuration from environment
    smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    smtp_port = int(os.getenv('SMTP_PORT', '587'))
    smtp_username = os.getenv('SMTP_USERNAME')
    smtp_password = os.getenv('SMTP_PASSWORD')
    from_email = os.getenv('FROM_EMAIL', smtp_username)
    from_name = os.getenv('FROM_NAME', 'Test Email')
    
    print("=== Gmail SMTP Test ===")
    print(f"SMTP Server: {smtp_server}:{smtp_port}")
    print(f"Username: {smtp_username}")
    print(f"From Email: {from_email}")
    print(f"From Name: {from_name}")
    print(f"Password Length: {len(smtp_password) if smtp_password else 0} characters")
    print()
    
    if not smtp_username or not smtp_password:
        print("❌ ERROR: SMTP credentials not found in environment variables")
        return False
    
    # Test email content
    to_email = smtp_username  # Send test email to yourself
    subject = "Test Email from Prometheus App"
    
    # Create message
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = f"{from_name} <{from_email}>"
    msg['To'] = to_email
    
    # Create HTML and text content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Email Test - DIG Prometheus</title>
        <style>
            body { 
                font-family: 'Poppins', Arial, sans-serif; 
                line-height: 1.6; 
                color: #333; 
                margin: 0; 
                padding: 0;
                background-color: #f8f9fa;
            }
            .container { 
                max-width: 600px; 
                margin: 0 auto; 
                padding: 20px; 
            }
            .header { 
                background: linear-gradient(135deg, #b99f70 0%, #8b7a5a 100%); 
                color: white; 
                padding: 30px; 
                text-align: center; 
                border-radius: 15px 15px 0 0;
                position: relative;
            }
            .logo {
                width: 50px;
                height: 50px;
                margin-bottom: 15px;
                background: white;
                border-radius: 50%;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                padding: 10px;
            }
            .company-name {
                font-size: 24px;
                font-weight: 700;
                margin-bottom: 5px;
                text-shadow: 0 2px 4px rgba(0,0,0,0.3);
            }
            .tagline {
                font-size: 14px;
                opacity: 0.9;
                font-weight: 300;
            }
            .content { 
                background: white; 
                padding: 40px; 
                border-radius: 0 0 15px 15px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }
            .test-success {
                background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                margin: 20px 0;
            }
            .feature-list {
                background: rgba(185, 159, 112, 0.1);
                border: 1px solid #b99f70;
                border-radius: 8px;
                padding: 20px;
                margin: 20px 0;
            }
            .footer { 
                margin-top: 30px; 
                padding-top: 25px; 
                border-top: 2px solid #e9e8e8; 
                font-size: 12px; 
                color: #6c757d;
                text-align: center;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="logo">
                    <span style="font-weight: bold; color: #1b1b1b; font-size: 18px;">DIG</span>
                </div>
                <div class="company-name">Doha Insurance Group</div>
                <div class="tagline">💉 Medical Group Assistant - DIG</div>
            </div>
            <div class="content">
                <div class="test-success">
                    <h2>🎉 Gmail SMTP Test Success!</h2>
                    <p>Email system is working perfectly!</p>
                </div>
                
                <p><strong>Great news!</strong> This email confirms that your Prometheus Insurance Assistant application can now send real emails through Gmail.</p>
                
                <div class="feature-list">
                    <h3>✅ What's Working:</h3>
                    <ul>
                        <li>📧 <strong>Gmail SMTP Connection:</strong> Successfully authenticated</li>
                        <li>🎨 <strong>Professional Design:</strong> Branded with DIG colors and logo</li>
                        <li>📱 <strong>Mobile Responsive:</strong> Looks great on all devices</li>
                        <li>🔐 <strong>Password Reset Emails:</strong> Ready to send securely</li>
                        <li>⏰ <strong>7-Day Token Expiry:</strong> Enhanced security timing</li>
                    </ul>
                </div>
                
                <p><strong>Next Steps:</strong> Your forgot password feature is now fully operational. Users can request password resets and will receive professional, branded emails with secure reset links.</p>
                
                <div class="footer">
                    <div>
                        <span style="background: #1b1b1b; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 10px;">DIG</span>
                        <strong>Doha Insurance Group</strong> - Prometheus Insurance Assistant
                    </div>
                    <div style="margin-top: 10px;">Test email sent successfully from your application</div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    text_content = """
    🎉 Gmail SMTP Test Success!
    
    This is a test email from your Prometheus Insurance Assistant application.
    If you're reading this, Gmail SMTP is working correctly!
    
    Configuration Details:
    - SMTP Server: smtp.gmail.com:587
    - Authentication: ✅ Successful
    - Email Delivery: ✅ Working
    
    Your forgot password emails should now work properly.
    """
    
    # Attach parts
    text_part = MIMEText(text_content, 'plain')
    html_part = MIMEText(html_content, 'html')
    
    msg.attach(text_part)
    msg.attach(html_part)
    
    try:
        print("🔄 Testing SMTP connection...")
        
        # Use aiosmtplib.send directly which handles TLS properly
        await aiosmtplib.send(
            msg,
            hostname=smtp_server,
            port=smtp_port,
            start_tls=True,
            username=smtp_username,
            password=smtp_password,
        )
        
        print("✅ Email sent successfully!")
        print("\n🎉 SUCCESS: Gmail SMTP is working correctly!")
        print(f"📧 Check your inbox at {to_email}")
        return True
        
    except aiosmtplib.SMTPAuthenticationError as e:
        print(f"❌ AUTHENTICATION ERROR: {e}")
        print("💡 Possible causes:")
        print("   - App Password is incorrect")
        print("   - 2-Step Verification not enabled")
        print("   - App Passwords not generated")
        return False
        
    except aiosmtplib.SMTPConnectError as e:
        print(f"❌ CONNECTION ERROR: {e}")
        print("💡 Possible causes:")
        print("   - Internet connection issues")
        print("   - SMTP server/port incorrect")
        print("   - Firewall blocking connection")
        return False
        
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        print(f"Error type: {type(e).__name__}")
        return False

if __name__ == "__main__":
    print("Starting Gmail SMTP test...")
    success = asyncio.run(test_gmail_smtp())
    
    if success:
        print("\n✅ Email test completed successfully!")
        print("Now you can use the forgot password feature in your app.")
    else:
        print("\n❌ Email test failed!")
        print("Please check the configuration and try again.") 