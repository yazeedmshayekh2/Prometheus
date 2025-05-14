#!/usr/bin/env python3
"""
Ngrok setup helper script for the Prometheus application.
This script helps users set up ngrok integration with their authtoken.
"""

import argparse
import sys

try:
    from pyngrok import ngrok, conf
    from pyngrok.exception import PyngrokError
except ImportError:
    print("Error: pyngrok package not installed.")
    print("Please install it using: pip install pyngrok")
    sys.exit(1)

def setup_ngrok_auth(authtoken, config_path=None):
    """Set up ngrok with the provided authentication token"""
    try:
        if config_path:
            # Use custom config path
            conf.set_default_config_path(config_path)
            print(f"Using custom ngrok config path: {config_path}")
        
        # Set the auth token
        ngrok.set_auth_token(authtoken)
        print("\n✅ Ngrok authentication token configured successfully!")
        print("You can now run the app with ngrok using: python app.py --ngrok")
        
        # Display ngrok version
        version = ngrok.get_ngrok_version()
        print(f"\nNgrok version: {version}")
        
        return True
    except PyngrokError as e:
        print(f"\n❌ Error setting up ngrok: {str(e)}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        return False

def main():
    """Main function to set up ngrok authentication"""
    parser = argparse.ArgumentParser(description="Set up ngrok authentication for Prometheus application")
    parser.add_argument("--authtoken", help="Your ngrok authentication token")
    parser.add_argument("--config", help="Path to ngrok config file (optional)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Prometheus - Ngrok Setup Utility")
    print("=" * 60)
    
    if not args.authtoken:
        print("\nTo use ngrok with this application, you need an ngrok account and authtoken.")
        print("1. Sign up for free at: https://ngrok.com/")
        print("2. Get your authtoken from: https://dashboard.ngrok.com/get-started/your-authtoken")
        
        authtoken = input("\nEnter your ngrok authtoken: ").strip()
        if not authtoken:
            print("\n❌ No authtoken provided. Setup canceled.")
            return False
    else:
        authtoken = args.authtoken
    
    return setup_ngrok_auth(authtoken, args.config)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 