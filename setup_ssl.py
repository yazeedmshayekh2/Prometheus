#!/usr/bin/env python3
"""
SSL setup helper script for the Prometheus application.
This script helps users generate and configure SSL certificates for secure HTTPS connections.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

def generate_self_signed_cert(cert_file="ssl/cert.pem", key_file="ssl/key.pem", common_name="localhost"):
    """Generate a self-signed certificate for development use"""
    ssl_dir = Path("ssl")
    ssl_dir.mkdir(exist_ok=True)
    
    cert_path = Path(cert_file)
    key_path = Path(key_file)
    
    # Check if certificate already exists
    if cert_path.exists() and key_path.exists():
        print(f"SSL certificate already exists at {cert_path} and {key_path}")
        replace = input("Do you want to replace it? (y/n): ").lower().strip()
        if replace != 'y':
            print("Using existing certificate.")
            return cert_file, key_file
    
    print(f"Generating self-signed SSL certificate for '{common_name}'...")
    
    try:
        # Use OpenSSL to generate a self-signed certificate
        subprocess.run([
            "openssl", "req", "-x509", "-newkey", "rsa:4096", "-nodes",
            "-out", cert_file, "-keyout", key_file,
            "-days", "365", "-subj", f"/CN={common_name}"
        ], check=True)
        
        print(f"\n✅ Self-signed certificate generated successfully!")
        print(f"Certificate file: {cert_file}")
        print(f"Key file: {key_file}")
        
        # Make key file read-only for security
        os.chmod(key_file, 0o600)
        print(f"Set key file permissions to read-only for owner")
        
        print("\nTo use this certificate with the application, run:")
        print(f"python app.py --ssl --cert {cert_file} --key {key_file}")
        
        return cert_file, key_file
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error generating self-signed certificate: {e}")
        print("Please install OpenSSL or provide your own certificate.")
        return None, None
    except FileNotFoundError:
        print("\n❌ OpenSSL not found. Please install OpenSSL or provide your own certificate.")
        return None, None

def check_certificate(cert_file):
    """Show information about a certificate"""
    try:
        print(f"Certificate information for: {cert_file}")
        print("-" * 60)
        subprocess.run(["openssl", "x509", "-in", cert_file, "-text", "-noout"], check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error checking certificate: {e}")
        return False

def main():
    """Main function to set up SSL certificates"""
    parser = argparse.ArgumentParser(description="Set up SSL certificates for Prometheus application")
    parser.add_argument("--cert", type=str, default="ssl/cert.pem", help="Path to SSL certificate")
    parser.add_argument("--key", type=str, default="ssl/key.pem", help="Path to SSL key")
    parser.add_argument("--cn", type=str, default="localhost", help="Common Name for the certificate")
    parser.add_argument("--check", action="store_true", help="Check existing certificate")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Prometheus - SSL Certificate Setup Utility")
    print("=" * 60)
    
    if args.check:
        if Path(args.cert).exists():
            check_certificate(args.cert)
        else:
            print(f"Certificate file does not exist: {args.cert}")
            return False
    else:
        return generate_self_signed_cert(args.cert, args.key, args.cn)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 