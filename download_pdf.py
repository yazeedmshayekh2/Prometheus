import urllib3
import requests
import pyodbc
from pathlib import Path
from urllib.parse import unquote
from concurrent.futures import ThreadPoolExecutor

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Database configuration
DB_USER = 'aiuser'
DB_PASS = 'AIP@ss0rdSQL'
DB_HOST = '172.16.15.161'
DB_NAME = 'InsuranceOnlinePortal'

# Create docs directory if it doesn't exist
DOCS_DIR = Path("docs")
DOCS_DIR.mkdir(exist_ok=True)

def get_pdf_links():
    """Get PDF links from database using direct pyodbc connection"""
    try:
        # Create connection string
        conn_str = (
            f"DRIVER={{ODBC Driver 18 for SQL Server}};"
            f"SERVER={DB_HOST};"
            f"DATABASE={DB_NAME};"
            f"UID={DB_USER};"
            f"PWD={DB_PASS};"
            "TrustServerCertificate=yes;"
            "Encrypt=yes;"
            "Timeout=30;"
        )
        
        print(f"Connecting to database...")
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        # First, let's check total number of contracts
        count_query = """
            SELECT COUNT(*) as total_contracts,
                   SUM(CASE WHEN PDFLink IS NOT NULL AND LEN(PDFLink) > 0 THEN 1 ELSE 0 END) as contracts_with_pdf,
                   SUM(CASE WHEN isDeleted = 0 THEN 1 ELSE 0 END) as active_contracts
            FROM tblHContracts
        """
        cursor.execute(count_query)
        counts = cursor.fetchone()
        print(f"\nDatabase statistics:")
        print(f"Total contracts: {counts[0]}")
        print(f"Contracts with PDFs: {counts[1]}")
        print(f"Active contracts: {counts[2]}")
        
        # Now get all PDFs
        query = """
            SELECT DISTINCT
                c.PDFLink,
                c.ContractID,
                c.StartDate,
                c.EndDate,
                c.CreationDate,
                c.isDeleted
            FROM
                tblHContracts c
            WHERE
                c.PDFLink IS NOT NULL
                AND LEN(c.PDFLink) > 0
            ORDER BY
                c.CreationDate DESC
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        # Print summary of found PDFs
        print(f"\nFound {len(results)} PDF files in database")
        print("\nSample of first 5 contracts:")
        for i, row in enumerate(results[:5]):
            print(f"Contract {i+1}: {row.ContractID} ({row.StartDate} to {row.EndDate}) - {'Active' if not row.isDeleted else 'Deleted'}")
        
        # Extract PDF links
        links = [row.PDFLink for row in results if row.PDFLink]
        
        cursor.close()
        conn.close()
        return links
        
    except Exception as e:
        print(f"Database error: {str(e)}")
        print("Available ODBC drivers:", pyodbc.drivers())
        return []

def download_pdf(url):
    """Download a single PDF file"""
    try:
        # Extract filename from URL
        filename = unquote(url.split('/')[-1])
        filepath = DOCS_DIR / filename
        
        # Skip if file already exists
        if filepath.exists():
            print(f"Skipping existing file: {filename}")
            return
        
        # Download file
        response = requests.get(url, verify=False, timeout=30)
        response.raise_for_status()
        
        # Check if it's actually a PDF
        if 'application/pdf' not in response.headers.get('content-type', '').lower():
            print(f"Not a PDF file: {url}")
            return
            
        # Save file
        with open(filepath, 'wb') as f:
            f.write(response.content)
            
        print(f"Downloaded: {filename}")
        
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")

def main():
    # Get PDF links from database
    print("Fetching PDF links from database...")
    pdf_links = get_pdf_links()
    
    if not pdf_links:
        print("No PDF links found")
        return
        
    print(f"Found {len(pdf_links)} PDF links")
    
    # Download PDFs in parallel
    print("Downloading PDFs...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(download_pdf, pdf_links)
    
    print("Download complete!")

if __name__ == "__main__":
    main()
