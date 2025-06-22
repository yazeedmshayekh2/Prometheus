#!/usr/bin/env python3
"""
PDF Caching Script for Prometheus Insurance System
Pre-processes and caches all PDF documents from the database for faster runtime performance.
Run this script before starting the main application to ensure all PDFs are indexed.
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Set
from db_utils import DatabaseConnection
from main import DocumentQASystem
import traceback

class PDFCacher:
    """
    Handles pre-processing and caching of all PDF documents in the system.
    """
    
    def __init__(self):
        """Initialize the PDF cacher with database connection"""
        print("🚀 Initializing PDF Caching System...")
        
        # Initialize database connection
        self.db = DatabaseConnection()
        db_connection_string = DatabaseConnection.get_connection_string(DatabaseConnection)
        
        if not db_connection_string:
            raise Exception("Could not get database connection string")
        
        # Initialize QA system for document processing
        self.qa_system = DocumentQASystem(db_connection_string)
        print("✅ Database and QA system initialized")
        
        # Statistics tracking
        self.stats = {
            'total_pdfs': 0,
            'processed': 0,
            'skipped': 0,
            'failed': 0,
            'start_time': None,
            'end_time': None
        }
    
    def get_all_pdfs_from_database(self) -> List[Tuple[str, str]]:
        """
        Get all PDF links and company names from the database
        Returns: List of (pdf_link, company_name) tuples
        """
        print("📋 Scanning database for PDF documents...")
        
        try:
            # Use the existing get_all_policies method from DatabaseConnection
            policies = self.db.get_all_policies()
            
            # Convert to the format we need
            pdfs = []
            print(len(policies))
            for policy in policies:
                pdf_link = policy.get('pdf_link', '').strip()
                company_name = policy.get('company_name', 'Unknown Company').strip()
                
                if pdf_link and pdf_link.lower().endswith('.pdf'):
                    pdfs.append((pdf_link, company_name))
            
            # Remove duplicates while preserving order
            seen = set()
            unique_pdfs = []
            for pdf_link, company_name in pdfs:
                if pdf_link not in seen:
                    seen.add(pdf_link)
                    unique_pdfs.append((pdf_link, company_name))
            
            print(f"📊 Found {len(unique_pdfs)} unique PDF documents in database")
            return unique_pdfs
            
        except Exception as e:
            print(f"❌ Error scanning database: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def is_pdf_already_cached(self, pdf_link: str) -> bool:
        """
        Check if a PDF is already processed and cached
        """
        try:
            # Check if document exists in vector store
            return self.qa_system.is_document_indexed(pdf_link)
        except Exception:
            return False
    
    def cache_pdf_document(self, pdf_link: str, company_name: str) -> bool:
        """
        Cache a single PDF document
        Returns: True if successful, False otherwise
        """
        try:
            print(f"📄 Processing: {company_name} - {pdf_link}")
            
            # Download PDF content
            pdf_content = self.db.download_policy_pdf(pdf_link)
            if not pdf_content:
                print(f"❌ Failed to download PDF: {pdf_link}")
                return False
            
            # Index the document using the QA system's method
            success = self.qa_system._index_new_policy_document(pdf_link, pdf_content, company_name)
            
            if success:
                print(f"✅ Successfully cached: {company_name}")
                return True
            else:
                print(f"❌ Failed to index PDF: {pdf_link}")
                return False
                
        except Exception as e:
            print(f"❌ Error processing {pdf_link}: {str(e)}")
            return False
    
    def cache_all_pdfs(self, force_reindex: bool = False):
        """
        Cache all PDF documents from the database
        
        Args:
            force_reindex: If True, re-process even already cached PDFs
        """
        print("🔄 Starting PDF caching process...")
        self.stats['start_time'] = time.time()
        
        # Get all PDFs from database
        all_pdfs = self.get_all_pdfs_from_database()
        self.stats['total_pdfs'] = len(all_pdfs)
        
        if not all_pdfs:
            print("⚠️  No PDFs found in database")
            return
        
        print(f"📦 Processing {len(all_pdfs)} PDF documents...")
        print("=" * 60)
        
        # Process each PDF
        for i, (pdf_link, company_name) in enumerate(all_pdfs, 1):
            try:
                print(f"\n[{i}/{len(all_pdfs)}] Processing: {company_name}")
                
                # Check if already cached (unless forcing reindex)
                if not force_reindex and self.is_pdf_already_cached(pdf_link):
                    print(f"⏭️  Already cached, skipping...")
                    self.stats['skipped'] += 1
                    continue
                
                # Cache the PDF
                if self.cache_pdf_document(pdf_link, company_name):
                    self.stats['processed'] += 1
                else:
                    self.stats['failed'] += 1
                
                # Show progress
                processed_count = self.stats['processed'] + self.stats['skipped']
                progress = (processed_count / len(all_pdfs)) * 100
                print(f"📊 Progress: {progress:.1f}% ({processed_count}/{len(all_pdfs)})")
                
            except KeyboardInterrupt:
                print("\n🛑 Process interrupted by user")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                traceback.print_exc()
                self.stats['failed'] += 1
                continue
        
        self.stats['end_time'] = time.time()
        self.print_final_report()
    
    def print_final_report(self):
        """Print final statistics report"""
        print("\n" + "=" * 60)
        print("📊 PDF CACHING COMPLETED")
        print("=" * 60)
        
        duration = self.stats['end_time'] - self.stats['start_time']
        
        print(f"⏱️  Total time: {duration:.2f} seconds")
        print(f"📄 Total PDFs found: {self.stats['total_pdfs']}")
        print(f"✅ Successfully processed: {self.stats['processed']}")
        print(f"⏭️  Skipped (already cached): {self.stats['skipped']}")
        print(f"❌ Failed: {self.stats['failed']}")
        
        if self.stats['total_pdfs'] > 0:
            success_rate = ((self.stats['processed'] + self.stats['skipped']) / self.stats['total_pdfs']) * 100
            print(f"📈 Success rate: {success_rate:.1f}%")
        
        if duration > 0:
            rate = self.stats['processed'] / duration
            print(f"⚡ Processing rate: {rate:.2f} PDFs/second")
        
        print("\n🎉 PDF caching process completed!")
        
        if self.stats['failed'] > 0:
            print(f"⚠️  {self.stats['failed']} documents failed to process. Check the logs above for details.")
    
    def verify_cache_integrity(self):
        """Verify that cached documents are accessible"""
        print("\n🔍 Verifying cache integrity...")
        
        all_pdfs = self.get_all_pdfs_from_database()
        cached_count = 0
        
        for pdf_link, company_name in all_pdfs:
            if self.is_pdf_already_cached(pdf_link):
                cached_count += 1
        
        print(f"✅ {cached_count}/{len(all_pdfs)} documents are properly cached")
        
        if cached_count == len(all_pdfs):
            print("🎯 All documents are successfully cached!")
        else:
            missing = len(all_pdfs) - cached_count
            print(f"⚠️  {missing} documents are missing from cache")


def main():
    """Main function to run the PDF caching process"""
    try:
        print("🏥 Prometheus Insurance System - PDF Cacher")
        print("=" * 50)
        
        # Parse command line arguments
        force_reindex = '--force' in sys.argv or '-f' in sys.argv
        verify_only = '--verify' in sys.argv or '-v' in sys.argv
        
        if '--help' in sys.argv or '-h' in sys.argv:
            print("""
Usage: python cache_pdfs.py [options]

Options:
    --force, -f     Force re-indexing of all PDFs (even if already cached)
    --verify, -v    Only verify cache integrity without processing
    --help, -h      Show this help message

Examples:
    python cache_pdfs.py                # Cache new PDFs only
    python cache_pdfs.py --force        # Re-cache all PDFs
    python cache_pdfs.py --verify       # Verify cache integrity
            """)
            return
        
        # Initialize cacher
        cacher = PDFCacher()
        
        if verify_only:
            cacher.verify_cache_integrity()
        else:
            cacher.cache_all_pdfs(force_reindex=force_reindex)
            
            # Optional verification after caching
            if input("\n🔍 Verify cache integrity? (y/N): ").lower() == 'y':
                cacher.verify_cache_integrity()
        
        print("\n✨ Process completed successfully!")
        
    except KeyboardInterrupt:
        print("\n🛑 Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 