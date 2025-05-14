"""
Script to reindex all documents with correct source filenames.
This script should be run when you need to rebuild the indices with proper metadata.
"""

import sys
from main import DocumentQASystem
from db_utils import DatabaseConnection
import argparse

def get_all_policy_documents(db):
    """Get all policy documents from database"""
    try:
        policies = db.get_all_policies()
        if not policies:
            print("No policies found in database")
            return []
            
        documents = []
        for policy in policies:
            if policy.get('pdf_link'):
                documents.append({
                    'pdf_link': policy['pdf_link'],
                    'company_name': policy.get('company_name'),
                    'start_date': policy.get('start_date'),
                    'end_date': policy.get('end_date')
                })
        return documents
    except Exception as e:
        print(f"Error getting policies: {str(e)}")
        return []

def reindex_documents(qa_system):
    """Reindex all documents using the provided QA system"""
    print("Starting reindexing process...")
    
    try:
        # Get all documents from database
        documents = get_all_policy_documents(qa_system.db)
        if not documents:
            print("No documents to process")
            return
            
        print(f"Found {len(documents)} documents to process")
        
        # Process each document
        successful_docs = 0
        for doc in documents:
            pdf_link = doc['pdf_link']
            
            try:
                # Download PDF
                pdf_content = qa_system.db.download_policy_pdf(pdf_link)
                if not pdf_content:
                    print(f"Failed to download {pdf_link}")
                    continue
                
                # Create metadata
                metadata = {
                    "company": doc.get('company_name'),
                    "start_date": doc.get('start_date'),
                    "end_date": doc.get('end_date'),
                    "type": "pdf"
                }
                
                # Process document
                if qa_system.process_policy_document(pdf_link, pdf_content, metadata):
                    successful_docs += 1
                
            except Exception as e:
                print(f"Error processing {pdf_link}: {str(e)}")
                continue
        
        print(f"\nReindexing completed. Successfully processed {successful_docs} out of {len(documents)} documents")
        
    except Exception as e:
        print(f"Error during reindexing: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--reset', action='store_true', help='Reset all indices before reindexing')
    args = parser.parse_args()
    
    try:
        # Initialize QA system once
        db_connection_string = DatabaseConnection.get_connection_string(DatabaseConnection)
        if not db_connection_string:
            print("Error: Could not get database connection string")
            sys.exit(1)
            
        qa_system = DocumentQASystem(db_connection_string)
        
        # Reset indices if requested
        if args.reset:
            print("Resetting indices...")
            qa_system.reset_indices()
        
        # Perform reindexing
        reindex_documents(qa_system)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)