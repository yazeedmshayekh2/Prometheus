"""
Fast reindexing script that skips LLM context generation for speed.
This script rebuilds the vector indices with correct dimensions.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List

# Add the current directory to the path to import main
sys.path.append(str(Path(__file__).parent))

from main import DocumentQASystem, DocumentChunk
from db_utils import DatabaseConnection
import argparse

def create_simple_chunks(text: str, metadata: Dict[str, Any], max_chunk_size: int = 1000) -> List[DocumentChunk]:
    """Create simple chunks without LLM context generation for speed"""
    # Simple text splitting
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    chunk_index = 0
    
    for para in paragraphs:
        if len(current_chunk) + len(para) > max_chunk_size and current_chunk:
            # Create chunk
            chunk = DocumentChunk(
                content=current_chunk.strip(),
                context="",  # Skip context generation for speed
                metadata=metadata,
                chunk_id=f"chunk_{chunk_index}",
                sequence_num=chunk_index
            )
            chunks.append(chunk)
            current_chunk = para
            chunk_index += 1
        else:
            current_chunk += "\n\n" + para if current_chunk else para
    
    # Add final chunk
    if current_chunk.strip():
        chunk = DocumentChunk(
            content=current_chunk.strip(),
            context="",
            metadata=metadata,
            chunk_id=f"chunk_{chunk_index}",
            sequence_num=chunk_index
        )
        chunks.append(chunk)
    
    return chunks

def fast_index_document(qa_system, pdf_filename: str, text: str, metadata: Dict[str, Any]) -> bool:
    """Fast document indexing without context generation"""
    try:
        # Create collection for this document
        collection_name = qa_system._generate_collection_name(pdf_filename)
        
        # Check if collection already exists with content
        try:
            existing_collection = qa_system.qdrant_client.get_collection(collection_name)
            if existing_collection.points_count > 0:
                print(f"Collection {collection_name} already exists with {existing_collection.points_count} points, skipping")
                return True
        except Exception:
            pass
        
        # Create collection
        collection_name = qa_system._create_document_collection(pdf_filename)
        if not collection_name:
            return False
        
        # Ensure required metadata fields
        if 'source' not in metadata:
            metadata['source'] = pdf_filename
        if 'type' not in metadata:
            metadata['type'] = 'pdf'
        
        # Create simple chunks without LLM context
        chunks = create_simple_chunks(text, metadata)
        
        # Create embeddings and points
        texts = [chunk.content for chunk in chunks]
        embeddings = qa_system.embeddings.embed_documents(texts)
        
        from qdrant_client.models import PointStruct
        
        points = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            # Create payload with all necessary metadata
            payload = {
                'content': text,
                'source': metadata['source'],
                'company': metadata.get('company'),
                'chunk_index': i,
                'type': metadata.get('type', 'pdf'),
                'context': ''  # Empty context for speed
            }
            # Add any additional metadata fields
            for key, value in metadata.items():
                if key not in payload and value is not None:
                    payload[key] = value
                
            points.append(
                PointStruct(
                    id=i,
                    vector=embedding,
                    payload=payload
                )
            )
        
        # Add points to collection
        qa_system.qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )
        
        print(f"Fast indexed {len(points)} chunks in collection {collection_name}")
        return True
        
    except Exception as e:
        print(f"Error fast indexing document {pdf_filename}: {str(e)}")
        return False

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

def fast_reindex_documents(qa_system):
    """Fast reindex all documents"""
    print("Starting fast reindexing process (skipping LLM context generation)...")
    
    try:
        # Get all documents from database
        documents = get_all_policy_documents(qa_system.db)
        if not documents:
            print("No documents to process")
            return
            
        # Limit to first 20 documents
        documents = documents[:1]
        print(f"Found {len(documents)} documents to process (limited to first 20)")
        
        # Process each document
        successful_docs = 0
        for i, doc in enumerate(documents, 1):
            pdf_link = doc['pdf_link']
            print(f"Processing ({i}/{len(documents)}): {pdf_link}")
            
            try:
                # Download PDF
                pdf_content = qa_system.db.download_policy_pdf(pdf_link)
                if not pdf_content:
                    print(f"Failed to download {pdf_link}")
                    continue
                
                # Extract text
                text = qa_system.text_processor.extract_text_from_pdf(pdf_content)
                if not text or len(text.strip()) < 100:
                    print(f"No valid text extracted from {pdf_link}")
                    continue
                
                # Create metadata
                metadata = {
                    "company": doc.get('company_name'),
                    "start_date": doc.get('start_date'),
                    "end_date": doc.get('end_date'),
                    "type": "pdf"
                }
                
                # Fast index document
                if fast_index_document(qa_system, pdf_link, text, metadata):
                    successful_docs += 1
                
            except Exception as e:
                print(f"Error processing {pdf_link}: {str(e)}")
                continue
        
        print(f"\nFast reindexing completed. Successfully processed {successful_docs} out of {len(documents)} documents")
        
    except Exception as e:
        print(f"Error during fast reindexing: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--reset', action='store_true', help='Reset all indices before reindexing')
    args = parser.parse_args()
    
    try:
        # Initialize QA system
        db_connection_string = DatabaseConnection.get_connection_string(DatabaseConnection)
        if not db_connection_string:
            print("Error: Could not get database connection string")
            sys.exit(1)
        
        # Create a minimal QA system (skip LLM initialization)
        print("Initializing QA system for fast reindexing...")
        qa_system = DocumentQASystem(db_connection_string)
        
        # Reset indices if requested
        if args.reset:
            print("Resetting indices...")
            qa_system.reset_indices()
        
        # Perform fast reindexing
        fast_reindex_documents(qa_system)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1) 