#!/usr/bin/env python3
"""
FAQ Indexing Script for Qdrant Vector Database

This script indexes FAQ questions and answers from the SQL Server database
into a dedicated Qdrant collection using multilingual embeddings.
"""

import sys
import argparse
from pathlib import Path
from db_utils import DatabaseConnection
from main import DocumentQASystem

def index_faq_data(qa_system, reset_collection=False):
    """Index FAQ data into Qdrant collection"""
    try:
        print("=" * 60)
        print("FAQ INDEXING PROCESS")
        print("=" * 60)
        
        if reset_collection:
            print("🔄 Resetting FAQ collection...")
            try:
                qa_system.qdrant_client.delete_collection(qa_system.FAQ_COLLECTION_NAME)
                print(f"✅ Deleted existing collection: {qa_system.FAQ_COLLECTION_NAME}")
            except Exception as e:
                print(f"ℹ️  Collection {qa_system.FAQ_COLLECTION_NAME} doesn't exist or couldn't be deleted: {e}")
            
            # Reinitialize vector stores to recreate the FAQ collection
            qa_system._initialize_vector_store()
        
        # Check collection status
        try:
            collection_info = qa_system.qdrant_client.get_collection(qa_system.FAQ_COLLECTION_NAME)
            print(f"📊 Current FAQ collection status:")
            print(f"   Collection: {qa_system.FAQ_COLLECTION_NAME}")
            print(f"   Points: {collection_info.points_count}")
            print(f"   Vector size: {collection_info.config.params.vectors.size}")
            print(f"   Distance metric: {collection_info.config.params.vectors.distance}")
        except Exception as e:
            print(f"❌ Error getting collection info: {e}")
            return False
        
        # Index FAQ data
        print("\n🚀 Starting FAQ indexing...")
        success = qa_system.index_faq_data()
        
        if success:
            # Get final collection status
            try:
                final_info = qa_system.qdrant_client.get_collection(qa_system.FAQ_COLLECTION_NAME)
                print(f"\n✅ FAQ indexing completed successfully!")
                print(f"📊 Final collection status:")
                print(f"   Total points indexed: {final_info.points_count}")
                print(f"   Collection size: {final_info.points_count} FAQ entries")
                
                # Test search functionality
                print(f"\n🔍 Testing FAQ search functionality...")
                test_questions = [
                    "What are the coverage limits?",
                    "How do I submit a claim?",
                    "ما هي حدود التغطية؟",  # Arabic: What are the coverage limits?
                ]
                
                for test_q in test_questions:
                    print(f"\n   Testing: '{test_q}'")
                    results = qa_system.search_faq_semantic(test_q, k=2, similarity_threshold=0.5)
                    if results:
                        for i, result in enumerate(results[:2]):
                            print(f"   Result {i+1}: {result['similarity']:.3f} - {result['question'][:50]}...")
                    else:
                        print(f"   No results found for: {test_q}")
                
                print(f"\n🎉 FAQ indexing and testing completed!")
                return True
            except Exception as e:
                print(f"❌ Error getting final collection status: {e}")
                return False
        else:
            print(f"❌ FAQ indexing failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error in FAQ indexing process: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Index FAQ data into Qdrant vector database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python index_faq.py                    # Index FAQ data
  python index_faq.py --reset           # Reset collection and reindex
  python index_faq.py --test-only       # Only test search functionality
        """
    )
    parser.add_argument('--reset', action='store_true', 
                       help='Reset FAQ collection before indexing')
    parser.add_argument('--test-only', action='store_true',
                       help='Only test search functionality without indexing')
    
    args = parser.parse_args()
    
    try:
        print("Initializing system...")
        
        # Get database connection string
        db_connection_string = DatabaseConnection.get_connection_string(DatabaseConnection)
        if not db_connection_string:
            print("❌ Error: Could not get database connection string")
            sys.exit(1)
        
        # Initialize QA system
        print("🔧 Initializing DocumentQA system...")
        qa_system = DocumentQASystem(db_connection_string)
        
        if not hasattr(qa_system, 'faq_vector_store') or qa_system.faq_vector_store is None:
            print("❌ Error: FAQ vector store not initialized properly")
            sys.exit(1)
        
        print("✅ System initialized successfully")
        
        if args.test_only:
            print("\n🔍 Testing search functionality only...")
            test_questions = [
                "What are the coverage limits?",
                "How do I submit a claim?", 
                "What is covered under maternity benefits?",
                "ما هي حدود التغطية؟",  # Arabic
            ]
            
            for test_q in test_questions:
                print(f"\n   Testing: '{test_q}'")
                results = qa_system.search_faq_semantic(test_q, k=3, similarity_threshold=0.4)
                if results:
                    for i, result in enumerate(results):
                        print(f"   Result {i+1}: {result['similarity']:.3f} - {result['question'][:60]}...")
                else:
                    print(f"   No results found")
        else:
            # Perform FAQ indexing
            success = index_faq_data(qa_system, reset_collection=args.reset)
            
            if success:
                print("\n🎉 FAQ indexing completed successfully!")
                sys.exit(0)
            else:
                print("\n❌ FAQ indexing failed!")
                sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 