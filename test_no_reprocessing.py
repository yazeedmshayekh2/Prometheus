#!/usr/bin/env python3
"""
Test script to verify that the system doesn't reprocess documents
"""

from main import DocumentQASystem
import time

def test_no_reprocessing():
    """Test that documents are not reprocessed when they already exist"""
    
    print("=== Testing No Reprocessing ===")
    
    # Initialize the system
    qa_system = DocumentQASystem()
    
    # Test national ID
    national_id = "28140001175"
    
    # First query - should process the document
    print("\n--- First Query (may process document) ---")
    start_time = time.time()
    result1 = qa_system.intelligent_search_and_answer(
        national_id, 
        "What is the annual maximum limit per person for coverage?"
    )
    first_query_time = time.time() - start_time
    print(f"First query took: {first_query_time:.2f} seconds")
    print(f"Answer: {result1.answer[:100]}...")
    
    # Second query - should NOT reprocess the document
    print("\n--- Second Query (should skip processing) ---")
    start_time = time.time()
    result2 = qa_system.intelligent_search_and_answer(
        national_id, 
        "What are the exclusions and limitations?"
    )
    second_query_time = time.time() - start_time
    print(f"Second query took: {second_query_time:.2f} seconds")
    print(f"Answer: {result2.answer[:100]}...")
    
    # Third query - should also skip processing
    print("\n--- Third Query (should also skip processing) ---")
    start_time = time.time()
    result3 = qa_system.intelligent_search_and_answer(
        national_id, 
        "What is the coinsurance percentage?"
    )
    third_query_time = time.time() - start_time
    print(f"Third query took: {third_query_time:.2f} seconds")
    print(f"Answer: {result3.answer[:100]}...")
    
    # Summary
    print(f"\n=== Performance Summary ===")
    print(f"First query:  {first_query_time:.2f}s (may include processing)")
    print(f"Second query: {second_query_time:.2f}s (should be faster)")
    print(f"Third query:  {third_query_time:.2f}s (should be faster)")
    
    if second_query_time < first_query_time * 0.8 and third_query_time < first_query_time * 0.8:
        print("✅ SUCCESS: Subsequent queries are significantly faster!")
        print("✅ Documents are NOT being reprocessed")
    else:
        print("❌ WARNING: Queries are taking similar time - documents may be reprocessing")
    
    print(f"\n=== Collection Status ===")
    collections = qa_system.qdrant_client.get_collections().collections
    for col in collections:
        if col.name.startswith('doc_'):
            info = qa_system.qdrant_client.get_collection(col.name)
            print(f"Collection: {col.name} - {info.points_count} points")

if __name__ == "__main__":
    test_no_reprocessing() 