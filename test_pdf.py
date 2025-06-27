import os
from app.config import settings
from app.services.pdf_processor import pdf_processor
from app.services.vector_service import vector_service

# List files in upload directory
upload_dir = settings.upload_dir
print(f"📁 Upload directory: {upload_dir}")
print("📄 Files in upload directory:")

if os.path.exists(upload_dir):
    for file in os.listdir(upload_dir):
        file_path = os.path.join(upload_dir, file)
        file_size = os.path.getsize(file_path)
        print(f"  - {file} ({file_size:,} bytes)")
        
        # Test processing the first PDF
        if file.endswith('.pdf'):
            print(f"\n🧪 Testing COMPLETE PDF processing pipeline for: {file}")
            try:
                # Test the complete process_document method
                print("🚀 Testing complete document processing...")
                result = pdf_processor.process_document(file_path, 999, file)
                
                if result["success"]:
                    print(f"✅ PDF Processing successful!")
                    print(f"📊 Chunks created: {result['chunks_created']}")
                    print(f"📊 Total characters: {result['total_characters']}")
                    print(f"📊 Embeddings generated: {len(result['embeddings'])}")
                    
                    # Now test storing in Qdrant
                    print("💾 Testing Qdrant storage...")
                    vector_ids = vector_service.add_embeddings(
                        result["embeddings"], 
                        result["metadata"]
                    )
                    print(f"✅ Stored {len(vector_ids)} embeddings in Qdrant")
                    
                    # Test Qdrant search
                    print("🔍 Testing Qdrant search...")
                    if result["embeddings"]:
                        test_query_embedding = result["embeddings"][0]  # Use first embedding as test query
                        search_results = vector_service.search_similar(test_query_embedding, limit=5)
                        print(f"✅ Search test successful: Found {len(search_results)} results")
                        
                        # Show search results
                        for i, res in enumerate(search_results):
                            print(f"  Result {i+1}: Similarity {res['similarity']:.3f}, Content: {res['metadata']['content'][:100]}...")
                    
                    # Check collection status
                    print("📊 Checking Qdrant collection status...")
                    collection_info = vector_service.get_collection_info()
                    if collection_info:
                        print(f"✅ Collection info: {collection_info}")
                    else:
                        print("❌ Failed to get collection info")
                        
                else:
                    print(f"❌ PDF Processing failed: {result.get('error', 'Unknown error')}")
                
            except Exception as e:
                print(f"❌ Complete pipeline test failed: {e}")
                import traceback
                traceback.print_exc()
            break
else:
    print("❌ Upload directory does not exist!")

print("\n" + "="*50)
print("🏁 Test completed")