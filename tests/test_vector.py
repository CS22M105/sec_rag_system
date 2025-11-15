import os
import sys
import numpy as np

from data_loader import SECDataLoader
from text_processor import TextProcessor
from vector_store import VectorStore

def test_data_loader():
    """Test the data loader component"""
    print("=" * 60)
    print("TESTING DATA LOADER")
    print("=" * 60)
    
    loader = SECDataLoader()
    pdf_path = "/Users/farhatjahan/Desktop/YU/YU2/AI/sec_rag_system/data/raw/apple_data.pdf"
    
    # Test 1: PDF Validation
    print("1. Testing PDF validation...")
    is_valid = loader.validate_pdf(pdf_path)
    print(f"   PDF Valid: {is_valid}")
    
    if not is_valid:
        print("   ❌ PDF validation failed! Cannot continue.")
        return False
    
    # Test 2: Document Info
    print("2. Testing document info...")
    doc_info = loader.get_document_info(pdf_path)
    print(f"   Pages: {doc_info.get('total_pages', 'N/A')}")
    print(f"   File Size: {doc_info.get('file_size_mb', 'N/A')} MB")
    
    # Test 3: Text Extraction
    print("3. Testing text extraction...")
    raw_text = loader.extract_text_from_pdf(pdf_path)
    print(f"   Raw text length: {len(raw_text)} characters")
    
    if len(raw_text) == 0:
        print("   ❌ Text extraction failed!")
        return False
    
    # Test 4: Text Cleaning
    print("4. Testing text cleaning...")
    clean_text = loader.clean_text(raw_text)
    print(f"   Clean text length: {len(clean_text)} characters")
    
    if len(clean_text) == 0:
        print("   ❌ Text cleaning failed!")
        return False
    
    print("   ✅ Data loader tests passed!")
    return clean_text

def test_text_processor(clean_text):
    """Test the text processor component"""
    print("\n" + "=" * 60)
    print("TESTING TEXT PROCESSOR")
    print("=" * 60)
    
    processor = TextProcessor(chunk_size=800, chunk_overlap=100)
    
    # Test 1: Section Identification
    print("1. Testing section identification...")
    sections = processor.identify_sections_with_content(clean_text)
    print(f"   Sections found: {list(sections.keys())}")
    
    for section_name, section_text in sections.items():
        print(f"     - {section_name}: {len(section_text)} chars")
    
    # Test 2: Chunk Creation
    print("2. Testing chunk creation...")
    chunks = processor.split_by_sections(clean_text)
    print(f"   Total chunks created: {len(chunks)}")
    
    # Show sample chunks
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        print(f"     Chunk {i+1}: {chunk.get('section', 'unknown')}, {len(chunk['text'])} chars")
        print(f"       Preview: {chunk['text'][:100]}...")
    
    if len(chunks) == 0:
        print("   ❌ Chunk creation failed!")
        return None
    
    # Test 3: Embedding Model Loading
    print("3. Testing embedding model...")
    processor.load_embedding_model()
    print(f"   Embedding model loaded: {processor.embedding_model is not None}")
    
    # Test 4: Embedding Creation
    print("4. Testing embedding creation...")
    embeddings = processor.create_embeddings(chunks)
    print(f"   Embeddings shape: {embeddings.shape}")
    print(f"   Number of embeddings: {len(embeddings)}")
    
    if len(embeddings) == 0:
        print("   ❌ Embedding creation failed!")
        return None
    
    print("   ✅ Text processor tests passed!")
    return chunks, embeddings

def test_vector_store(chunks, embeddings):
    """Test the vector store component"""
    print("\n" + "=" * 60)
    print("TESTING VECTOR STORE")
    print("=" * 60)
    
    vector_store = VectorStore("./models/test_faiss_index")
    
    # Test 1: Index Creation
    print("1. Testing index creation...")
    
    # Prepare metadata
    metadata = [{'section': chunk.get('section', 'unknown'), 'type': chunk.get('type', 'content')} 
               for chunk in chunks]
    
    vector_store.create_index(embeddings, chunks, metadata)
    print(f"   Index created with {len(vector_store.chunks)} chunks")
    
    # Test 2: Index Saving
    print("2. Testing index saving...")
    vector_store.save_index()
    print("   Index saved successfully")
    
    # Test 3: Similarity Search
    print("3. Testing similarity search...")
    
    # Create a test query embedding
    processor = TextProcessor()
    processor.load_embedding_model()
    
    test_queries = [
        "risk factors",
        "financial statements", 
        "business strategy"
    ]
    
    for query in test_queries:
        print(f"   Query: '{query}'")
        query_embedding = processor.embedding_model.encode([query])
        results = vector_store.similarity_search(query_embedding, k=2)
        
        for i, (text, score, meta) in enumerate(results):
            section = meta.get('section', 'unknown')
            print(f"     Result {i+1}: score={score:.3f}, section={section}")
            print(f"       Preview: {text[:80]}...")
    
    # Test 4: Index Loading
    print("4. Testing index loading...")
    new_vector_store = VectorStore("./models/test_faiss_index")
    load_success = new_vector_store.load_index()
    print(f"   Index loaded: {load_success}")
    if load_success:
        print(f"   Loaded chunks: {len(new_vector_store.chunks)}")
    
    print("   ✅ Vector store tests passed!")
    return load_success

def cleanup_test_files():
    """Clean up test files"""
    test_files = [
        "./models/test_faiss_index.faiss",
        "./models/test_faiss_index_meta.pkl"
    ]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Cleaned up: {file_path}")

def main():
    """Main test function"""
    print("CORE COMPONENTS TEST")
    print("Testing: data_loader → text_processor → vector_store")
    print()
    
    try:
        # Test Data Loader
        clean_text = test_data_loader()
        if clean_text is False:
            return
        
        # Test Text Processor  
        result = test_text_processor(clean_text)
        if result is None:
            return
            
        chunks, embeddings = result
        
        # Test Vector Store
        success = test_vector_store(chunks, embeddings)
        
        if success:
            print("\n" + "=" * 60)
            print("🎉 ALL TESTS PASSED! 🎉")
            print("=" * 60)
            print("All core components are working correctly:")
            print("✅ Data Loader - PDF reading and text extraction")
            print("✅ Text Processor - Section identification and embeddings") 
            print("✅ Vector Store - Index creation and similarity search")
        else:
            print("\n❌ Vector store test failed!")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up test files
        print("\nCleaning up test files...")
        cleanup_test_files()

if __name__ == "__main__":
    main()