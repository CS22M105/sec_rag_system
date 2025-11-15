"""
Complete test script for the RAG system
Tests the full pipeline: PDF -> Embeddings -> Retrieval -> LLM Answer
"""
import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.main import RAGSystem


def test_complete_rag_pipeline():
    """Test the complete RAG system end-to-end"""
    
    print("=" * 80)
    print("TESTING COMPLETE RAG PIPELINE")
    print("=" * 80)
    
    # Initialize RAG system
    print("\n1. Initializing RAG System...")
    rag_system = RAGSystem("config.yaml")
    
    # Path to your PDF
    pdf_path = "data/raw/apple_data.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"ERROR: PDF file not found at {pdf_path}")
        print("Please ensure apple_data.pdf is in the data/raw/ directory")
        return False
    
    # Check if index already exists
    print("\n2. Checking for existing index...")
    if not rag_system.load_existing_index():
        print("No existing index found. Processing PDF...")
        print("\n3. Processing the 10-K filing...")
        rag_system.process_filing(pdf_path)
    else:
        print("Existing index loaded successfully!")
    
    # Sample questions to show the user
    sample_questions = [
        "What are the main risk factors mentioned in the report?",
        "What is Apple's business strategy?",
        "What products does Apple sell?",
        "What were the total revenues for the last fiscal year?",
        "What segments does Apple operate in?",
        "What are the major legal proceedings mentioned?",
        "What is Apple's approach to research and development?",
        "How does Apple manage competition?",
        "What are the key markets for Apple?",
        "What is Apple's dividend policy?"
    ]
    
    print("\n" + "=" * 80)
    print("SAMPLE QUESTIONS YOU CAN ASK:")
    for i, question in enumerate(sample_questions, 1):
        print(f"{i}. {question}")
    print("=" * 80)
    
    # Ask the user for a query
    user_question = input("\nEnter your question: ").strip()
    if not user_question:
        print("No question entered. Exiting test.")
        return False
    
    print("\n" + "-" * 80)
    print(f"USER QUESTION: {user_question}")
    print("-" * 80)
    
    try:
        answer = rag_system.ask_question(user_question)
        print(f"\nANSWER:\n{answer}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    try:
        test_complete_rag_pipeline()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
