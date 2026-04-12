import streamlit as st
from src.main import RAGSystem
import os

st.set_page_config(page_title="SEC 10-K Q&A System", page_icon="📊", layout="wide")

def initialize_rag_system():
    """Initialize the RAG system"""
    if 'rag_system' not in st.session_state:
        try:
            with st.spinner("Initializing RAG system..."):
                st.session_state.rag_system = RAGSystem("config.yaml")
                
                # Try to load existing index
                if not st.session_state.rag_system.load_existing_index():
                    st.warning("⚠️ No processed 10-K filing found.")
                    st.info("Please upload and process a 10-K PDF using the sidebar.")
                    return False
                else:
                    st.success("✅ RAG system initialized successfully!")
                    return True
        except Exception as e:
            st.error(f"Error initializing RAG system: {e}")
            return False
    return True

def main():
    st.title("📊 SEC 10-K Q&A System")
    st.markdown("Ask questions about company SEC 10-K filings")
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # File upload section in sidebar
    st.sidebar.header("📁 Process New Filing")
    uploaded_file = st.sidebar.file_uploader("Upload 10-K PDF", type="pdf")
    
    if uploaded_file is not None:
        if st.sidebar.button("🔄 Process Filing", type="primary"):
            with st.spinner("Processing filing... This may take a few minutes."):
                try:
                    # Save uploaded file
                    file_path = f"data/raw/{uploaded_file.name}"
                    os.makedirs("data/raw", exist_ok=True)
                    
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Initialize and process
                    st.session_state.rag_system = RAGSystem("config.yaml")
                    st.session_state.rag_system.process_filing(file_path)
                    
                    st.sidebar.success("✅ Filing processed successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.sidebar.error(f"Error processing filing: {e}")
    
    # Initialize RAG system
    if not initialize_rag_system():
        st.stop()
    
    # Main chat interface
    st.header("💬 Ask Questions")
    
    # Sample questions
    st.subheader("Quick Questions")
    sample_questions = [
        "What are the main risk factors?",
        "What is the company's business strategy?",
        "What were the total revenues last year?",
        "What segments does the company operate in?",
        "What are the major legal proceedings?",
        "How does the company manage competition?",
        "What is the company's R&D approach?",
        "What are the key markets?"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(sample_questions):
        with cols[i % 2]:
            if st.button(question, key=f"btn_{i}"):
                st.session_state.current_question = question
    
    # Question input
    question = st.text_input(
        "Or type your own question:",
        value=st.session_state.get('current_question', ''),
        placeholder="e.g., What are the company's main competitors?",
        key="question_input"
    )
    
    # Clear button
    if st.button("🗑️ Clear"):
        st.session_state.current_question = ""
        st.rerun()
    
    # Get answer button
    if st.button("🔍 Get Answer", type="primary") and question:
        with st.spinner("Searching through the 10-K filing..."):
            try:
                # Create columns for answer and context
                answer_col, context_col = st.columns([2, 1])
                
                with answer_col:
                    st.subheader("Answer")
                    
                    # Get answer
                    answer = st.session_state.rag_system.ask_question(question)
                    
                    # Display answer
                    st.markdown(answer)
                
                with context_col:
                    st.subheader("Context Used")
                    
                    # Get retrieval results for transparency
                    results = st.session_state.rag_system.retriever.retrieve(
                        question,
                        top_k=st.session_state.rag_system.config['retrieval']['top_k'],
                        similarity_threshold=st.session_state.rag_system.config['retrieval']['similarity_threshold']
                    )
                    
                    for i, (text, score, metadata) in enumerate(results, 1):
                        section = metadata.get('section', 'unknown')
                        with st.expander(f"Context {i}: {section} (Score: {score:.3f})"):
                            st.text(text[:300] + "..." if len(text) > 300 else text)
                
            except Exception as e:
                st.error(f"Error getting answer: {e}")
                import traceback
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **How it works:**
    1. Upload a 10-K PDF
    2. System extracts and indexes the content
    3. Ask questions in natural language
    4. Get AI-powered answers based on the filing
    
    **Tips:**
    - Be specific in your questions
    - Questions about financials, risks, and strategy work best
    - The system only answers based on the filed document
    """)

if __name__ == "__main__":
    main()