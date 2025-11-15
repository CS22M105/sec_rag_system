import os
import yaml
from .data_loader import SECDataLoader
from .text_processor import TextProcessor
from .vector_store import VectorStore
from .retriever import Retriever
from .generator import ResponseGenerator


class RAGSystem:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self.load_config(config_path)
        self.data_loader = SECDataLoader()
        self.text_processor = TextProcessor(
            chunk_size=self.config['processing']['chunk_size'],
            chunk_overlap=self.config['processing']['chunk_overlap']
        )
        self.vector_store = VectorStore(
            index_path=self.config['embedding']['vector_store_path']
        )
        self.retriever = None
        self.generator = ResponseGenerator(
            model=self.config['generation']['model'],
            temperature=self.config['generation']['temperature']
        )
    
    def load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def process_filing(self, pdf_path: str):
        """Process SEC filing and build vector store"""
        print("=" * 80)
        print("PROCESSING SEC 10-K FILING")
        print("=" * 80)
        
        # Step 1: Extract text from PDF
        print("\n1. Extracting text from PDF...")
        text = self.data_loader.extract_text_from_pdf(pdf_path)
        
        if not text:
            raise ValueError("Failed to extract text from PDF")
        
        # Step 2: Clean text
        print("\n2. Cleaning extracted text...")
        text = self.data_loader.clean_text(text)
        
        # Step 3: Identify sections and create chunks
        print("\n3. Identifying sections and creating chunks...")
        self.text_processor.load_embedding_model(self.config['embedding']['model_name'])
        
        # Use split_by_sections which calls identify_sections_with_content internally
        chunks = self.text_processor.split_by_sections(text)
        
        if not chunks:
            print("Warning: No sections found, falling back to simple chunking...")
            chunks = self.text_processor.split_into_chunks(text)
        
        print(f"Created {len(chunks)} chunks")
        
        # Step 4: Create embeddings
        print("\n4. Creating embeddings...")
        embeddings = self.text_processor.create_embeddings(chunks)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        
        # Step 5: Build vector store
        print("\n5. Building vector store...")
        # Prepare metadata
        metadata = []
        for chunk in chunks:
            metadata.append({
                'section': chunk.get('section', 'unknown'),
                'type': chunk.get('type', 'content')
            })
        
        self.vector_store.create_index(embeddings, chunks, metadata)
        self.vector_store.save_index()
        
        # Step 6: Initialize retriever
        self.retriever = Retriever(self.vector_store, self.text_processor)
        
        print("\n" + "=" * 80)
        print("✅ Vector store built successfully!")
        print("=" * 80)
    
    def load_existing_index(self):
        """Load existing vector store"""
        print("Loading existing vector store...")
        if self.vector_store.load_index():
            self.text_processor.load_embedding_model(self.config['embedding']['model_name'])
            self.retriever = Retriever(self.vector_store, self.text_processor)
            print("✅ Loaded existing vector store")
            return True
        print("⚠️  No existing vector store found")
        return False
    
    def ask_question(self, question: str) -> str:
        """Ask a question about the SEC filing"""
        if not self.retriever:
            raise ValueError("Retriever not initialized. Please process a filing first or load an existing index.")
        
        print(f"\n{'=' * 80}")
        print(f"Question: {question}")
        print(f"{'=' * 80}")
        
        # Step 1: Retrieve relevant context
        print("Retrieving relevant context...")
        results = self.retriever.retrieve(
            question,
            top_k=self.config['retrieval']['top_k'],
            similarity_threshold=self.config['retrieval']['similarity_threshold']
        )
        
        if not results:
            return "No relevant information found in the document to answer this question."
        
        print(f"Found {len(results)} relevant chunks")
        
        # Show relevance scores
        for i, (text, score, metadata) in enumerate(results, 1):
            section = metadata.get('section', 'unknown')
            print(f"  Chunk {i}: Section={section}, Score={score:.3f}")
        
        # Step 2: Format context
        context = self.retriever.format_context(results)
        
        # Step 3: Generate answer
        print("\nGenerating answer...")
        answer = self.generator.generate_answer(
            question, 
            context,
            max_tokens=self.config['generation']['max_tokens']
        )
        
        return answer