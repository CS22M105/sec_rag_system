from typing import List, Dict
import re
from sentence_transformers import SentenceTransformer
import numpy as np


class TextProcessor:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = None
    
    def load_embedding_model(self, model_name: str = "all-MiniLM-L6-v2"):
        """Load the sentence transformer model for embeddings"""
        print(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
    
    def create_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        """Create embeddings for text chunks"""
        if not self.embedding_model:
            self.load_embedding_model()
        
        # Extract just the text from chunks
        texts = [chunk['text'] for chunk in chunks]
        print(f"Creating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def split_by_sections(self, text: str) -> List[Dict]:
        """Split text by 10-K sections for better context"""
        sections = self.identify_sections_with_content(text)
        chunks = []
        
        print("Splitting into sections...")
        for section_name, section_text in sections.items():
            print(f"  - Processing section: {section_name} ({len(section_text)} chars)")
            
            # Further split large sections
            if len(section_text.split()) > 500:
                section_chunks = self.split_into_chunks(section_text)
                for chunk in section_chunks:
                    chunks.append({
                        'text': chunk['text'],
                        'section': section_name,
                        'type': 'section_content'
                    })
            else:
                chunks.append({
                    'text': section_text,
                    'section': section_name,
                    'type': 'section_content'
                })
        
        print(f"Created {len(chunks)} total chunks from sections")
        return chunks
    
    def identify_sections_with_content(self, text: str) -> Dict[str, str]:
        """Identify and extract complete sections from 10-K"""
        # Enhanced section patterns for 10-K
        section_patterns = {
            'business': r'ITEM\s*1\.?\s*BUSINESS',
            'risk_factors': r'ITEM\s*1A\.?\s*RISK\s*FACTORS',
            'properties': r'ITEM\s*2\.?\s*PROPERTIES',
            'legal': r'ITEM\s*3\.?\s*LEGAL\s*PROCEEDINGS',
            'md_a': r'ITEM\s*7\.?\s*MANAGEMENT\'S\s*DISCUSSION',
            'financials': r'ITEM\s*8\.?\s*FINANCIAL\s*STATEMENTS'
        }
        
        sections = {}
        
        # Find all section boundaries
        section_starts = {}
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                section_starts[section_name] = match.start()
                print(f"Found section '{section_name}' at position {match.start()}")
        
        if not section_starts:
            print("No sections found! Using fallback chunking.")
            # Fallback: if no sections found, use regular chunking
            return {'full_document': text}
        
        # Sort sections by position
        sorted_sections = sorted(section_starts.items(), key=lambda x: x[1])
        
        # Extract content between sections
        for i, (section_name, start_pos) in enumerate(sorted_sections):
            if i + 1 < len(sorted_sections):
                end_pos = sorted_sections[i + 1][1]
                section_content = text[start_pos:end_pos].strip()
            else:
                section_content = text[start_pos:].strip()
            
            sections[section_name] = section_content
        
        return sections
    
    def split_into_chunks(self, text: str) -> List[Dict]:
        """Improved chunking that respects sentence boundaries"""
        if not text:
            return []
        
        # Split into sentences first for better context preservation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'start_word': len(chunks) * self.chunk_size,
                    'end_word': len(chunks) * self.chunk_size + len(chunk_text.split())
                })
                
                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_start:]
                current_length = len(' '.join(current_chunk).split())
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'start_word': len(chunks) * self.chunk_size,
                'end_word': len(chunks) * self.chunk_size + len(chunk_text.split())
            })
        
        return chunks