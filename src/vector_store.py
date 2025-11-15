import faiss
import numpy as np
import pickle
import os
from typing import List, Tuple, Dict


class VectorStore:
    def __init__(self, index_path: str = "./models/faiss_index"):
        self.index_path = index_path
        self.index = None
        self.chunks = []
        self.metadata = []
    
    def create_index(self, embeddings: np.ndarray, chunks: List[Dict], metadata: List[Dict] = None):
        """Create FAISS index from embeddings"""
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Using inner product for similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        # Store chunks - ensure they're dictionaries
        self.chunks = chunks
        self.metadata = metadata if metadata else [{}] * len(chunks)
        
        print(f"✅ Index created with {len(self.chunks)} chunks")
        # Debug: Check chunk structure
        if self.chunks and len(self.chunks) > 0:
            print(f"   Sample chunk type: {type(self.chunks[0])}")
            if isinstance(self.chunks[0], dict):
                print(f"   Sample chunk keys: {self.chunks[0].keys()}")
    
    def save_index(self):
        """Save FAISS index and metadata"""
        if not os.path.exists(os.path.dirname(self.index_path)):
            os.makedirs(os.path.dirname(self.index_path))
        
        # Save FAISS index
        faiss.write_index(self.index, f"{self.index_path}.faiss")
        
        # Save chunks and metadata - ensure proper structure
        save_data = {
            'chunks': self.chunks,
            'metadata': self.metadata
        }
        
        with open(f"{self.index_path}_meta.pkl", 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"✅ Index saved to {self.index_path}")
    
    def load_index(self) -> bool:
        """Load FAISS index and metadata"""
        try:
            # Check if files exist
            if not os.path.exists(f"{self.index_path}.faiss"):
                print(f"Index file not found: {self.index_path}.faiss")
                return False
            
            if not os.path.exists(f"{self.index_path}_meta.pkl"):
                print(f"Metadata file not found: {self.index_path}_meta.pkl")
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(f"{self.index_path}.faiss")
            
            # Load chunks and metadata
            with open(f"{self.index_path}_meta.pkl", 'rb') as f:
                data = pickle.load(f)
                self.chunks = data['chunks']
                self.metadata = data['metadata']
            
            print(f"✅ Loaded index with {len(self.chunks)} chunks")
            
            # Debug: Verify chunk structure after loading
            if self.chunks and len(self.chunks) > 0:
                print(f"   Loaded chunk type: {type(self.chunks[0])}")
                if isinstance(self.chunks[0], dict):
                    print(f"   Loaded chunk keys: {self.chunks[0].keys()}")
                else:
                    print(f"   ⚠️  WARNING: Chunks are not dictionaries!")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading index: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def similarity_search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple]:
        """Search for similar chunks"""
        if self.index is None:
            raise ValueError("Index not initialized")
        
        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1)
        
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                
                # Handle both dict and string formats (for backward compatibility)
                if isinstance(chunk, dict):
                    text = chunk.get('text', str(chunk))
                elif isinstance(chunk, str):
                    text = chunk
                else:
                    text = str(chunk)
                
                metadata = self.metadata[idx] if idx < len(self.metadata) else {}
                
                results.append((text, float(score), metadata))
        
        return results