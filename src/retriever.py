from typing import List, Tuple
import numpy as np


class Retriever:
    def __init__(self, vector_store, text_processor):
        self.vector_store = vector_store
        self.text_processor = text_processor
    
    def retrieve(self, query: str, top_k: int = 5, similarity_threshold: float = 0.7) -> List[Tuple]:
        """
        Retrieve relevant chunks for a query
        
        Args:
            query: The question to search for
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of tuples: (text, score, metadata)
        """
        # Create query embedding
        query_embedding = self.text_processor.embedding_model.encode([query])
        
        # Search vector store
        results = self.vector_store.similarity_search(query_embedding, k=top_k)
        
        # Filter by threshold
        filtered_results = [
            (text, score, metadata) 
            for text, score, metadata in results 
            if score >= similarity_threshold
        ]
        
        return filtered_results
    
    def format_context(self, results: List[Tuple]) -> str:
        """
        Format retrieved chunks into a context string for the LLM
        
        Args:
            results: List of (text, score, metadata) tuples
            
        Returns:
            Formatted context string
        """
        if not results:
            return ""
        
        context_parts = []
        for i, (text, score, metadata) in enumerate(results, 1):
            section = metadata.get('section', 'unknown')
            context_parts.append(f"[Context {i} - Section: {section}, Relevance: {score:.3f}]\n{text}\n")
        
        return "\n".join(context_parts)


class EnhancedRetriever(Retriever):
    """Enhanced retriever with section-aware boosting"""
    
    def retrieve_for_question_type(self, question: str, top_k: int = 5):
        """Route questions to appropriate sections with boosting"""
        question_lower = question.lower()
        
        # Map question types to relevant sections
        if any(word in question_lower for word in ['risk', 'risk factors', 'challenge', 'threat']):
            preferred_sections = ['risk_factors']
            print(f"Routing to sections: {preferred_sections}")
        elif any(word in question_lower for word in ['business', 'strategy', 'model', 'product']):
            preferred_sections = ['business']
            print(f"Routing to sections: {preferred_sections}")
        elif any(word in question_lower for word in ['financial', 'revenue', 'income', 'profit']):
            preferred_sections = ['financials', 'md_a']
            print(f"Routing to sections: {preferred_sections}")
        else:
            preferred_sections = None
            print("Using general retrieval")
        
        return self.retrieve_with_section_boost(question, top_k, preferred_sections)
    
    def retrieve_with_section_boost(self, query: str, top_k: int = 5, preferred_sections: List[str] = None):
        """Retrieve with boosting for specific sections"""
        # First, get regular results
        query_embedding = self.text_processor.embedding_model.encode([query])
        results = self.vector_store.similarity_search(query_embedding, k=top_k * 3)  # Get more initially
        
        if not preferred_sections:
            return results[:top_k]
        
        # Boost scores for preferred sections
        boosted_results = []
        for text, score, metadata in results:
            boosted_score = score
            
            # Boost if in preferred section
            if metadata.get('section') in preferred_sections:
                boosted_score *= 1.5  # 50% boost
            
            boosted_results.append((text, boosted_score, metadata))
        
        # Sort by boosted score and return top_k
        boosted_results.sort(key=lambda x: x[1], reverse=True)
        return boosted_results[:top_k]