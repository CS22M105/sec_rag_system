from openai import OpenAI
from typing import List
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ResponseGenerator:
    def __init__(self, model: str = None, temperature: float = 0.1):
        self.model = model or os.getenv('MODEL', 'gpt-4o-mini')  # ← Changed
        self.temperature = temperature
        
        # Get API key from environment
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY in your .env file"
            )
        
        self.client = OpenAI(api_key=api_key)
    
    def generate_answer(self, question: str, context: str, max_tokens: int = 500) -> str:
        """Generate answer using LLM with retrieved context"""
        
        prompt = f"""Based on the following context from the company's SEC 10-K filing, answer the question thoroughly and accurately.

Context:
{context}

Question: {question}

Instructions:
- Answer based ONLY on the provided context
- If the context doesn't contain relevant information, say so
- Be specific and cite numbers/dates when available
- Keep the answer focused and professional
- If you're unsure, acknowledge the uncertainty

Answer:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a financial analyst expert at analyzing SEC filings. Provide accurate, well-structured answers based solely on the provided context."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=max_tokens
            )
            
            answer = response.choices[0].message.content
            return answer
            
        except Exception as e:
            error_msg = f"Error generating answer: {str(e)}"
            print(f"ERROR: {error_msg}")
            
            # Provide more helpful error messages
            if "authentication" in str(e).lower():
                return "ERROR: Invalid OpenAI API key. Please check your .env file."
            elif "rate_limit" in str(e).lower():
                return "ERROR: OpenAI rate limit exceeded. Please try again in a moment."
            elif "insufficient_quota" in str(e).lower():
                return "ERROR: OpenAI quota exceeded. Please check your account."
            else:
                return f"ERROR: Failed to generate answer - {str(e)}"


class OpenSourceGenerator:
    """
    Alternative using open-source models (placeholder for Hugging Face integration)
    This can be implemented if you want to avoid OpenAI costs
    """
    def __init__(self, model_name: str = "microsoft/phi-2"):
        self.model_name = model_name
        # Implementation would require transformers library
        pass
    
    def generate_answer(self, question: str, context: str) -> str:
        """Generate answer using open-source model"""
        # TODO: Implement with Hugging Face transformers
        return "Open-source model implementation - coming soon!"