"""
Response Generation Service

Handles:
- Context-aware response generation
- Response caching
- Token management
- Error handling
"""
from typing import List, Dict
from openai import AzureOpenAI
from loguru import logger
import time

from config import config
from database.redis_cache import redis_cache


class GenerationService:
    """
    Service for generating responses with Azure OpenAI GPT
    
    Features:
    - Context-aware generation
    - Automatic caching
    - Token management
    - Streaming support (optional)
    """
    
    def __init__(self):
        """Initialize Azure OpenAI client"""
        self.client = AzureOpenAI(
            api_version=config.AZURE_OPENAI_API_VERSION,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_API_KEY,
        )
        
        self.model = config.AZURE_OPENAI_DEPLOYMENT
        self.temperature = config.TEMPERATURE
        self.max_tokens = config.MAX_TOKENS
        
        logger.info(f"✅ Generation service initialized (model={self.model})")
    
    def generate_response(
        self,
        query: str,
        context_chunks: List[Dict],
        use_cache: bool = True,
        context_hash: str = None
    ) -> Dict:
        """
        Generate response based on query and retrieved context
        
        Args:
            query: User's query
            context_chunks: Retrieved chunks with metadata
            use_cache: Whether to use cache
            context_hash: Hash of context for caching
            
        Returns:
            Dictionary with answer and metadata
        """
        start_time = time.time()
        
        # Check cache
        if use_cache and context_hash:
            cached = redis_cache.get_response(query, context_hash)
            if cached is not None:
                elapsed = (time.time() - start_time) * 1000
                logger.info(f"✅ Cached response retrieved in {elapsed:.2f}ms")
                return {
                    "answer": cached,
                    "cached": True,
                    "response_time_ms": elapsed,
                    "sources": [c['document']['filename'] for c in context_chunks if c.get('document')]
                }
        
        # Build context
        context = self._build_context(context_chunks)
        
        # Create prompt
        system_message = self._get_system_prompt()
        user_message = self._build_user_message(query, context)
        
        logger.debug(f"Generating response (context length: {len(context)} chars)...")
        
        try:
            # Call GPT
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            answer = response.choices[0].message.content
            
            # Cache the response
            if use_cache and context_hash:
                redis_cache.set_response(query, context_hash, answer)
            
            elapsed = (time.time() - start_time) * 1000
            
            logger.info(f"✅ Response generated in {elapsed:.2f}ms ({len(answer)} chars)")
            
            return {
                "answer": answer,
                "cached": False,
                "response_time_ms": elapsed,
                "sources": [c['document']['filename'] for c in context_chunks if c.get('document')],
                "tokens_used": response.usage.total_tokens,
                "model": self.model
            }
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            raise
    
    def _build_context(self, chunks: List[Dict]) -> str:
        """
        Build context string from retrieved chunks
        
        Args:
            chunks: Retrieved chunks with metadata
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get('document', {}).get('filename', 'Unknown')
            content = chunk.get('content', '')
            similarity = chunk.get('similarity', 0)
            
            context_parts.append(
                f"[Document {i}: {source} (relevance: {similarity:.2f})]\n{content}"
            )
        
        return "\n\n".join(context_parts)
    
    def _get_system_prompt(self) -> str:
        """
        Get system prompt for GPT
        
        Returns:
            System prompt string
        """
        return """You are a helpful AI assistant that answers questions based on provided context documents.

Your responsibilities:
1. Answer questions accurately using ONLY the information in the provided context
2. Cite which document number you're referencing (e.g., "According to Document 1...")
3. If the context doesn't contain enough information, say so honestly
4. Be concise but comprehensive
5. Use clear, professional language

Important:
- Do NOT make up information not in the context
- Do NOT use your general knowledge if it contradicts the context
- If multiple documents provide conflicting information, mention both perspectives
- Always cite your sources by document number"""
    
    def _build_user_message(self, query: str, context: str) -> str:
        """
        Build user message with context and query
        
        Args:
            query: User's query
            context: Retrieved context
            
        Returns:
            User message string
        """
        return f"""Context Documents:
{context}

---

Question: {query}

Please provide a detailed answer based on the context above. Remember to cite which documents you're using."""
    
    def generate_summary(self, text: str, max_length: int = 200) -> str:
        """
        Generate summary of text
        
        Args:
            text: Text to summarize
            max_length: Maximum summary length
            
        Returns:
            Summary text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"Summarize the following text in {max_length} characters or less."
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                temperature=0.3,  # Lower temperature for summaries
                max_tokens=max_length // 2  # Rough estimate
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return text[:max_length] + "..."
    
    def get_stats(self) -> Dict:
        """Get generation service statistics"""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }


# Singleton instance
generator = GenerationService()


if __name__ == "__main__":
    # Test generation service
    print("Testing Generation Service...")
    
    # Mock context chunks
    mock_chunks = [
        {
            "content": "Python is a high-level programming language known for its simplicity.",
            "similarity": 0.95,
            "document": {"filename": "python_basics.txt"}
        },
        {
            "content": "Python was created by Guido van Rossum in 1991.",
            "similarity": 0.87,
            "document": {"filename": "python_history.txt"}
        }
    ]
    
    # Test would require actual API call
    stats = generator.get_stats()
    print(f"Generation Stats: {stats}")
    
    print("\n✅ Generation Service interface ready")
