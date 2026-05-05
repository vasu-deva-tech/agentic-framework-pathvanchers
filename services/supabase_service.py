import os
from typing import Optional, List, Dict, Any
from supabase import create_client, Client
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)

# Initialize OpenRouter embedding client
embedding_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)


def generate_embedding(text: str) -> list[float]:
    """
    Generate embedding for text using OpenRouter API
    
    Args:
        text: Text to generate embedding for
        
    Returns:
        List of float values representing the embedding
    """
    try:
        response = embedding_client.embeddings.create(
            model="openai/text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        return []


def search_knowledge_base(query: str, supabase_client: Client, match_threshold: float = 0.7, match_count: int = 5) -> List[Dict[str, Any]]:
    """
    Search RAG knowledge base using vector similarity in Supabase
    
    Args:
        query: Search query text
        supabase_client: Supabase client instance
        match_threshold: Similarity threshold (0-1)
        match_count: Number of results to return
        
    Returns:
        List of matching documents with similarity scores
    """
    try:
        # Generate embedding for the query
        query_embedding = generate_embedding(query)
        
        if not query_embedding:
            logger.warning("Failed to generate embedding for query")
            return []
        
        # Search using vector similarity in "chatbot" table
        result = supabase_client.rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "match_threshold": match_threshold,
                "match_count": match_count,
                "table_name": "chatbot"
            }
        ).execute()
        
        return result.data if result.data else []
    except Exception as e:
        logger.error(f"Error searching knowledge base: {str(e)}")
        return []


def format_rag_context(search_results: List[Dict[str, Any]]) -> str:
    """
    Format RAG search results into context string for AI response
    
    Args:
        search_results: List of documents from RAG search
        
    Returns:
        Formatted context string
    """
    if not search_results:
        return "No relevant context found."
    
    context = ""
    for i, result in enumerate(search_results, 1):
        content = result.get("content", "")
        similarity = result.get("similarity", 0)
        context += f"[{i}] {content}\n"
    
    return context


class SupabaseService:
    """Service for managing vector search in Supabase with pre-existing embeddings"""
    
    def __init__(self):
        """
        Initialize Supabase service
        Embeddings are managed externally - this service only performs searches
        on existing embeddings already stored in Supabase
        """
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not all([supabase_url, supabase_key]):
            raise ValueError("Missing Supabase URL or Key")
        
        self.supabase: Client = create_client(supabase_url, supabase_key)
    
    def search_rag(self, query: str, match_threshold: float = 0.7, match_count: int = 5) -> List[Dict[str, Any]]:
        """
        Search RAG knowledge base using vector similarity
        
        Args:
            query: Search query text
            match_threshold: Similarity threshold (0-1)
            match_count: Number of results to return
            
        Returns:
            List of matching documents with similarity scores
        """
        return search_knowledge_base(query, self.supabase, match_threshold, match_count)
    
    def get_rag_context(self, query: str, match_count: int = 5) -> str:
        """
        Get formatted RAG context for a query
        
        Args:
            query: Search query text
            match_count: Number of results to include
            
        Returns:
            Formatted context string for AI response
        """
        results = self.search_rag(query, match_count=match_count)
        return format_rag_context(results)
        

    
    def store_document(
        self,
        content: str,
        metadata: Dict[str, Any],
        embedding: Optional[List[float]] = None,
        table_name: str = "documents"
    ) -> Optional[Dict[str, Any]]:
        """
        Store document in Supabase (embeddings must be provided externally)
        
        Args:
            content: Document content
            metadata: Document metadata
            embedding: Pre-generated embedding vector (optional)
            table_name: Supabase table name
            
        Returns:
            Stored document data or None
        """
        try:
            document = {
                "content": content,
                "metadata": metadata
            }
            
            if embedding:
                document["embedding"] = embedding
            
            response = self.supabase.table(table_name).insert(document).execute()
            logger.info(f"Document stored successfully in {table_name}")
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error storing document: {str(e)}")
            return None
    
    def vector_search(
        self,
        query: str,
        limit: int = 5,
        table_name: str = "documents"
    ) -> List[Dict[str, Any]]:
        """
        Perform text-based search for documents
        For vector similarity search, use search_knowledge_base() with a pre-generated embedding
        
        Args:
            query: Search query (text-based keyword search)
            limit: Number of results to return
            table_name: Supabase table name
            
        Returns:
            List of matching documents
        """
        try:
            # Perform text-based keyword search on content
            response = self.supabase.table(table_name).select("*").ilike(
                "content",
                f"%{query}%"
            ).limit(limit).execute()
            
            results = []
            if response.data:
                for item in response.data:
                    results.append({
                        'id': item.get('id'),
                        'content': item.get('content'),
                        'metadata': item.get('metadata', {})
                    })
            
            logger.info(f"Text search found {len(results)} results for query: {query}")
            return results
        except Exception as e:
            logger.error(f"Error performing text search: {str(e)}")
            return []
    
    def get_document(self, doc_id: str, table_name: str = "documents") -> Optional[Dict[str, Any]]:
        """
        Retrieve document by ID
        
        Args:
            doc_id: Document ID
            table_name: Supabase table name
            
        Returns:
            Document data or None
        """
        try:
            response = self.supabase.table(table_name).select("*").eq("id", doc_id).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error retrieving document: {str(e)}")
            return None
    
    def delete_document(self, doc_id: str, table_name: str = "documents") -> bool:
        """
        Delete document by ID
        
        Args:
            doc_id: Document ID
            table_name: Supabase table name
            
        Returns:
            True if deleted successfully
        """
        try:
            self.supabase.table(table_name).delete().eq("id", doc_id).execute()
            logger.info(f"Document {doc_id} deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False
    
    def search_knowledge_base(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search knowledge base in chatbot table using text search
        
        Args:
            query: Search query
            limit: Number of results to return (default 5)
            
        Returns:
            List of matching documents from chatbot table
        """
        try:
            response = self.supabase.table("chatbot").select("*").ilike(
                "content", f"%{query}%"
            ).limit(limit).execute()
            
            results = []
            if response.data:
                for item in response.data:
                    results.append({
                        'id': item.get('id'),
                        'content': item.get('content'),
                        'session_id': item.get('session_id'),
                        'created_at': item.get('created_at'),
                        'metadata': item.get('metadata', {})
                    })
            
            logger.info(f"Knowledge base search found {len(results)} results for query: {query}")
            return results
        except Exception as e:
            logger.error(f"Error searching knowledge base: {str(e)}")
            return []
    
    def save_to_supabase(
        self,
        session_id: str,
        question: str,
        answer: str
    ) -> Optional[Dict[str, Any]]:
        """
        Save question and answer to chatbot table in Supabase
        
        Args:
            session_id: Session identifier
            question: User question
            answer: Generated answer
            
        Returns:
            Saved record data or None if error
        """
        try:
            from datetime import datetime
            
            record = {
                "session_id": session_id,
                "question": question,
                "answer": answer,
                "created_at": datetime.utcnow().isoformat()
            }
            
            response = self.supabase.table("chatbot").insert(record).execute()
            logger.info(f"Saved Q&A to chatbot table for session {session_id}")
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error saving to Supabase: {str(e)}")
            return None
    
    def list_documents(self, limit: int = 10, table_name: str = "documents") -> List[Dict[str, Any]]:
        """
        List documents from table
        
        Args:
            limit: Number of documents to return
            table_name: Supabase table name
            
        Returns:
            List of documents
        """
        try:
            response = self.supabase.table(table_name).select("*").limit(limit).execute()
            return response.data or []
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            return []
    

    
    def search_knowledge_base(
        self,
        query_embedding: List[float],
        threshold: float = 0.7,
        match_count: int = 5,
        table_name: str = "documents"
    ) -> List[Dict[str, Any]]:
        """
        Search knowledge base using vector embedding with Supabase RPC function
        
        Args:
            query_embedding: Query embedding vector (list of floats)
            threshold: Similarity threshold (0.0-1.0), default 0.7
            match_count: Number of results to return, default 5
            table_name: Supabase table name (used for logging)
            
        Returns:
            List of matching documents with similarity scores
            Format: [{content, similarity, ...}, ...]
        """
        try:
            if not query_embedding or not isinstance(query_embedding, list):
                logger.warning("Invalid query embedding input")
                return []
            
            # Call Supabase RPC function "match_documents"
            response = self.supabase.rpc(
                'match_documents',
                {
                    'query_embedding': query_embedding,
                    'match_threshold': threshold,
                    'match_count': match_count
                }
            ).execute()
            
            results = []
            if response.data:
                for item in response.data:
                    results.append({
                        'id': item.get('id'),
                        'content': item.get('content'),
                        'similarity': float(item.get('similarity', 0)),
                        'metadata': item.get('metadata', {}),
                        'created_at': item.get('created_at'),
                        'updated_at': item.get('updated_at')
                    })
            
            logger.info(f"Knowledge base search found {len(results)} results (threshold: {threshold})")
            return results
        except Exception as e:
            logger.error(f"Error searching knowledge base: {str(e)}")
            return []
    
    def save_embedding(
        self,
        session_id: str,
        question: str,
        answer: str,
        embedding: List[float],
        table_name: str = "qa_embeddings"
    ) -> Optional[Dict[str, Any]]:
        """
        Save or update question-answer pair with embedding to Supabase
        
        Args:
            session_id: Session identifier for grouping
            question: User question
            answer: AI generated answer
            embedding: Embedding vector for the Q&A pair
            table_name: Supabase table name
            
        Returns:
            Upserted record data or None if error
        """
        try:
            if not all([session_id, question, answer, embedding]):
                logger.warning("Missing required fields for saving embedding")
                return None
            
            record = {
                'session_id': session_id,
                'question': question,
                'answer': answer,
                'embedding': embedding,
                'created_at': 'now()',  # Supabase will handle this
                'updated_at': 'now()'   # Supabase will handle this
            }
            
            # Upsert operation: insert or update if exists
            response = self.supabase.table(table_name).upsert(
                record,
                on_conflict='session_id,question'  # Composite key for conflict resolution
            ).execute()
            
            if response.data:
                logger.info(f"Embedding saved for session {session_id}")
                return response.data[0]
            else:
                logger.warning(f"No data returned from embedding save")
                return None
        except Exception as e:
            logger.error(f"Error saving embedding: {str(e)}")
            return None
    
    def get_session_embeddings(
        self,
        session_id: str,
        table_name: str = "qa_embeddings",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all embeddings for a session
        
        Args:
            session_id: Session identifier
            table_name: Supabase table name
            limit: Maximum number of records to return
            
        Returns:
            List of session embeddings
        """
        try:
            response = self.supabase.table(table_name).select("*").eq(
                "session_id", session_id
            ).limit(limit).execute()
            
            logger.info(f"Retrieved {len(response.data or [])} embeddings for session {session_id}")
            return response.data or []
        except Exception as e:
            logger.error(f"Error retrieving session embeddings: {str(e)}")
            return []
