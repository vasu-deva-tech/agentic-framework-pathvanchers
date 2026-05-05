import os
from typing import Optional, List, Dict, Any
from supabase import create_client, Client
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)


class SupabaseService:
    """Service for managing embeddings and vector search in Supabase"""
    
    def __init__(self, openrouter_api_key: str):
        """
        Initialize Supabase service with Open Router API
        
        Args:
            openrouter_api_key: Open Router API key
        """
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not all([supabase_url, supabase_key]):
            raise ValueError("Missing Supabase URL or Key")
        
        self.supabase: Client = create_client(supabase_url, supabase_key)
        
        # Initialize Open Router client (OpenAI-compatible API)
        self.openrouter_client = OpenAI(
            api_key=openrouter_api_key,
            base_url="https://openrouter.io/api/v1"
        )
        self.embedding_model = "text-embedding-3-small"
        
    def create_embedding(self, text: str) -> Optional[List[float]]:
        """
        Create embedding for text using Open Router (OpenAI-compatible)
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None
        """
        try:
            response = self.openrouter_client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error creating embedding: {str(e)}")
            return None
    
    def store_document(
        self,
        content: str,
        metadata: Dict[str, Any],
        table_name: str = "documents"
    ) -> Optional[Dict[str, Any]]:
        """
        Store document with embedding in Supabase
        
        Args:
            content: Document content
            metadata: Document metadata
            table_name: Supabase table name
            
        Returns:
            Stored document data or None
        """
        try:
            embedding = self.create_embedding(content)
            if not embedding:
                return None
            
            document = {
                "content": content,
                "embedding": embedding,
                "metadata": metadata
            }
            
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
        Perform vector similarity search
        
        Args:
            query: Search query
            limit: Number of results to return
            table_name: Supabase table name
            
        Returns:
            List of similar documents
        """
        try:
            query_embedding = self.create_embedding(query)
            if not query_embedding:
                return []
            
            # Using Supabase RPC for vector search
            response = self.supabase.rpc(
                'match_documents',
                {
                    'query_embedding': query_embedding,
                    'match_threshold': 0.5,
                    'match_count': limit
                }
            ).execute()
            
            results = []
            if response.data:
                for item in response.data:
                    results.append({
                        'id': item.get('id'),
                        'content': item.get('content'),
                        'similarity': item.get('similarity', 0),
                        'metadata': item.get('metadata', {})
                    })
            
            logger.info(f"Vector search found {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Error performing vector search: {str(e)}")
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
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for text using Open Router (OpenAI-compatible) text-embedding-3-small model
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector (list of floats) or None if error
        """
        try:
            if not text or not isinstance(text, str):
                logger.warning("Invalid text input for embedding")
                return None
            
            response = self.openrouter_client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding for text (length: {len(embedding)})")
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return None
    
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
