from typing import Optional, Dict, Any, List
from models.schemas import SessionData
from services.google_sheets import GoogleSheetsService
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


class SessionAgent:
    """Agent responsible for managing user sessions"""
    
    def __init__(self, sheets_service: GoogleSheetsService):
        """
        Initialize session agent
        
        Args:
            sheets_service: GoogleSheetsService instance
        """
        self.sheets_service = sheets_service
    
    def create_session(
        self,
        session_id: str,
        user_id: str,
        initial_message: str
    ) -> Optional[SessionData]:
        """
        Create a new user session
        
        Args:
            session_id: Unique session identifier
            user_id: User identifier
            initial_message: Initial user message
            
        Returns:
            SessionData if successful, None otherwise
        """
        try:
            timestamp = datetime.utcnow().isoformat()
            success = self.sheets_service.create_session(
                session_id=session_id,
                user_id=user_id,
                message=initial_message,
                timestamp=timestamp
            )
            
            if success:
                session = SessionData(
                    session_id=session_id,
                    user_id=user_id,
                    messages=[{
                        "role": "user",
                        "content": initial_message,
                        "timestamp": timestamp
                    }],
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                logger.info(f"Session created: {session_id} for user {user_id}")
                return session
            return None
        except Exception as e:
            logger.error(f"Error creating session: {str(e)}")
            return None
    
    def get_session(self, session_id: str) -> Optional[SessionData]:
        """
        Retrieve existing session
        
        Args:
            session_id: Session identifier
            
        Returns:
            SessionData if found, None otherwise
        """
        try:
            session_data = self.sheets_service.lookup_session(session_id)
            if session_data:
                # Parse conversation history
                messages = []
                conv_history = session_data.get('conversation_history', '[]')
                try:
                    messages = json.loads(conv_history)
                except (json.JSONDecodeError, TypeError):
                    messages = []
                
                # Parse context data to get user_id
                context_data = session_data.get('context_data', '{}')
                try:
                    context = json.loads(context_data)
                    user_id = context.get('user_id', 'unknown')
                except (json.JSONDecodeError, TypeError):
                    user_id = 'unknown'
                
                return SessionData(
                    session_id=session_id,
                    user_id=user_id,
                    messages=messages,
                    created_at=datetime.fromisoformat(session_data.get('created_at', datetime.utcnow().isoformat())),
                    updated_at=datetime.utcnow()
                )
            logger.warning(f"Session not found: {session_id}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving session: {str(e)}")
            return None
    
    def add_message(self, session_id: str, role: str, content: str) -> bool:
        """
        Add message to session history and update last activity
        
        Args:
            session_id: Session identifier
            role: Message role (user/assistant)
            content: Message content
            
        Returns:
            True if successful
        """
        try:
            # Get current session
            session = self.get_session(session_id)
            if not session:
                return False
            
            # Add new message
            messages = session.messages or []
            messages.append({
                'role': role,
                'content': content,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            # Update session with new messages and last activity
            last_activity = datetime.utcnow().isoformat()
            success = self.sheets_service.update_session(
                session_id=session_id,
                conversation_history=json.dumps(messages),
                last_activity=last_activity
            )
            
            if success:
                logger.info(f"Message added to session {session_id}")
            return success
        except Exception as e:
            logger.error(f"Error adding message: {str(e)}")
            return False
    
    def close_session(self, session_id: str) -> bool:
        """
        Close a session (update last_activity timestamp)
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful
        """
        try:
            session = self.get_session(session_id)
            if not session:
                return False
            
            last_activity = datetime.utcnow().isoformat()
            success = self.sheets_service.update_session(
                session_id=session_id,
                conversation_history=json.dumps(session.messages or []),
                last_activity=last_activity
            )
            
            if success:
                logger.info(f"Session closed: {session_id}")
            return success
        except Exception as e:
            logger.error(f"Error closing session: {str(e)}")
            return False
    
    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get complete message history for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of messages
        """
        try:
            session = self.get_session(session_id)
            if session:
                return session.messages or []
            return []
        except Exception as e:
            logger.error(f"Error retrieving session history: {str(e)}")
            return []
    
    def save_customer_info(
        self,
        session_id: str,
        name: str,
        website: str,
        email: str,
        phone: str
    ) -> bool:
        """
        Save customer information for a session
        
        Args:
            session_id: Session identifier
            name: Customer name
            website: Customer website
            email: Customer email
            phone: Customer phone number
            
        Returns:
            True if saved successfully
        """
        try:
            success = self.sheets_service.save_customer_info(
                session_id=session_id,
                name=name,
                website=website,
                email=email,
                phone=phone
            )
            
            if success:
                logger.info(f"Customer info saved for session {session_id}")
            return success
        except Exception as e:
            logger.error(f"Error saving customer info: {str(e)}")
            return False
