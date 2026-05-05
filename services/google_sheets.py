import os
import json
import base64
from typing import Optional, List, Dict, Any
from google.oauth2.service_account import Credentials
import gspread
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class GoogleSheetsService:
    """Service for managing session data in Google Sheets using gspread"""
    
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
    
    def __init__(self):
        """
        Initialize Google Sheets service
        
        Uses GOOGLE_SERVICE_ACCOUNT_JSON (base64 encoded) and SPREADSHEET_ID env vars
        """
        self.spreadsheet_id = os.getenv("SPREADSHEET_ID")
        if not self.spreadsheet_id:
            raise ValueError("SPREADSHEET_ID environment variable not set")
        
        self.client = self._authenticate()
        self.spreadsheet = self.client.open_by_key(self.spreadsheet_id)
        logger.info(f"Google Sheets authenticated for spreadsheet {self.spreadsheet_id}")
        
    def _authenticate(self):
        """Authenticate using base64 encoded service account credentials"""
        try:
            # Get base64 encoded credentials from env
            encoded_creds = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
            if not encoded_creds:
                raise ValueError("GOOGLE_SERVICE_ACCOUNT_JSON environment variable not set")
            
            # Decode base64 credentials
            decoded_creds = base64.b64decode(encoded_creds).decode('utf-8')
            creds_dict = json.loads(decoded_creds)
            
            # Create credentials object
            credentials = Credentials.from_service_account_info(
                creds_dict,
                scopes=self.SCOPES
            )
            
            # Authorize and return gspread client
            client = gspread.Authorized(auth=credentials)
            logger.info("Google Sheets service authenticated successfully")
            return client
        except Exception as e:
            logger.error(f"Failed to authenticate Google Sheets: {str(e)}")
            raise
    
    def lookup_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Look up session from SessionInformationChatbot sheet
        
        Args:
            session_id: Session identifier
            
        Returns:
            Row dict with keys: id, conversation_history, context_data, created_at, last_activity
            Returns None if not found
        """
        try:
            sheet = self.spreadsheet.worksheet("SessionInformationChatbot")
            
            # Get all rows
            all_rows = sheet.get_all_records()
            
            # Find row matching session_id
            for row in all_rows:
                if row.get('id') == session_id:
                    logger.info(f"Session found: {session_id}")
                    return row
            
            logger.warning(f"Session not found: {session_id}")
            return None
        except Exception as e:
            logger.error(f"Error looking up session {session_id}: {str(e)}")
            return None
    
    def create_session(
        self,
        session_id: str,
        user_id: str,
        message: str,
        timestamp: str
    ) -> bool:
        """
        Create a new session in SessionInformationChatbot sheet
        
        Args:
            session_id: Unique session identifier
            user_id: User identifier
            message: Initial message
            timestamp: Creation timestamp
            
        Returns:
            True if session created successfully
        """
        try:
            sheet = self.spreadsheet.worksheet("SessionInformationChatbot")
            
            # Initialize conversation history
            conversation_history = json.dumps([{
                "role": "user",
                "content": message,
                "timestamp": timestamp
            }])
            
            # Append new row
            new_row = [
                session_id,              # id
                conversation_history,    # conversation_history
                json.dumps({"user_id": user_id}),  # context_data
                timestamp,               # created_at
                timestamp                # last_activity
            ]
            
            sheet.append_row(new_row)
            logger.info(f"Session created: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Error creating session {session_id}: {str(e)}")
            return False
    
    def update_session(
        self,
        session_id: str,
        conversation_history: str,
        last_activity: str
    ) -> bool:
        """
        Update existing session in SessionInformationChatbot sheet
        
        Args:
            session_id: Session identifier
            conversation_history: Updated conversation history (JSON string)
            last_activity: Last activity timestamp
            
        Returns:
            True if updated successfully
        """
        try:
            sheet = self.spreadsheet.worksheet("SessionInformationChatbot")
            
            # Get all rows to find the session
            all_rows = sheet.get_all_records()
            
            for idx, row in enumerate(all_rows, start=2):  # start=2 because row 1 is header
                if row.get('id') == session_id:
                    # Update conversation_history (column B) and last_activity (column E)
                    sheet.update_cell(idx, 2, conversation_history)  # Column B
                    sheet.update_cell(idx, 5, last_activity)         # Column E
                    logger.info(f"Session updated: {session_id}")
                    return True
            
            logger.warning(f"Session not found for update: {session_id}")
            return False
        except Exception as e:
            logger.error(f"Error updating session {session_id}: {str(e)}")
            return False
    
    def save_customer_info(
        self,
        session_id: str,
        name: str,
        website: str,
        email: str,
        phone: str
    ) -> bool:
        """
        Save or update customer information in ChatbotCustomerInformation sheet
        
        Args:
            session_id: Session identifier (used as id)
            name: Customer name
            website: Customer website
            email: Customer email address
            phone: Customer contact number
            
        Returns:
            True if saved/updated successfully
        """
        try:
            sheet = self.spreadsheet.worksheet("ChatbotCustomerInformation")
            
            # Get all rows to check if customer exists
            all_rows = sheet.get_all_records()
            
            customer_data = [
                session_id,  # id
                name,        # name
                website,     # website
                email,       # email_address
                phone        # contact_number
            ]
            
            # Look for existing customer by id
            for idx, row in enumerate(all_rows, start=2):
                if row.get('id') == session_id:
                    # Update existing row
                    sheet.update_cell(idx, 1, session_id)
                    sheet.update_cell(idx, 2, name)
                    sheet.update_cell(idx, 3, website)
                    sheet.update_cell(idx, 4, email)
                    sheet.update_cell(idx, 5, phone)
                    logger.info(f"Customer info updated: {session_id}")
                    return True
            
            # Append new row if not found
            sheet.append_row(customer_data)
            logger.info(f"Customer info created: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving customer info for {session_id}: {str(e)}")
            return False
