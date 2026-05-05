from typing import Optional, Dict, Any
from models.schemas import IntentAnalysis
from services.supabase_service import SupabaseService
from openai import OpenAI
import json
import logging
import re
import os

logger = logging.getLogger(__name__)


class IntentAgent:
    """Agent responsible for analyzing user intent and context"""
    
    # Regex patterns for extracting contact information
    EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    WEBSITE_PATTERN = r'https?://(?:www\.)?[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*|(?:www\.)?[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9]{2,})'
    PHONE_PATTERN = r'(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
    
    def __init__(self, openrouter_api_key: str, supabase_service: SupabaseService):
        """
        Initialize intent agent with Open Router API
        
        Args:
            openrouter_api_key: Open Router API key
            supabase_service: SupabaseService instance for context retrieval
        """
        # Initialize Open Router client (OpenAI-compatible API)
        self.openrouter_client = OpenAI(
            api_key=openrouter_api_key,
            base_url="https://openrouter.io/api/v1"
        )
        self.supabase_service = supabase_service
        self.model = "openai/gpt-4o-mini"
    
    def analyze_intent(self, message: str, context: Optional[list] = None) -> IntentAnalysis:
        """
        Analyze user message intent
        
        Args:
            message: User message
            context: Optional conversation history
            
        Returns:
            IntentAnalysis with detected intent and confidence
        """
        try:
            system_prompt = """You are an intent classification agent. Analyze the user message and classify it into one of these intents:
            - greeting: General greetings
            - question: Information seeking
            - request: Action or service request
            - feedback: User feedback or complaint
            - other: Anything else
            
            Respond in JSON format with fields: intent, confidence (0-1), entities (dict with extracted entities)"""
            
            context_str = ""
            if context:
                context_str = "\n\nConversation history:\n" + "\n".join([f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in context[-3:]])
            
            response = self.openrouter_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{message}{context_str}"}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            response_text = response.choices[0].message.content
            # Parse JSON response
            try:
                intent_data = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback if response is not valid JSON
                intent_data = {
                    "intent": "other",
                    "confidence": 0.5,
                    "entities": {}
                }
            
            analysis = IntentAnalysis(
                intent=intent_data.get('intent', 'other'),
                confidence=float(intent_data.get('confidence', 0.5)),
                entities=intent_data.get('entities', {})
            )
            
            logger.info(f"Intent analyzed: {analysis.intent} (confidence: {analysis.confidence})")
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing intent: {str(e)}")
            return IntentAnalysis(intent="other", confidence=0.0, entities={})
    
    def retrieve_context(self, query: str, limit: int = 3) -> list:
        """
        Retrieve relevant context from vector database
        
        Args:
            query: Query string
            limit: Number of results to retrieve
            
        Returns:
            List of relevant documents
        """
        try:
            results = self.supabase_service.vector_search(query, limit=limit)
            logger.info(f"Retrieved {len(results)} context documents")
            return results
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []
    
    def extract_entities(self, message: str) -> Dict[str, Any]:
        """
        Extract entities from user message
        
        Args:
            message: User message
            
        Returns:
            Dictionary of extracted entities
        """
        try:
            system_prompt = """Extract named entities from the user message.
            Return JSON with entity types and values.
            Common entity types: person, organization, location, product, date, time, quantity.
            If no entities found, return empty dict."""
            
            response = self.openrouter_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                temperature=0.2,
                max_tokens=200
            )
            
            response_text = response.choices[0].message.content
            try:
                entities = json.loads(response_text)
            except json.JSONDecodeError:
                entities = {}
            
            logger.info(f"Extracted entities: {entities}")
            return entities
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return {}
    
    def get_intent_recommendation(self, message: str, context: Optional[list] = None) -> Dict[str, Any]:
        """
        Get comprehensive intent analysis with recommendations
        
        Args:
            message: User message
            context: Optional conversation history
            
        Returns:
            Dictionary with intent analysis and recommendations
        """
        try:
            intent_analysis = self.analyze_intent(message, context)
            entities = self.extract_entities(message)
            context_docs = self.retrieve_context(message)
            
            recommendation = {
                "intent": intent_analysis.intent,
                "confidence": intent_analysis.confidence,
                "entities": entities,
                "context_docs": [
                    {
                        "content": doc.get('content', ''),
                        "similarity": doc.get('similarity', 0)
                    } for doc in context_docs
                ]
            }
            
            return recommendation
        except Exception as e:
            logger.error(f"Error getting intent recommendation: {str(e)}")
            return {"intent": "other", "confidence": 0.0, "entities": {}, "context_docs": []}
    
    def _extract_contact_info(self, message: str) -> Dict[str, Any]:
        """
        Extract contact information from message using regex patterns
        
        Args:
            message: User message
            
        Returns:
            Dictionary with extracted contact info (name, website, email, phone)
        """
        extracted = {
            "name": None,
            "website": None,
            "email": None,
            "phone": None
        }
        
        try:
            # Extract email
            email_match = re.search(self.EMAIL_PATTERN, message)
            if email_match:
                extracted["email"] = email_match.group(0)
                logger.debug(f"Extracted email: {extracted['email']}")
            
            # Extract website
            website_matches = re.findall(self.WEBSITE_PATTERN, message)
            if website_matches:
                # Take the first match
                website = website_matches[0]
                # Clean up the website if it's a tuple from groups
                if isinstance(website, tuple):
                    website = website[0] if website[0] else website[1] if len(website) > 1 else None
                extracted["website"] = website
                logger.debug(f"Extracted website: {extracted['website']}")
            
            # Extract phone
            phone_match = re.search(self.PHONE_PATTERN, message)
            if phone_match:
                # Reconstruct phone number from groups
                phone = f"{phone_match.group(1)}-{phone_match.group(2)}-{phone_match.group(3)}"
                extracted["phone"] = phone
                logger.debug(f"Extracted phone: {extracted['phone']}")
            
        except Exception as e:
            logger.error(f"Error extracting contact info: {str(e)}")
        
        return extracted
    
    def detect_intent(self, message: str) -> Dict[str, Any]:
        """
        Classify buying intent and extract contact information
        
        Args:
            message: User message
            
        Returns:
            Dictionary with:
            {
                "intent": "new_user" | "provide_details" | "buying_intent" | "general_query",
                "has_contact_info": bool,
                "extracted": {
                    "name": str,
                    "website": str,
                    "email": str,
                    "phone": str
                }
            }
        """
        try:
            # Extract contact information using regex
            extracted = self._extract_contact_info(message)
            has_contact_info = any(v is not None for v in extracted.values())
            
            # Classify intent using OpenAI
            system_prompt = """You are a buying intent classifier for a sales chatbot. Analyze the customer message and classify their intent into one of these categories:

- "new_user": First-time user introducing themselves or expressing initial interest
- "provide_details": User is providing company information, website, or contact details
- "buying_intent": User shows clear interest in purchasing or needs specific product/service information
- "general_query": General questions or information seeking without clear buying intent

Respond in JSON format with field: intent (one of the above values)"""
            
            response = self.openrouter_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                temperature=0.3,
                max_tokens=100
            )
            
            response_text = response.choices[0].message.content
            
            # Parse JSON response
            try:
                intent_data = json.loads(response_text)
                intent = intent_data.get('intent', 'general_query')
            except json.JSONDecodeError:
                # Fallback intent classification
                intent = 'general_query'
            
            # Validate intent is one of expected values
            valid_intents = ["new_user", "provide_details", "buying_intent", "general_query"]
            if intent not in valid_intents:
                intent = 'general_query'
            
            result = {
                "intent": intent,
                "has_contact_info": has_contact_info,
                "extracted": extracted
            }
            
            logger.info(f"Intent detected: {intent} | Contact info: {has_contact_info}")
            return result
            
        except Exception as e:
            logger.error(f"Error detecting intent: {str(e)}")
            return {
                "intent": "general_query",
                "has_contact_info": False,
                "extracted": {
                    "name": None,
                    "website": None,
                    "email": None,
                    "phone": None
                }
            }
