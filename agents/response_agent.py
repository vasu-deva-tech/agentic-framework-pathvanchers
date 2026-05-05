from typing import Optional, List, Dict, Any
from openai import OpenAI
import json
import logging

logger = logging.getLogger(__name__)


class ResponseAgent:
    """Agent responsible for generating contextual responses as Briva"""
    
    def __init__(self, openrouter_api_key: str):
        """
        Initialize response agent with Open Router API
        
        Args:
            openrouter_api_key: Open Router API key
        """
        # Initialize Open Router client (OpenAI-compatible API)
        self.openrouter_client = OpenAI(
            api_key=openrouter_api_key,
            base_url="https://openrouter.io/api/v1"
        )
        self.model = "openai/gpt-4o-mini"
    
    def generate_response(
        self,
        message: str,
        session_id: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        knowledge_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate structured response using Briva AI assistant with Open Router
        
        Args:
            message: User message
            session_id: Session identifier
            conversation_history: Conversation history (list of {role, content})
            knowledge_context: Retrieved knowledge base context
            
        Returns:
            Dictionary with structured output:
            {
                "answer": str,
                "requires_human": bool,
                "sentiment": "positive" | "neutral" | "negative"
            }
        """
        try:
            # Build system prompt for Briva
            system_prompt = """You are Briva, a helpful AI assistant for PathVancer. 
You help businesses with their queries professionally and warmly.
Always respond in a friendly, concise manner.
If you have knowledge base context, use it to answer accurately.

You must respond with ONLY valid JSON (no markdown, no code blocks) in this exact format:
{
  "answer": "Your response here",
  "requires_human": false,
  "sentiment": "neutral"
}

The sentiment field should be: "positive" if the query is positive, "negative" if it's negative/complaint, "neutral" otherwise.
The requires_human field should be true if the issue needs human intervention."""
            
            # Build messages array
            messages = []
            
            # Add conversation history
            if conversation_history:
                for msg in conversation_history[-5:]:  # Use last 5 messages for context
                    messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    })
            
            # Build user message with knowledge context
            user_content = message
            if knowledge_context:
                user_content = f"{message}\n\n[Knowledge Base Context]\n{knowledge_context}"
            
            messages.append({"role": "user", "content": user_content})
            
            # Call Open Router API with JSON response format
            response = self.openrouter_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system_prompt}] + messages,
                temperature=0.7,
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            
            response_text = response.choices[0].message.content
            
            # Parse JSON response
            try:
                result = json.loads(response_text)
                
                # Validate required fields
                if "answer" not in result:
                    result["answer"] = response_text
                if "requires_human" not in result:
                    result["requires_human"] = False
                if "sentiment" not in result:
                    result["sentiment"] = "neutral"
                
                # Validate sentiment values
                if result["sentiment"] not in ["positive", "negative", "neutral"]:
                    result["sentiment"] = "neutral"
                
                logger.info(f"Response generated for session {session_id} | Sentiment: {result['sentiment']}")
                return result
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Error parsing JSON response: {str(e)}")
                return {
                    "answer": response_text,
                    "requires_human": False,
                    "sentiment": "neutral"
                }
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "answer": "I apologize, but I encountered an error processing your request. Please try again.",
                "requires_human": True,
                "sentiment": "neutral"
            }
    
    def generate_followup_suggestions(self, response: str, intent: str) -> List[str]:
        """
        Generate suggested follow-up questions
        
        Args:
            response: Generated response
            intent: User intent
            
        Returns:
            List of suggested follow-up questions
        """
        try:
            system_prompt = """Generate 2-3 natural follow-up questions that the user might ask next, 
given the intent and the response provided. Return as a JSON array of strings with NO extra text."""
            
            prompt = f"""Intent: {intent}
Response: {response}

Generate follow-up suggestions as a JSON array. Example: ["Question 1?", "Question 2?", "Question 3?"]"""
            
            response = self.openrouter_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,
                max_tokens=200
            )
            
            response_text = response.choices[0].message.content
            
            # Try to parse JSON
            try:
                suggestions = json.loads(response_text)
                if isinstance(suggestions, list):
                    return suggestions[:3]
            except json.JSONDecodeError:
                pass
            
            return []
        except Exception as e:
            logger.error(f"Error generating follow-up suggestions: {str(e)}")
            return []
    
    def validate_response(self, user_message: str, generated_response: str) -> Dict[str, Any]:
        """
        Validate quality of generated response
        
        Args:
            user_message: Original user message
            generated_response: Generated response to validate
            
        Returns:
            Validation result with scores and feedback
        """
        try:
            system_prompt = """Evaluate the quality of the chatbot response on these criteria:
1. relevance (0-1): How relevant is the response to the user message?
2. clarity (0-1): How clear and understandable is the response?
3. completeness (0-1): Does it fully address the user's question?
4. tone (0-1): Is the tone appropriate and professional?

Return ONLY valid JSON with NO extra text."""
            
            prompt = f"""User message: {user_message}
Response: {generated_response}

Evaluate the response and return JSON with scores and feedback. Example: {{"relevance": 0.9, "clarity": 0.8, "completeness": 0.85, "tone": 0.9, "feedback": "Good response"}}"""
            
            response = self.openrouter_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=300
            )
            
            response_text = response.choices[0].message.content
            
            try:
                validation = json.loads(response_text)
            except json.JSONDecodeError:
                validation = {
                    "relevance": 0.5,
                    "clarity": 0.5,
                    "completeness": 0.5,
                    "tone": 0.5,
                    "feedback": "Unable to evaluate response"
                }
            
            return validation
        except Exception as e:
            logger.error(f"Error validating response: {str(e)}")
            return {
                "relevance": 0.0,
                "clarity": 0.0,
                "completeness": 0.0,
                "tone": 0.0,
                "feedback": f"Validation error: {str(e)}"
            }
