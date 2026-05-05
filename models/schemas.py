from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class ChatbotRequest(BaseModel):
    """Request schema for chatbot endpoint"""
    message: str = Field(..., description="User message")
    session_id: str = Field(..., description="Session identifier")
    user_id: str = Field(default="anonymous", description="User identifier")


class ChatbotResponse(BaseModel):
    """Response schema for chatbot endpoint"""
    answer: str = Field(..., description="AI-generated response")
    session_id: str = Field(..., description="Session identifier")
    status: str = Field(default="success", description="Status of the request")
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)


class SessionData(BaseModel):
    """Session data schema"""
    session_id: str
    user_id: str
    messages: list = Field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class IntentAnalysis(BaseModel):
    """Intent analysis result"""
    intent: str = Field(..., description="Detected intent")
    confidence: float = Field(..., description="Confidence score")
    entities: Optional[dict] = Field(default_factory=dict)


class EmbeddingResult(BaseModel):
    """Embedding search result"""
    id: str
    content: str
    similarity: float
