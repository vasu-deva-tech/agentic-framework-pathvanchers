import os
import logging
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from models.schemas import ChatbotRequest, ChatbotResponse
from services.google_sheets import GoogleSheetsService
from services.supabase_service import SupabaseService
from agents.session_agent import SessionAgent
from agents.intent_agent import IntentAgent
from agents.response_agent import ResponseAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
sheets_service = None
supabase_service = None
session_agent = None
intent_agent = None
response_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    global sheets_service, supabase_service, session_agent, intent_agent, response_agent
    
    # Startup
    logger.info("Initializing services...")
    try:
        # Initialize Google Sheets service
        sheets_service = GoogleSheetsService()
        logger.info("✓ Google Sheets service initialized")
        
        # Get Open Router API key
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        
        if not openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY not configured")
        
        # Initialize Supabase service (uses existing embeddings stored in Supabase)
        supabase_service = SupabaseService()
        logger.info("✓ Supabase service initialized")
        
        # Initialize agents with Open Router API key
        session_agent = SessionAgent(sheets_service)
        logger.info("✓ Session agent initialized")
        
        intent_agent = IntentAgent(openrouter_api_key, supabase_service)
        logger.info("✓ Intent agent initialized")
        
        response_agent = ResponseAgent(openrouter_api_key)
        logger.info("✓ Response agent initialized")
        
        logger.info("All services initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down services...")


# Create FastAPI app
app = FastAPI(
    title="Pathvancer Chatbot",
    description="Multi-user agentic chatbot API",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Info"])
async def root():
    """
    Root endpoint with API information
    
    Returns:
        API metadata including name, version, docs URL, and health status
    """
    return {
        "name": "Pathvancer Chatbot",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "ok"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/chatbot-session", response_model=ChatbotResponse, tags=["Chatbot"])
async def chatbot_session(request: ChatbotRequest):
    """
    Process chatbot message and return response
    
    Args:
        request: ChatbotRequest with message, session_id, user_id (defaults to "anonymous")
        
    Returns:
        ChatbotResponse with answer, session_id, and status
    """
    try:
        # Validate session_id is present (required field)
        if not request.session_id:
            logger.warning("Missing session_id in request")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="session_id is required"
            )
        
        # Validate message is present
        if not request.message:
            logger.warning(f"Missing message for session {request.session_id}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="message is required"
            )
        
        logger.info(f"[SESSION:{request.session_id}] Processing message from {request.user_id}")
        
        # Step 1: Lookup session from Google Sheets
        try:
            logger.debug(f"[SESSION:{request.session_id}] Looking up session...")
            session = session_agent.get_session(request.session_id)
        except Exception as e:
            logger.error(f"[SESSION:{request.session_id}] Error looking up session: {str(e)}")
            logger.debug(traceback.format_exc())
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to lookup session: {str(e)}"
            )
        
        # If no session found, create a new one
        if not session:
            try:
                logger.info(f"[SESSION:{request.session_id}] Creating new session...")
                session = session_agent.create_session(
                    session_id=request.session_id,
                    user_id=request.user_id,
                    initial_message=request.message
                )
                if not session:
                    logger.error(f"[SESSION:{request.session_id}] Failed to create session (returned None)")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Failed to create session"
                    )
                
                # Return greeting for new session
                greeting_message = "Hello! I'm Briva 👋 To help you best, please share: 1. Your Name 2. Company website 3. Phone Number 4. Email Address"
                logger.info(f"[SESSION:{request.session_id}] New session created, returning greeting")
                
                return ChatbotResponse(
                    answer=greeting_message,
                    session_id=request.session_id,
                    status="new_session"
                )
            except Exception as e:
                logger.error(f"[SESSION:{request.session_id}] Error creating new session: {str(e)}")
                logger.debug(traceback.format_exc())
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to create session: {str(e)}"
                )
        
        # Session exists - process the message normally
        logger.info(f"[SESSION:{request.session_id}] Processing message for existing session")
        
        # Step 2: Get session history
        try:
            logger.debug(f"[SESSION:{request.session_id}] Retrieving session history...")
            history = session_agent.get_session_history(request.session_id)
            logger.debug(f"[SESSION:{request.session_id}] Retrieved {len(history) if history else 0} messages")
        except Exception as e:
            logger.error(f"[SESSION:{request.session_id}] Error retrieving session history: {str(e)}")
            logger.debug(traceback.format_exc())
            history = []
        
        # Step 3: Add user message to history
        try:
            logger.debug(f"[SESSION:{request.session_id}] Adding user message to history...")
            session_agent.add_message(request.session_id, "user", request.message)
        except Exception as e:
            logger.error(f"[SESSION:{request.session_id}] Error adding message to history: {str(e)}")
            logger.debug(traceback.format_exc())
        
        # Step 4: Detect buying intent and extract contact info
        intent = "general_query"
        has_contact_info = False
        extracted_info = {}
        try:
            logger.debug(f"[SESSION:{request.session_id}] Detecting intent...")
            intent_result = intent_agent.detect_intent(request.message)
            intent = intent_result.get("intent", "general_query")
            has_contact_info = intent_result.get("has_contact_info", False)
            extracted_info = intent_result.get("extracted", {})
            logger.info(f"[SESSION:{request.session_id}] Intent detected: {intent}, contact_info: {has_contact_info}")
        except Exception as e:
            logger.error(f"[SESSION:{request.session_id}] Error detecting intent: {str(e)}")
            logger.debug(traceback.format_exc())
            # Don't fail - continue with default intent
        
        # Step 5: Save customer info if contact details were extracted
        if has_contact_info and any(extracted_info.values()):
            try:
                logger.info(f"[SESSION:{request.session_id}] Saving customer info...")
                session_agent.save_customer_info(
                    session_id=request.session_id,
                    name=extracted_info.get("name") or "",
                    website=extracted_info.get("website") or "",
                    email=extracted_info.get("email") or "",
                    phone=extracted_info.get("phone") or ""
                )
                logger.info(f"[SESSION:{request.session_id}] Customer info saved")
            except Exception as e:
                logger.error(f"[SESSION:{request.session_id}] Error saving customer info: {str(e)}")
                logger.debug(traceback.format_exc())
                # Don't fail - continue processing
        
        # Step 6: Retrieve knowledge context using embedding and search
        knowledge_context = ""
        try:
            logger.debug(f"[SESSION:{request.session_id}] Retrieving knowledge base context...")
            context_docs = intent_agent.retrieve_context(request.message, limit=3)
            if context_docs:
                knowledge_context = "\n".join([doc.get('content', '') for doc in context_docs])
                logger.info(f"[SESSION:{request.session_id}] Retrieved {len(context_docs)} context documents")
            else:
                logger.debug(f"[SESSION:{request.session_id}] No context documents found")
        except Exception as e:
            logger.error(f"[SESSION:{request.session_id}] Error retrieving context: {str(e)}")
            logger.debug(traceback.format_exc())
            # Don't fail - continue with empty context
        
        # Step 7: Generate response using Briva
        try:
            logger.info(f"[SESSION:{request.session_id}] Generating response for intent: {intent}")
            response_dict = response_agent.generate_response(
                message=request.message,
                session_id=request.session_id,
                conversation_history=history,
                knowledge_context=knowledge_context if knowledge_context else None
            )
            
            if not response_dict or "answer" not in response_dict:
                logger.error(f"[SESSION:{request.session_id}] Invalid response format from response_agent")
                raise ValueError("Response agent returned invalid format")
            
            answer = response_dict.get("answer", "Unable to process your request")
            logger.info(f"[SESSION:{request.session_id}] Response generated successfully")
        except Exception as e:
            logger.error(f"[SESSION:{request.session_id}] Error generating response: {str(e)}")
            logger.debug(traceback.format_exc())
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate response: {str(e)}"
            )
        
        # Step 8: Add assistant response to history
        try:
            logger.debug(f"[SESSION:{request.session_id}] Adding assistant response to history...")
            session_agent.add_message(request.session_id, "assistant", answer)
        except Exception as e:
            logger.error(f"[SESSION:{request.session_id}] Error adding response to history: {str(e)}")
            logger.debug(traceback.format_exc())
            # Don't fail - continue
        
        logger.info(f"[SESSION:{request.session_id}] Request completed successfully")
        
        return ChatbotResponse(
            answer=answer,
            session_id=request.session_id,
            status="success"
        )
        
    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing chatbot message: {str(e)}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/session-history", tags=["Session"])
async def get_session_history(session_id: str):
    """
    Get conversation history for a session
    
    Args:
        session_id: Session identifier
        
    Returns:
        List of messages in the session
    """
    try:
        if not session_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="session_id is required"
            )
        
        logger.info(f"[SESSION:{session_id}] Retrieving session history...")
        history = session_agent.get_session_history(session_id)
        logger.info(f"[SESSION:{session_id}] Retrieved {len(history) if history else 0} messages")
        return {"session_id": session_id, "messages": history}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SESSION:{session_id}] Error retrieving session history: {str(e)}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving session history: {str(e)}"
        )


@app.post("/close-session", tags=["Session"])
async def close_session(session_id: str):
    """
    Close a session
    
    Args:
        session_id: Session identifier
        
    Returns:
        Status of the operation
    """
    try:
        if not session_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="session_id is required"
            )
        
        logger.info(f"[SESSION:{session_id}] Closing session...")
        success = session_agent.close_session(session_id)
        if success:
            logger.info(f"[SESSION:{session_id}] Session closed successfully")
            return {"status": "success", "session_id": session_id, "message": "Session closed"}
        else:
            logger.warning(f"[SESSION:{session_id}] Session not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SESSION:{session_id}] Error closing session: {str(e)}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error closing session: {str(e)}"
        )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Global exception handler for unhandled exceptions"""
    logger.error(f"Unhandled exception in {request.url.path}: {str(exc)}")
    logger.debug(f"Exception type: {type(exc).__name__}")
    logger.debug(traceback.format_exc())
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "error_type": type(exc).__name__}
    )


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    debug = os.getenv("DEBUG", "False").lower() == "true"
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )
