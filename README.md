# Pathvancer Chatbot - Multi-User Agentic FastAPI

A production-ready FastAPI chatbot with multi-user session management, intent detection, and knowledge base integration powered by OpenRouter AI and Google Sheets.

## Features

- 🚀 FastAPI with async support
- 👥 Multi-user session management via Google Sheets
- 🧠 Intent detection with OpenRouter LLM
- 🔍 Knowledge base search with Supabase embeddings
- 📊 Customer info extraction and persistence
- 🔐 CORS-enabled for web applications
- 🐳 Docker optimized (<200MB)
- ✅ Health checks and monitoring

## Project Structure

```
pathvancer-chatbot/
├── agents/              # Agentic logic
│   ├── session_agent.py      # Session management
│   ├── intent_agent.py       # Intent detection & context retrieval
│   └── response_agent.py     # Response generation
├── models/              # Data schemas
│   └── schemas.py            # Pydantic models
├── services/            # External integrations
│   ├── google_sheets.py      # Google Sheets service
│   └── supabase_service.py   # Vector DB service
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── Dockerfile           # Multi-stage Docker build
├── render.yaml          # Render deployment config
├── .env.example         # Environment variables template
└── README.md            # This file
```

## Prerequisites

- Python 3.11+
- OpenRouter API key
- Google Service Account (for Google Sheets)
- Supabase project (for vector embeddings)
- Docker (for containerized deployment)

## Setup

### 1. Clone and Install

```bash
git clone <repository-url>
cd pathvancer-chatbot
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your actual values:
- `OPENROUTER_API_KEY` - Get from [OpenRouter](https://openrouter.io/)
- `SPREADSHEET_ID` - Google Sheet ID for session storage
- `SUPABASE_URL` & `SUPABASE_KEY` - From your Supabase project

### 3. Encode Google Service Account

Your Google Service Account JSON needs to be base64 encoded:

**On Linux/Mac:**
```bash
base64 -i service_account.json | tr -d '\n'
```

**On Windows PowerShell:**
```powershell
$file = [System.IO.File]::ReadAllBytes("service_account.json")
$encoded = [Convert]::ToBase64String($file)
$encoded
```

Copy the output and paste it as `GOOGLE_SERVICE_ACCOUNT_JSON` in your `.env` file.

## Local Development

### Run with Python

```bash
python main.py
```

The API will start at `http://localhost:8000`

### View API Docs

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Docker

### Build Image

```bash
docker build -t pathvancer-chatbot .
```

This creates an optimized image (~150MB) using multi-stage build.

### Run Container

```bash
docker run -p 8000:8000 \
  -e OPENROUTER_API_KEY=your_key \
  -e SPREADSHEET_ID=your_id \
  -e GOOGLE_SERVICE_ACCOUNT_JSON=your_base64_json \
  -e SUPABASE_URL=your_url \
  -e SUPABASE_KEY=your_key \
  pathvancer-chatbot
```

Or use `--env-file`:

```bash
docker run -p 8000:8000 --env-file .env pathvancer-chatbot
```

## API Endpoints

### `GET /`
Returns API metadata.

**Response:**
```json
{
  "name": "Pathvancer Chatbot",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "ok"
}
```

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2026-05-05T10:30:45.123456"
}
```

### `POST /chatbot-session`
Process user message and get response.

**Request:**
```json
{
  "message": "What services do you offer?",
  "session_id": "user-123-session-1",
  "user_id": "user-123"
}
```

**Response (New Session):**
```json
{
  "answer": "Hello! I'm Briva 👋 To help you best, please share: 1. Your Name 2. Company website 3. Phone Number 4. Email Address",
  "session_id": "user-123-session-1",
  "status": "new_session",
  "timestamp": "2026-05-05T10:30:45.123456"
}
```

**Response (Existing Session):**
```json
{
  "answer": "We offer comprehensive digital transformation services including...",
  "session_id": "user-123-session-1",
  "status": "success",
  "timestamp": "2026-05-05T10:30:46.654321"
}
```

### Session Management

**New Session Flow:**
1. User sends first message
2. System returns greeting asking for: Name, Company Website, Phone, Email
3. User provides information in follow-up messages
4. System extracts and saves customer info automatically

**Existing Session Flow:**
1. Intent detection on message
2. Automatic contact info extraction if present
3. Knowledge base search for context
4. AI-generated response with Briva persona
5. Session history updated

## Deployment

### Render.com (Recommended)

**Option 1: Automatic Deployment**

1. Push code to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com)
3. Click "New +" → "Web Service"
4. Connect your GitHub repository
5. Fill in settings:
   - **Name:** `pathvancer-chatbot`
   - **Region:** Choose closest to users
   - **Branch:** `main` (or your default branch)
   - **Runtime:** Docker
   - **Plan:** Free or Starter
6. Add Environment Variables:
   - `OPENROUTER_API_KEY`
   - `SPREADSHEET_ID`
   - `GOOGLE_SERVICE_ACCOUNT_JSON`
   - `SUPABASE_URL`
   - `SUPABASE_KEY`
7. Click "Deploy"

**Option 2: Using render.yaml (One-Click)**

1. Commit `render.yaml` to your repository
2. Go to [Render Dashboard](https://dashboard.render.com)
3. Click "New +" → "Blueprint"
4. Connect your GitHub repository
5. Select the repository and branch
6. Render auto-detects `render.yaml`
7. Add missing environment variables
8. Click "Deploy"

**Deployment Details:**
- Dockerfile detected automatically
- Runs on port 8000
- Auto-restart on failure
- Built-in health checks
- Free SSL/HTTPS included

### Verifying Deployment

After deployment, test the API:

```bash
curl https://your-render-app.onrender.com/health
```

Should return:
```json
{
  "status": "ok",
  "timestamp": "2026-05-05T..."
}
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes | API key for OpenRouter LLM |
| `SPREADSHEET_ID` | Yes | Google Sheet ID for session storage |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | Yes | Base64-encoded service account JSON |
| `SUPABASE_URL` | Yes | Supabase project URL |
| `SUPABASE_KEY` | Yes | Supabase anonymous key |
| `API_HOST` | No | Server host (default: `0.0.0.0`) |
| `API_PORT` | No | Server port (default: `8000`) |
| `DEBUG` | No | Debug mode (default: `False`) |

## Architecture

### Services Layer

**GoogleSheetsService**
- Session creation and lookup
- Conversation history storage
- Customer info persistence

**SupabaseService**
- Vector embeddings
- Knowledge base search
- Similarity matching

**OpenRouter API**
- Intent classification
- Response generation
- All LLM operations

### Agents Layer

**SessionAgent**
- Manages user sessions
- Maintains conversation history
- Extracts and saves customer info

**IntentAgent**
- Detects user intent (greeting, question, request, etc.)
- Extracts contact information (email, phone, website)
- Retrieves knowledge base context via embeddings

**ResponseAgent**
- Generates personalized responses as "Briva"
- Leverages conversation history and knowledge context
- Outputs structured JSON responses

## Performance

- **Response Time:** <2s (avg)
- **Container Size:** ~150MB (optimized slim image)
- **Startup Time:** <5s
- **Concurrent Sessions:** Unlimited

## Monitoring

The app includes health checks:

```bash
# Manual health check
curl http://localhost:8000/health

# Docker health check
docker inspect --format='{{.State.Health.Status}}' <container-id>
```

Logs are streamed to stdout for container orchestration integration.

## Troubleshooting

### "OPENROUTER_API_KEY not configured"
- Ensure `.env` file exists with `OPENROUTER_API_KEY=your_key`
- For Docker, pass via `--env-file .env` or `-e OPENROUTER_API_KEY=...`

### "Google Service Account authentication failed"
- Verify `GOOGLE_SERVICE_ACCOUNT_JSON` is properly base64 encoded
- Check service account has access to the Google Sheet
- Verify `SPREADSHEET_ID` is correct

### "Failed to create session"
- Check Google Sheets service account permissions
- Verify sheet has required columns

### Docker image too large
- Current size: ~150MB (within 200MB limit)
- Already using slim base and multi-stage build

## Support

For issues or questions:
1. Check logs: `docker logs <container-id>`
2. Verify all environment variables are set
3. Test health endpoint: `/health`
4. Review API docs at `/docs`

## License

Proprietary - PathVancer
