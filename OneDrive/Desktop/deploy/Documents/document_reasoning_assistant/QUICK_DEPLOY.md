# Quick Deployment Guide for HackRX API

## Frontend Links Removed ✅
The API has been cleaned up and all frontend-related routes and static file serving have been removed. The API now serves only the essential endpoints needed for the hackathon.

## Core API Endpoints
- `POST /hackrx/run` - Main hackathon endpoint
- `POST /api/v1/hackrx/run` - Alternative endpoint
- `GET /health` - Health check
- `GET /test` - System test
- `GET /` - API info

## Easiest Deployment Methods

### 1. Render (Recommended)
1. Go to [render.com](https://render.com) and create account
2. Connect your GitHub repository
3. Create a new Web Service
4. Set these environment variables in Render dashboard:
   ```
   GROQ_API_KEY=your_groq_key
   PINECONE_API_KEY=your_pinecone_key
   PINECONE_INDEX_NAME=document-reasoning
   PINECONE_NAMESPACE=default
   LLM_PROVIDER=groq
   MODEL_NAME=llama3-70b-8192
   EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200
   TOP_K_RESULTS=5
   ```
5. Use start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
6. Deploy!

### 2. Railway
1. Go to [railway.app](https://railway.app)
2. Connect GitHub repo
3. Railway will automatically detect the `railway.json` config
4. Set the same environment variables as above
5. Deploy!

### 3. Heroku
```bash
# Install Heroku CLI first
heroku create your-hackrx-api-name
heroku config:set GROQ_API_KEY=your_key
heroku config:set PINECONE_API_KEY=your_key
heroku config:set PINECONE_INDEX_NAME=document-reasoning
heroku config:set PINECONE_NAMESPACE=default
heroku config:set LLM_PROVIDER=groq
heroku config:set MODEL_NAME=llama3-70b-8192
heroku config:set EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
heroku config:set CHUNK_SIZE=1000
heroku config:set CHUNK_OVERLAP=200
heroku config:set TOP_K_RESULTS=5
git push heroku main
```

### 4. Docker (if needed)
```bash
# Build and run locally
docker build -t hackrx-api .
docker run -p 8000:8000 --env-file .env hackrx-api
```

## Required Environment Variables
Create `.env` file with:
```
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=document-reasoning
PINECONE_NAMESPACE=default
LLM_PROVIDER=groq
MODEL_NAME=llama3-70b-8192
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5
```

## Test Your Deployment
```bash
curl -X POST "https://your-deployed-url.com/hackrx/run" \
     -H "Authorization: Bearer 6be388e87eae07a6e1ee672992bc2a22f207bbc7ff7e043758105f7d1fa45ffd" \
     -H "Content-Type: application/json" \
     -d '{
       "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
       "questions": ["What is the grace period for premium payment?"]
     }'
```

## Most Recommended: Render
Render is the easiest and most reliable for this hackathon deployment because:
- Automatic HTTPS
- Simple environment variable management
- Good free tier
- Reliable uptime
- Easy GitHub integration

Just push your code to GitHub, connect to Render, set the environment variables, and you're live!
