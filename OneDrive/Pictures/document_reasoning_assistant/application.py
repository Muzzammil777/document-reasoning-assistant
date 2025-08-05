#!/usr/bin/env python3
"""
Application entry point for Azure Web App
"""
import sys
import os

# Add the app directory to Python path
app_dir = os.path.join(os.path.dirname(__file__), 'app')
sys.path.insert(0, app_dir)

# Import the FastAPI app
from main import app

# This is what Azure will look for
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
