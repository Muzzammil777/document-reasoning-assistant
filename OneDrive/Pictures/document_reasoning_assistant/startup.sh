#!/bin/bash

# Startup script for Azure Web App
echo "Starting Document Reasoning Assistant..."

# Install dependencies
echo "Installing Python dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt

# Start the application
echo "Starting FastAPI application..."
cd app
python -m uvicorn main:app --host 0.0.0.0 --port 8000
