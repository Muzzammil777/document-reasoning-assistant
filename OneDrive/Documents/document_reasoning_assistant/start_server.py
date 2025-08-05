#!/usr/bin/env python3
"""
Startup script for HackRX API server
"""

import uvicorn
import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("🚀 Starting HackRX API Server...")
    print("📍 URL: http://localhost:8000")
    print("🔧 Endpoints:")
    print("   - POST /hackrx/run")
    print("   - POST /api/v1/hackrx/run")
    print("   - GET /health")
    print("   - GET /docs (API documentation)")
    print("\n" + "="*50)
    print("Press Ctrl+C to stop the server")
    print("="*50 + "\n")
    
    try:
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)
