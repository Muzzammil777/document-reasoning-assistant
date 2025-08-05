#!/usr/bin/env python3
"""
Quick test to verify system components
"""

import sys
import os
sys.path.insert(0, '.')

def test_system():
    print("🧪 Testing HackRX API System Components...")
    print("=" * 50)
    
    try:
        # Test imports
        from app.main import app
        from app.file_utils import DocumentExtractor
        from app.llm_utils import DocumentReasoningLLM
        from app.vector_retriever import CloudDocumentRetriever
        print("✅ All modules imported successfully")
        
        # Test basic components
        extractor = DocumentExtractor()
        print("✅ DocumentExtractor initialized")
        
        # Test FastAPI app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        print("✅ FastAPI test client created")
        
        print("\n🎉 System is ready!")
        print("\nTo start the server, run:")
        print("   python start_server.py")
        print("\nOr use:")
        print("   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")
        
        print("\nOnce running, test with:")
        print("   python test_hackrx.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_system()
