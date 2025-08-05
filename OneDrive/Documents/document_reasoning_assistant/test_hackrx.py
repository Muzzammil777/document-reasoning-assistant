#!/usr/bin/env python3
"""
Test script for HackRX API endpoint
"""

import requests
import json

# Test configuration
BASE_URL = "http://localhost:8000"
API_TOKEN = "6be388e87eae07a6e1ee672992bc2a22f207bbc7ff7e043758105f7d1fa45ffd"

def test_hackrx_endpoint():
    """Test the /hackrx/run endpoint"""
    
    # Test data from the hackathon specifications
    test_data = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?",
            "What is the waiting period for cataract surgery?",
            "Are the medical expenses for an organ donor covered under this policy?",
            "What is the No Claim Discount (NCD) offered in this policy?",
            "Is there a benefit for preventive health check-ups?",
            "How does the policy define a 'Hospital'?",
            "What is the extent of coverage for AYUSH treatments?",
            "Are there any sub-limits on room rent and ICU charges for Plan A?"
        ]
    }
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {API_TOKEN}"
    }
    
    print("Testing /hackrx/run endpoint...")
    print(f"URL: {BASE_URL}/hackrx/run")
    print(f"Document URL: {test_data['documents']}")
    print(f"Number of questions: {len(test_data['questions'])}")
    print("\n" + "="*50 + "\n")
    
    try:
        response = requests.post(
            f"{BASE_URL}/hackrx/run",
            headers=headers,
            json=test_data,
            timeout=120  # 2 minutes timeout
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("SUCCESS! Response received:")
            print(json.dumps(result, indent=2))
            
            # Validate response structure
            if "answers" in result and len(result["answers"]) == len(test_data["questions"]):
                print(f"\n✅ Response structure is correct!")
                print(f"✅ Received {len(result['answers'])} answers for {len(test_data['questions'])} questions")
                
                # Print each Q&A pair
                print("\n" + "="*50)
                print("QUESTIONS AND ANSWERS:")
                print("="*50)
                for i, (question, answer) in enumerate(zip(test_data["questions"], result["answers"]), 1):
                    print(f"\nQ{i}: {question}")
                    print(f"A{i}: {answer}")
                    print("-" * 50)
            else:
                print("❌ Response structure is incorrect!")
                
        else:
            print(f"❌ Request failed with status code: {response.status_code}")
            print(f"Error: {response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out (took longer than 2 minutes)")
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the server. Make sure it's running on localhost:8000")
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")

def test_api_v1_endpoint():
    """Test the /api/v1/hackrx/run endpoint"""
    
    print("\n" + "="*50)
    print("Testing /api/v1/hackrx/run endpoint...")
    
    # Simple test data
    test_data = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment?"
        ]
    }
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {API_TOKEN}"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/hackrx/run",
            headers=headers,
            json=test_data,
            timeout=60
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API v1 endpoint works correctly!")
            print(f"Answer: {result['answers'][0]}")
        else:
            print(f"❌ API v1 endpoint failed: {response.text}")
            
    except Exception as e:
        print(f"❌ API v1 endpoint error: {str(e)}")

def test_health_endpoint():
    """Test the health endpoint"""
    
    print("\n" + "="*50)
    print("Testing /health endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Health check passed!")
            print(f"System status: {health_data['status']}")
            
            components = health_data.get('components', {})
            for component, status in components.items():
                status_icon = "✅" if status else "❌"
                print(f"{status_icon} {component}: {status}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")

if __name__ == "__main__":
    print("HackRX API Test Suite")
    print("=" * 50)
    
    # Test health first
    test_health_endpoint()
    
    # Test main hackrx endpoint
    test_hackrx_endpoint()
    
    # Test API v1 endpoint
    test_api_v1_endpoint()
    
    print("\n" + "="*50)
    print("Testing complete!")
