import os
import json
import logging
import requests
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMProvider:
    """Abstract base class for LLM providers"""
    
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError

class TogetherAIProvider(LLMProvider):
    """Together AI provider for Qwen and other models"""
    
    def __init__(self, api_key: str, model_name: str):
        super().__init__(api_key, model_name)
        self.base_url = "https://api.together.xyz/inference"
    
    def generate_response(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.3) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "stop": ["<|im_end|>", "<|endoftext|>"]
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result["output"]["choices"][0]["text"].strip()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Together AI API request failed: {str(e)}")
            raise
        except KeyError as e:
            logger.error(f"Unexpected Together AI response format: {str(e)}")
            raise

class GroqProvider(LLMProvider):
    """Groq provider for fast inference"""
    
    def __init__(self, api_key: str, model_name: str = "llama3-70b-8192"):
        super().__init__(api_key, model_name)
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
    
    def generate_response(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.3) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Groq API request failed: {str(e)}")
            raise
        except KeyError as e:
            logger.error(f"Unexpected Groq response format: {str(e)}")
            raise

class FireworksProvider(LLMProvider):
    """Fireworks AI provider"""
    
    def __init__(self, api_key: str, model_name: str):
        super().__init__(api_key, model_name)
        self.base_url = "https://api.fireworks.ai/inference/v1/completions"
    
    def generate_response(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.3) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["text"].strip()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Fireworks AI API request failed: {str(e)}")
            raise
        except KeyError as e:
            logger.error(f"Unexpected Fireworks response format: {str(e)}")
            raise

class DocumentReasoningLLM:
    """Main LLM class for document reasoning tasks"""
    
    def __init__(self):
        self.provider_name = os.getenv("LLM_PROVIDER", "together").lower()
        self.model_name = os.getenv("MODEL_NAME", "Qwen/Qwen1.5-14B-Chat")
        self.provider = self._initialize_provider()
        
        logger.info(f"Initialized LLM with provider: {self.provider_name}, model: {self.model_name}")
    
    def _initialize_provider(self) -> LLMProvider:
        """Initialize the appropriate LLM provider"""
        if self.provider_name == "together":
            api_key = os.getenv("TOGETHER_API_KEY")
            if not api_key:
                raise ValueError("TOGETHER_API_KEY not found in environment")
            return TogetherAIProvider(api_key, self.model_name)
            
        elif self.provider_name == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in environment")
            return GroqProvider(api_key, self.model_name)
            
        elif self.provider_name == "fireworks":
            api_key = os.getenv("FIREWORKS_API_KEY")
            if not api_key:
                raise ValueError("FIREWORKS_API_KEY not found in environment")
            return FireworksProvider(api_key, self.model_name)
            
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider_name}")
    
    def create_reasoning_prompt(self, query: str, relevant_chunks: List[Dict]) -> str:
        """Create a structured prompt for document reasoning"""
        
        # Format relevant chunks
        context_sections = []
        for i, chunk in enumerate(relevant_chunks[:5], 1):  # Limit to top 5
            clause_id = chunk.get("clause_id", f"Section {i}")
            text = chunk["text"][:1000]  # Truncate if too long
            
            context_sections.append(f"""### {clause_id}
{text}
""")
        
        context = "\n".join(context_sections)
        
        prompt = f"""You are an expert document analysis assistant. Your task is to analyze documents and provide direct, concise responses to user queries.

## CONTEXT FROM DOCUMENT:
{context}

## USER QUERY:
{query}

## INSTRUCTIONS:
Analyze the provided document sections and answer the user's query. You must respond with a valid JSON object in exactly this format:

{{
  "direct_answer": "A concise, direct answer to the user's question (e.g., 'Yes, according to the document the policy covers paralysis' or 'No, the scheme does not cover this condition')",
  "decision": "Approved" | "Denied" | "Uncertain",
  "justification": "Clear reasoning based on the document analysis",
  "referenced_clauses": [
    {{
      "clause_id": "section identifier from document",
      "text": "relevant excerpt from the clause",
      "reasoning": "why this clause is relevant to the decision"
    }}
  ],
  "additional_info": "Any additional relevant information, context, or conditions that the user should know about"
}}

## DECISION CRITERIA:
- **Approved**: The document clearly supports the user's request/claim
- **Denied**: The document explicitly prohibits or excludes the request/claim
- **Uncertain**: The document is ambiguous, lacks specific coverage details, or requires additional information

## REQUIREMENTS:
1. The direct_answer should be conversational and directly address the user's question
2. Base your decision ONLY on the provided document sections
3. Quote relevant text excerpts in referenced_clauses
4. Provide clear reasoning for each referenced clause
5. If multiple clauses are relevant, include up to 3 most important ones
6. Be precise and factual in your justification
7. Include any relevant conditions, limitations, or exceptions in additional_info
8. Return ONLY the JSON object, no additional text

## RESPONSE:"""
        
        return prompt
    
    def analyze_document_query(self, query: str, relevant_chunks: List[Dict]) -> Dict[str, Any]:
        """Analyze query against document chunks and return structured response"""
        
        if not relevant_chunks:
            return {
                "direct_answer": "I couldn't find relevant information in the document to answer your question.",
                "decision": "Uncertain",
                "justification": "No relevant information found in the document to answer this query.",
                "referenced_clauses": [],
                "additional_info": "Please ensure your question is related to the content of the uploaded document."
            }
        
        # Create prompt
        prompt = self.create_reasoning_prompt(query, relevant_chunks)
        
        logger.info(f"Sending query to LLM: {query[:100]}...")
        
        try:
            # Get response from LLM
            raw_response = self.provider.generate_response(
                prompt,
                max_tokens=2000,
                temperature=0.2  # Lower temperature for more consistent JSON output
            )
            
            logger.info(f"Received LLM response: {raw_response[:200]}...")
            
            # Try to parse JSON response
            try:
                # Clean the response - remove any markdown formatting
                cleaned_response = self._clean_json_response(raw_response)
                response_data = json.loads(cleaned_response)
                
                # Validate response structure
                if self._validate_response_structure(response_data):
                    return response_data
                else:
                    logger.warning("Invalid response structure from LLM")
                    return self._create_fallback_response(query, relevant_chunks)
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM JSON response: {str(e)}")
                logger.error(f"Raw response: {raw_response}")
                return self._create_fallback_response(query, relevant_chunks, raw_response)
        
        except Exception as e:
            logger.error(f"LLM API call failed: {str(e)}")
            return self._create_error_response(str(e))
    
    def _clean_json_response(self, response: str) -> str:
        """Clean LLM response to extract valid JSON"""
        # Remove markdown code blocks if present
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        
        # Find the first { and last }
        start = response.find('{')
        end = response.rfind('}') + 1
        
        if start != -1 and end != 0:
            return response[start:end]
        
        return response
    
    def _validate_response_structure(self, response: Dict) -> bool:
        """Validate that response has required structure"""
        required_keys = ["direct_answer", "decision", "justification", "referenced_clauses", "additional_info"]
        
        if not all(key in response for key in required_keys):
            return False
        
        # Validate decision value
        valid_decisions = ["Approved", "Denied", "Uncertain"]
        if response["decision"] not in valid_decisions:
            return False
        
        # Validate referenced_clauses is a list
        if not isinstance(response["referenced_clauses"], list):
            return False
        
        # Validate each clause has required keys
        for clause in response["referenced_clauses"]:
            if not isinstance(clause, dict):
                return False
            clause_keys = ["clause_id", "text", "reasoning"]
            if not all(key in clause for key in clause_keys):
                return False
        
        return True
    
    def _create_fallback_response(self, query: str, chunks: List[Dict], raw_response: str = None) -> Dict[str, Any]:
        """Create fallback response when LLM fails to provide valid JSON"""
        referenced_clauses = []
        
        for chunk in chunks[:3]:  # Take top 3 chunks
            clause = {
                "clause_id": chunk.get("clause_id", chunk.get("chunk_id", "Unknown")),
                "text": chunk["text"][:300] + "..." if len(chunk["text"]) > 300 else chunk["text"],
                "reasoning": f"Relevant content with similarity score: {chunk.get('similarity_score', 0):.3f}"
            }
            referenced_clauses.append(clause)
        
        justification = "Unable to provide definitive analysis due to processing error. "
        if raw_response:
            justification += "Please review the referenced clauses for relevant information."
        else:
            justification += "The document contains relevant information but requires manual review."
        
        return {
            "direct_answer": "I'm unable to provide a definitive answer due to a processing error.",
            "decision": "Uncertain",
            "justification": justification,
            "referenced_clauses": referenced_clauses,
            "additional_info": "Please try rephrasing your question or contact support if the issue persists."
        }
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create error response when LLM is completely unavailable"""
        return {
            "direct_answer": "I'm unable to process your question due to a system error.",
            "decision": "Uncertain",
            "justification": f"Unable to process query due to system error: {error_message}",
            "referenced_clauses": [],
            "additional_info": "Please try again later or contact support if the issue persists."
        }
