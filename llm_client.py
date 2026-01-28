"""
LLM Client - Connects to LM Studio
This file handles all communication with the local AI model
"""

import requests
import yaml
from typing import Dict, List, Optional
import json

class LMStudioClient:
    """
    Client for connecting to LM Studio's local API server.
    LM Studio provides an OpenAI-compatible API.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the LM Studio client.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration from YAML file
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Extract LLM settings
        self.base_url = self.config['llm']['base_url']
        self.temperature = self.config['llm']['temperature']
        self.max_tokens = self.config['llm']['max_tokens']
        
        # We'll get the actual model name from LM Studio
        self.model = None
        
        # Test connection and get model info
        self._test_connection()
    
    def _test_connection(self):
        """
        Test if LM Studio server is running and get loaded model.
        Raises an error if connection fails.
        """
        try:
            # Try to get list of available models
            response = requests.get(f"{self.base_url}/models", timeout=5)
            
            if response.status_code == 200:
                models_data = response.json()
                
                # Extract model IDs from response
                if 'data' in models_data and len(models_data['data']) > 0:
                    # Use the first available model
                    self.model = models_data['data'][0]['id']
                    print(f"✓ Successfully connected to LM Studio")
                    print(f"✓ Using model: {self.model}")
                else:
                    print("⚠ Warning: LM Studio is running but no model is loaded!")
                    print("\nPlease load a model in LM Studio:")
                    print("1. Open LM Studio")
                    print("2. Go to 'Search' and download a model (e.g., Mistral-7B)")
                    print("3. Go to 'Local Server' tab")
                    print("4. Select the model from dropdown")
                    print("5. Click 'Start Server'")
                    # Set a default model name anyway
                    self.model = "local-model"
            else:
                print(f"⚠ Warning: LM Studio responded with status {response.status_code}")
                self.model = "local-model"
                
        except requests.exceptions.ConnectionError:
            print("\n❌ ERROR: Cannot connect to LM Studio!")
            print("Please make sure:")
            print("1. LM Studio is installed and running")
            print("2. A model is loaded")
            print("3. Local server is started (look for 'Local Server' tab)")
            print("4. Server is running on http://localhost:1234")
            print("\nStarting LM Studio server:")
            print("- Open LM Studio")
            print("- Click 'Local Server' tab (↔️ icon)")
            print("- Select a model from the dropdown")
            print("- Click 'Start Server' button")
            raise ConnectionError("LM Studio is not running")
        except Exception as e:
            print(f"❌ Unexpected error testing LM Studio connection: {e}")
            raise
    
    def generate_response(
        self, 
        prompt: str, 
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The user's question or prompt
            system_message: Optional system message to guide the AI's behavior
            temperature: Override default temperature (0=focused, 1=creative)
            max_tokens: Override default max response length
            
        Returns:
            The AI's response as a string
        """
        # Use defaults if not specified
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        # Build messages array (OpenAI chat format)
        messages = []
        
        # Add system message if provided (tells AI how to behave)
        if system_message:
            messages.append({
                "role": "system",
                "content": system_message
            })
        
        # Add user prompt
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Prepare request payload - simplified for LM Studio
        payload = {
            "messages": messages,
            "temperature": temp,
            "max_tokens": tokens,
            "stream": False  # Get complete response at once
        }
        
        # Only include model if we have one
        if self.model:
            payload["model"] = self.model
        
        try:
            # Send request to LM Studio
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120  # 2 minute timeout for slow responses
            )
            
            # Debug: print response if there's an error
            if response.status_code != 200:
                print(f"\n❌ LM Studio Error (Status {response.status_code}):")
                print(f"Response: {response.text[:500]}")
                return f"Error: LM Studio returned status {response.status_code}. Please check that a model is loaded and the server is running properly."
            
            # Extract the AI's response from JSON
            result = response.json()
            
            # Handle different response formats
            if 'choices' in result and len(result['choices']) > 0:
                ai_response = result['choices'][0]['message']['content']
                return ai_response.strip()
            else:
                return "Error: Unexpected response format from LM Studio"
            
        except requests.exceptions.Timeout:
            return "Error: LM Studio took too long to respond. Try a simpler query or check if the model is too large for your system."
        except requests.exceptions.ConnectionError:
            return "Error: Lost connection to LM Studio. Please make sure the server is still running."
        except requests.exceptions.RequestException as e:
            return f"Error communicating with LM Studio: {str(e)}"
        except json.JSONDecodeError:
            return "Error: Could not parse LM Studio response. The model may not be loaded correctly."
        except KeyError as e:
            return f"Error: Unexpected response format from LM Studio (missing key: {e})"
        except Exception as e:
            return f"Unexpected error: {str(e)}"
    
    def generate_with_context(
        self, 
        query: str, 
        context: str,
        system_message: Optional[str] = None
    ) -> str:
        """
        Generate a response using retrieved context (for RAG).
        
        This is the key method for RAG - we provide relevant data as context,
        then ask the question.
        
        Args:
            query: The user's question
            context: Relevant information retrieved from vector store
            system_message: Optional instructions for the AI
            
        Returns:
            AI's answer based on the provided context
        """
        # Default system message for manufacturing domain
        if system_message is None:
            system_message = """You are an AI assistant for manufacturing operations.
You help engineers and operators analyze production data, investigate faults, 
and understand machine performance.

IMPORTANT RULES:
1. Base your answers ONLY on the provided context data
2. If the context doesn't contain enough information, say so
3. Always cite specific numbers and timestamps from the data
4. Use clear, professional language
5. If asked about faults, explain the fault code meaning
6. Format timestamps in a readable way
7. Be concise but thorough"""
        
        # Combine context and query into a single prompt
        full_prompt = f"""CONTEXT (Production Data):
{context}

QUESTION:
{query}

Please provide a detailed answer based on the context above."""
        
        return self.generate_response(full_prompt, system_message)
    
    def is_available(self) -> bool:
        """
        Check if LM Studio is currently available.
        
        Returns:
            True if LM Studio is responding, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/models", timeout=3)
            return response.status_code == 200
        except:
            return False


# Test the client if run directly
if __name__ == "__main__":
    print("Testing LM Studio Client...")
    print("-" * 50)
    
    try:
        # Initialize client
        client = LMStudioClient()
        
        # Check if client initialized properly
        if client.model is None:
            print("\n❌ No model loaded in LM Studio!")
            print("Please load a model before testing.")
            exit(1)
        
        # Test basic response
        print("\n1. Testing basic generation:")
        print("   Question: 'What is OEE in manufacturing?'")
        response = client.generate_response(
            "Explain what OEE means in manufacturing in 2-3 sentences.",
            system_message="You are a helpful manufacturing expert. Be concise."
        )
        print(f"   Response: {response}")
        
        # Test with context (simulating RAG)
        print("\n2. Testing with context (RAG simulation):")
        test_context = """
Machine: MC001
Timestamp: 2026-01-19 14:30:00
Speed: 85 boards/min
OEE: 78%
Fault: E002 (Material Jam)
Temperature: 72°C
        """
        print("   Question: 'What fault occurred on MC001?'")
        response = client.generate_with_context(
            "What fault occurred on MC001 and what was the machine doing?",
            test_context
        )
        print(f"   Response: {response}")
        
        print("\n" + "="*50)
        print("✓ LM Studio client is working correctly!")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure LM Studio is running")
        print("2. Make sure a model is loaded and selected")
        print("3. Make sure 'Start Server' is clicked in the Local Server tab")
        print("4. Verify the server is on http://localhost:1234")