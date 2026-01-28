"""
RAG Pipeline - The brain of the system
Combines vector search with LLM to answer questions
"""

import yaml
from typing import Dict, Optional
from vector_store import ManufacturingVectorStore
from llm_client import LMStudioClient

class ManufacturingRAGPipeline:
    """
    Complete RAG pipeline for manufacturing data queries.
    
    How it works:
    1. User asks a question
    2. Find relevant data chunks from vector store
    3. Send question + context to LLM
    4. Return AI's answer
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the RAG pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        print("Initializing Manufacturing RAG Pipeline...")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        print("Loading vector store...")
        self.vector_store = ManufacturingVectorStore(config_path)
        self.vector_store.load_existing()
        
        print("Connecting to LM Studio...")
        self.llm_client = LMStudioClient(config_path)
        
        # Get fault codes for reference
        self.fault_codes = self.config['manufacturing']['fault_codes']
        
        print("✓ RAG Pipeline ready!")
    
    def query(self, user_question: str, k_results: int = None, verbose: bool = False) -> Dict:
        """
        Answer a question about manufacturing data.
        
        This is the main method you'll use.
        
        Args:
            user_question: The user's question
            k_results: Number of context chunks to retrieve (uses default if None)
            verbose: If True, show the retrieved context
            
        Returns:
            Dictionary with:
            - question: The original question
            - answer: The AI's response
            - context: The retrieved context (if verbose=True)
            - sources: Number of sources used
        """
        if not user_question.strip():
            return {
                "question": user_question,
                "answer": "Please provide a valid question.",
                "sources": 0
            }
        
        # Step 1: Retrieve relevant context from vector store
        if verbose:
            print(f"\n🔍 Searching for relevant data...")
        
        context = self.vector_store.get_relevant_context(user_question, k=k_results)
        
        if not context.strip():
            return {
                "question": user_question,
                "answer": "I couldn't find any relevant data to answer your question.",
                "sources": 0
            }
        
        if verbose:
            print(f"✓ Retrieved context from vector store")
            print(f"\n📄 Context being sent to AI:")
            print("-" * 60)
            print(context[:500] + "..." if len(context) > 500 else context)
            print("-" * 60)
        
        # Step 2: Create system message with domain knowledge
        system_message = self._create_system_message()
        
        # Step 3: Generate answer using LLM
        if verbose:
            print(f"\n🤖 Generating answer with LM Studio...")
        
        answer = self.llm_client.generate_with_context(
            query=user_question,
            context=context,
            system_message=system_message
        )
        
        # Step 4: Format and return response
        result = {
            "question": user_question,
            "answer": answer,
            "sources": k_results or self.config['rag']['top_k_results']
        }
        
        if verbose:
            result["context"] = context
        
        return result
    
    def _create_system_message(self) -> str:
        """
        Create a comprehensive system message for the LLM.
        This teaches the AI about manufacturing domain.
        
        Returns:
            System message string
        """
        # Build fault codes reference
        fault_codes_text = "\n".join([
            f"  - {code}: {name}" 
            for code, name in self.fault_codes.items()
        ])
        
        system_message = f"""You are an expert AI assistant for manufacturing operations at a corrugator plant.

Your role is to help engineers and operators:
- Analyze machine performance data
- Investigate production issues and faults
- Understand operational patterns
- Make data-driven decisions

CRITICAL RULES:
1. Base ALL answers on the provided context data ONLY
2. If the context doesn't contain information to answer, say so clearly
3. Always cite specific data points (timestamps, machine IDs, values)
4. Use precise numbers from the data, not approximations
5. Be concise but thorough - engineers need clear, actionable information

MANUFACTURING KNOWLEDGE:

Fault Codes:
{fault_codes_text}

Key Metrics:
- OEE (Overall Equipment Effectiveness): Percentage combining availability, performance, and quality. Higher is better (>85% is excellent)
- Speed: Boards produced per minute. Typical range: 80-100 boards/min
- Temperature: Operating temperature in Celsius. Normal range: 60-75°C. Above 80°C is concerning
- Vibration: Measured in mm/s. Below 3.0 is normal, above 4.0 may indicate issues
- Feeder Status: RUNNING (normal), STOPPED (planned stop), FAULTED (problem)

When answering:
- Format timestamps as readable dates/times
- Explain fault codes using their names
- Compare values to normal ranges
- Provide context for why metrics matter
- If multiple machines are relevant, compare them
- Suggest root causes when investigating faults

Remember: You're helping busy engineers make quick decisions. Be accurate, clear, and helpful."""
        
        return system_message
    
    def batch_query(self, questions: list, verbose: bool = False) -> list:
        """
        Answer multiple questions efficiently.
        
        Args:
            questions: List of question strings
            verbose: Show detailed output
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        for i, question in enumerate(questions, 1):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Question {i}/{len(questions)}: {question}")
                print('='*60)
            
            result = self.query(question, verbose=verbose)
            results.append(result)
            
            if verbose:
                print(f"\n💡 Answer: {result['answer']}")
        
        return results
    
    def interactive_session(self):
        """
        Start an interactive Q&A session.
        Useful for testing the pipeline.
        """
        print("\n" + "="*60)
        print("Manufacturing RAG - Interactive Mode")
        print("="*60)
        print("Ask questions about machine performance, faults, and operations.")
        print("Type 'quit' or 'exit' to end the session.")
        print("Type 'verbose' to toggle detailed output.")
        print("="*60 + "\n")
        
        verbose = False
        
        while True:
            try:
                # Get user input
                question = input("Your question: ").strip()
                
                if not question:
                    continue
                
                # Check for commands
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if question.lower() == 'verbose':
                    verbose = not verbose
                    print(f"✓ Verbose mode: {'ON' if verbose else 'OFF'}\n")
                    continue
                
                # Process query
                result = self.query(question, verbose=verbose)
                
                # Display answer
                print(f"\n💡 Answer:")
                print("-" * 60)
                print(result['answer'])
                print("-" * 60)
                print(f"(Based on {result['sources']} data sources)\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")


# Test queries for validation
def run_test_queries():
    """
    Run a set of test queries to validate the RAG pipeline.
    """
    print("="*60)
    print("Testing RAG Pipeline with Sample Queries")
    print("="*60)
    
    # Initialize pipeline
    rag = ManufacturingRAGPipeline()
    
    # Define test queries (matching assessment requirements)
    test_queries = [
        # Performance Analysis
        "What was the average OEE for MC001?",
        "Which machine had the highest productivity?",
        "Show me the production speed trend for all machines",
        
        # Fault Investigation
        "What faults occurred on MC002?",
        "How many times did fault E002 occur across all machines?",
        "What caused downtime on any machine?",
        
        # Operational Insights
        "When did the temperature exceed 75°C?",
        "What patterns do you see in the production data?",
        "Compare the OEE performance of all three machines"
    ]
    
    # Run queries
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Test Query {i}/{len(test_queries)}")
        print('='*60)
        print(f"Q: {query}")
        print()
        
        result = rag.query(query, verbose=False)
        
        print(f"A: {result['answer']}")
        print(f"\n(Retrieved from {result['sources']} sources)")
    
    print("\n" + "="*60)
    print("✓ Test queries completed!")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run automated tests
        run_test_queries()
    elif len(sys.argv) > 1 and sys.argv[1] == "interactive":
        # Start interactive session
        rag = ManufacturingRAGPipeline()
        rag.interactive_session()
    else:
        print("RAG Pipeline Usage:")
        print("  python rag_pipeline.py test        - Run test queries")
        print("  python rag_pipeline.py interactive - Start interactive Q&A")
        print("\nExample:")
        print("  python rag_pipeline.py interactive")