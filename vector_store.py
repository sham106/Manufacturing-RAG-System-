"""
Vector Store - Creates searchable database of manufacturing data
Uses LM Studio embeddings via OpenAI-compatible API (NO PyTorch!)
"""

import yaml
import os
import warnings
import sys

# Disable ChromaDB telemetry to prevent errors
os.environ["CHROMA_ANONYMIZED_TELEMETRY"] = "False"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Suppress ChromaDB telemetry warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Monkey-patch ChromaDB telemetry to prevent errors
try:
    import chromadb.telemetry.events as chroma_telemetry
    # Replace the broken capture function with a no-op
    def noop_capture(*args, **kwargs):
        pass
    chroma_telemetry.capture = noop_capture
except:
    pass

# Suppress stderr for ChromaDB telemetry errors
class TelemetryFilter:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
    
    def write(self, text):
        if "Failed to send telemetry event" not in text:
            self.original_stderr.write(text)
    
    def flush(self):
        self.original_stderr.flush()

# Only filter if not already filtered
if not isinstance(sys.stderr, TelemetryFilter):
    sys.stderr = TelemetryFilter(sys.stderr)

from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
import chromadb
from chromadb.config import Settings
import requests
import json


class LMStudioEmbeddings(Embeddings):
    """
    Custom embedding class for LM Studio's OpenAI-compatible embedding API.
    Handles the correct request format that LM Studio expects.
    """
    
    def __init__(self, base_url: str = "http://localhost:1234/v1", 
                 api_key: str = "lm-studio",
                 model: str = "text-embedding-nomic-embed-text-v1.5"):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = []
        # Process in larger batches for better performance
        # LM Studio can handle 50-100 chunks per request efficiently
        batch_size = 50  # Increased from 10 for better throughput
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            
            # Show progress for large batches
            if len(texts) > 100:
                print(f"  Embedding batch {batch_num}/{total_batches} ({len(batch)} chunks)...", end='\r')
            
            batch_embeddings = self._embed_batch(batch)
            embeddings.extend(batch_embeddings)
        
        if len(texts) > 100:
            print()  # New line after progress indicator
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self._embed_batch([text])[0]
    
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts using LM Studio API."""
        url = f"{self.base_url}/embeddings"
        
        # LM Studio expects 'input' field - always send as array for consistency
        # Some LM Studio versions expect array even for single items
        payload = {
            "model": self.model,
            "input": texts  # Always send as array
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            
            if response.status_code != 200:
                error_text = response.text[:500]
                raise ValueError(
                    f"LM Studio API error (status {response.status_code}): {error_text}\n"
                    f"Make sure LM Studio is running with an embedding model loaded!"
                )
            
            result = response.json()
            
            # Handle response format - OpenAI-compatible format
            if isinstance(result, dict) and 'data' in result:
                # Standard OpenAI format: {"data": [{"embedding": [...], ...}, ...]}
                embeddings = []
                for item in result['data']:
                    if isinstance(item, dict) and 'embedding' in item:
                        embeddings.append(item['embedding'])
                    elif isinstance(item, list):
                        embeddings.append(item)
                    else:
                        raise ValueError(f"Unexpected item format in response: {item}")
                return embeddings
            elif isinstance(result, list):
                # Direct array format
                return result
            else:
                raise ValueError(f"Unexpected response format: {result}")
            
        except requests.exceptions.ConnectionError:
            raise ValueError(
                f"Could not connect to LM Studio at {url}.\n"
                "Make sure LM Studio is running with an embedding model loaded!"
            )
        except requests.exceptions.Timeout:
            raise ValueError("LM Studio embedding API timed out. Try reducing batch size.")
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error calling LM Studio embedding API: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response from LM Studio: {e}")


class ManufacturingVectorStore:
    """
    Manages vector storage and retrieval for manufacturing data.
    Uses LM Studio embeddings via OpenAI-compatible API - NO PyTorch!
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the vector store."""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Get settings
        self.persist_directory = self.config['vector_store']['persist_directory']
        self.collection_name = self.config['vector_store']['collection_name']
        self.chunk_size = self.config['rag']['chunk_size']
        self.chunk_overlap = self.config['rag']['chunk_overlap']
        self.top_k = self.config['rag']['top_k_results']
        
        print("Initializing ChromaDB with LM Studio embeddings...")
        print("(No PyTorch - using OpenAI-compatible API)")
        
        # Use custom LM Studio embeddings class that handles the correct API format
        # Make sure LM Studio is running with an embedding model loaded!
        self.embeddings = LMStudioEmbeddings(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",  # Dummy key for LM Studio
            model="text-embedding-nomic-embed-text-v1.5"  # Adjust to your LM Studio embedding model
        )
        
        print("✓ Embedding function ready (LM Studio via OpenAI API)")
        
        # Vector store
        self.vectorstore = None
        self.chroma_client = None
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def create_from_texts(self, texts: List[str], metadatas: List[Dict] = None):
        """Create vector store from text chunks."""
        if not texts:
            raise ValueError("Cannot create vector store from empty text list")
        
        print(f"\nCreating vector store from {len(texts)} documents...")
        
        # Convert to Document objects
        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {"source": f"chunk_{i}"}
            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)
        
        # Split documents
        print("Splitting documents into chunks...")
        split_docs = self.text_splitter.split_documents(documents)
        print(f"✓ Split into {len(split_docs)} chunks")
        
        # Create ChromaDB client
        print("Creating embeddings with ChromaDB + LM Studio...")
        
        self.chroma_client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Delete old collection if exists
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            print(f"✓ Deleted old collection '{self.collection_name}'")
        except:
            pass
        
        # Create LangChain wrapper - it will create the collection automatically
        # The embeddings will be handled by OpenAIEmbeddings (LM Studio)
        self.vectorstore = Chroma(
            client=self.chroma_client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings
        )
        
        # Add documents using LangChain wrapper (handles embeddings automatically)
        print(f"Adding {len(split_docs)} documents to collection...")
        print("(This will create embeddings via LM Studio - may take a few minutes)")
        texts = [doc.page_content for doc in split_docs]
        metadatas = [doc.metadata for doc in split_docs]
        ids = [f"doc_{i}" for i in range(len(split_docs))]
        
        # Add in batches for progress tracking
        # Note: LangChain's add_texts will call embed_documents internally
        batch_size = 100
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            batch_num = (i // batch_size) + 1
            print(f"  Processing batch {batch_num}/{total_batches} ({len(batch_texts)} chunks)...")
            
            self.vectorstore.add_texts(
                texts=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            
            print(f"    ✓ Batch {batch_num} completed")
        
        print(f"✓ Vector store created: {self.persist_directory}")
        print(f"  Total chunks: {len(split_docs)}")
    
    def load_existing(self):
        """Load existing vector store from disk."""
        if not os.path.exists(self.persist_directory):
            raise FileNotFoundError(
                f"No vector store at {self.persist_directory}. "
                "Run: python vector_store.py build"
            )
        
        print(f"Loading vector store from {self.persist_directory}...")
        
        # Create client
        self.chroma_client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get collection (check if it exists)
        try:
            collection = self.chroma_client.get_collection(name=self.collection_name)
            print(f"✓ Loaded collection: {self.collection_name}")
        except Exception as e:
            raise ValueError(f"Collection '{self.collection_name}' not found: {e}")
        
        # Create LangChain wrapper
        self.vectorstore = Chroma(
            client=self.chroma_client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings
        )
        
        print("✓ Vector store loaded successfully")
    
    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        """Find relevant documents."""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        k = k if k is not None else self.top_k
        results = self.vectorstore.similarity_search(query, k=k)
        return results
    
    def get_relevant_context(self, query: str, k: int = None) -> str:
        """Get context as string for LLM."""
        results = self.similarity_search(query, k=k)
        
        context_parts = []
        for i, doc in enumerate(results, 1):
            context_parts.append(f"--- Context {i} ---")
            context_parts.append(doc.page_content)
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def delete_collection(self):
        """Delete the vector store."""
        if os.path.exists(self.persist_directory):
            import shutil
            shutil.rmtree(self.persist_directory)
            print(f"✓ Deleted: {self.persist_directory}")
    
    def get_stats(self) -> Dict:
        """Get vector store statistics."""
        if self.vectorstore is None:
            return {"status": "not initialized"}
        
        collection = self.chroma_client.get_collection(name=self.collection_name)
        
        return {
            "status": "active",
            "collection_name": self.collection_name,
            "document_count": collection.count(),
            "persist_directory": self.persist_directory
        }


def build_vector_store_from_scratch():
    """Build vector store from manufacturing data."""
    from data_processor import ManufacturingDataProcessor
    
    print("=" * 60)
    print("Building Vector Store (LM Studio Embeddings)")
    print("=" * 60)
    
    # Load data
    print("\nStep 1: Loading manufacturing data...")
    processor = ManufacturingDataProcessor()
    processor.load_machine_data()
    
    # Create chunks
    print("\nStep 2: Creating text chunks...")
    text_chunks = processor.create_text_chunks(chunk_size_minutes=30)
    
    # Create metadata
    print("\nStep 3: Creating metadata...")
    metadatas = []
    for i, chunk in enumerate(text_chunks):
        machine_id = "unknown"
        if "MC001" in chunk:
            machine_id = "MC001"
        elif "MC002" in chunk:
            machine_id = "MC002"
        elif "MC003" in chunk:
            machine_id = "MC003"
        
        metadatas.append({
            "source": f"chunk_{i}",
            "machine_id": machine_id,
            "chunk_index": i
        })
    
    # Create vector store
    print("\nStep 4: Creating vector store...")
    vector_store = ManufacturingVectorStore()
    
    if os.path.exists(vector_store.persist_directory):
        print("Removing old vector store...")
        vector_store.delete_collection()
    
    vector_store.create_from_texts(text_chunks, metadatas)
    
    # Test
    print("\n" + "=" * 60)
    print("Testing Vector Store")
    print("=" * 60)
    
    test_queries = [
        "What faults occurred on MC002?",
        "Show me the OEE performance",
        "What was the temperature?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        results = vector_store.similarity_search(query, k=2)
        for i, doc in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(doc.page_content[:200] + "...")
    
    print("\n" + "=" * 60)
    print("✓ Vector store built successfully!")
    print("=" * 60)
    
    stats = vector_store.get_stats()
    print(f"\nStats:")
    print(f"  Documents: {stats['document_count']}")
    print(f"  Location: {stats['persist_directory']}")


def test_existing_store():
    """Test existing vector store."""
    print("=" * 60)
    print("Testing Existing Vector Store")
    print("=" * 60)
    
    vector_store = ManufacturingVectorStore()
    vector_store.load_existing()
    
    query = "What was the average OEE?"
    print(f"\nTest Query: {query}")
    print("-" * 40)
    
    context = vector_store.get_relevant_context(query, k=3)
    print("\nContext:")
    print(context[:400] + "...")
    
    stats = vector_store.get_stats()
    print(f"\n✓ Working! Documents: {stats['document_count']}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_vector_store_from_scratch()
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        test_existing_store()
    else:
        print("Usage:")
        print("  python vector_store.py build  - Build vector store")
        print("  python vector_store.py test   - Test vector store")
        print("\nFirst time? Run: python vector_store.py build")