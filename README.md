# Manufacturing RAG System - Technical Assessment

**Candidate:** Shambach Simiyu 
**Date:** January 28, 2026  
**Implementation:** Python with LangChain + ChromaDB + LM Studio

## Executive Summary

This project implements a production-ready RAG (Retrieval-Augmented Generation) system for manufacturing operations, combining real-time data analysis with predictive inventory forecasting. The system processes 7 days of production data across 3 corrugator machines and provides natural language query capabilities for operational insights.

---

## Features

✅ **Machine Performance Analysis** - Query OEE metrics, production speeds, and efficiency patterns  
✅ **Fault Investigation** - Analyze downtime events, fault frequencies, and root causes  
✅ **Trend Analysis** - Identify production patterns and anomalies  
✅ **Inventory Forecasting** - Predict paper consumption with confidence intervals  
✅ **Purchase Recommendations** - Generate data-driven ordering suggestions with urgency levels

---

## Quick Start

```bash
# 1. Setup
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 2. Start LM Studio (download from lmstudio.ai)
# - Load a model (e.g., Mistral-7B)
# - Start server on localhost:1234

# 3. Generate data and build vector store
python data_generator.py
python vector_store.py build

# 4. Run the assistant
python main.py
```

---

## Architecture

```
User Query → Query Router → {
    Machine Data → Vector Store → LLM → Answer
    Forecasting → Statistical Model → Prediction
    Purchase → Forecasting + Rules → Recommendation
}
```

### Components

| Component | Purpose | Technology |
|-----------|---------|------------|
| **main.py** | Interactive interface | Python CLI with colorama |
| **rag_pipeline.py** | Query orchestration | LangChain RAG |
| **vector_store.py** | Document retrieval | ChromaDB (default embeddings) |
| **llm_client.py** | LLM communication | OpenAI-compatible API |
| **data_processor.py** | Data transformation | Pandas |
| **forecasting_module.py** | Time-series prediction | Statsmodels (Exponential Smoothing) |

---

## Installation Guide

### Prerequisites

- **Python 3.8+**
- **LM Studio** (https://lmstudio.ai/)
- **8GB RAM minimum**

### Step-by-Step Setup

**1. Create Virtual Environment**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
python -m pip install --upgrade pip
```

**2. Install Dependencies**
```bash
pip install -r requirements.txt
```

**Dependencies:**
- langchain==0.1.20
- chromadb==0.4.22 (with built-in embeddings)
- pandas, numpy (data processing)
- statsmodels, scikit-learn (forecasting)
- openai (LM Studio client)
- pyyaml, colorama, prompt-toolkit

**3. Setup LM Studio**
- Download from https://lmstudio.ai/
- Install and open application
- Download model: Search for "Mistral-7B-Instruct" or similar
- Go to "Local Server" tab
- Select model from dropdown
- Click "Start Server" (should show port 1234)

**4. Generate Data**
```bash
python data_generator.py
```
Creates `manufacturing_data.csv` with 6,048 records (7 days × 3 machines × 5-min intervals)

**5. Build Vector Store**
```bash
python vector_store.py build
```
Creates `chroma_db/` with 1,008 searchable document chunks

---

## Usage

### Interactive Chat Mode

```bash
python main.py
```

### Sample Queries

**Performance Analysis:**
```
What was the average OEE for MC001?
Which machine had the highest productivity this week?
Show me production speed trends for all machines
```

**Fault Investigation:**
```
What faults occurred on MC002?
How many times did fault E002 (Material Jam) occur?
What caused downtime between 14:00-16:00 on January 20th?
When did temperature exceed 75°C?
```

**Operational Insights:**
```
Compare OEE performance of all three machines
What patterns do you see in the production data?
Show correlation between temperature and faults
```

**Inventory Forecasting:**
```
Forecast for Flute 1700mm
What's the predicted consumption for next 3 months?
Show consumption summary
```

**Purchase Planning:**
```
What should we order?
Show purchase recommendations
Which paper types need urgent restocking?
```

### Testing Individual Components

```bash
python llm_client.py          # Test LM Studio connection
python data_processor.py      # Test data loading
python vector_store.py test   # Test retrieval
python forecasting_module.py  # Test predictions
python rag_pipeline.py test   # Test full RAG flow
```

---

## Configuration

Edit `config.yaml` to customize:

### LLM Settings
```yaml
llm:
  base_url: "http://localhost:1234/v1"
  temperature: 0.1  # Lower = more focused
  max_tokens: 1000
```

### Vector Store
```yaml
rag:
  chunk_size: 500        # Characters per chunk
  chunk_overlap: 50      # Overlap for context
  top_k_results: 5       # Documents to retrieve
```

### Forecasting
```yaml
forecasting:
  forecast_periods: 3      # Months ahead
  confidence_level: 0.95   # 95% confidence
  safety_factor: 1.2       # 20% buffer stock
  lead_time_days: 30       # Supplier lead time
```

---

## Technical Implementation

### Data Processing Pipeline

1. **Load CSV** → Parse timestamps, machines, metrics
2. **Chunk Creation** → Group by 30-min windows
3. **Text Generation** → Convert numbers to narrative descriptions
4. **Metadata Enrichment** → Add machine IDs, time periods

Example output:
```
=== MC001 Performance Report ===
Date: Monday, January 20, 2024
Time Period: 2024-01-20 14:00

Status: OPERATIONAL
Production Speed: 87.3 boards per minute
Overall Equipment Effectiveness (OEE): 81.2%
Feeder Status: RUNNING

Operating Conditions:
- Temperature: 71.5°C
- Vibration: 2.1 mm/s

✓ No faults detected in this period
```

### Vector Store Design

**Embeddings:** ChromaDB default (ONNX-based, no PyTorch)
- **Why:** Zero dependency issues on Windows
- **Performance:** Adequate for 1,000 documents
- **Alternative:** Can use HuggingFace if needed

**Storage:** Persistent ChromaDB collection
- **Size:** ~1,008 chunks from 7 days of data
- **Retrieval:** Top-5 similarity search
- **Indexing:** Automatic by ChromaDB

### RAG Query Flow

```
1. User asks: "What faults occurred on MC002?"
2. Vector Store finds 5 most relevant chunks mentioning MC002
3. LLM receives context + original question
4. LLM generates answer citing specific data points
5. System returns formatted response
```

### Forecasting Methodology

**Model:** Exponential Smoothing (Holt-Winters)
- **Why:** Works well with limited data (8 historical points)
- **Alternatives:** ARIMA, Prophet (if more data available)

**Features:**
- Trend detection
- Confidence intervals (95% default)
- Safety stock calculation
- Urgency classification (CRITICAL/HIGH/MEDIUM/LOW)

**Limitations:**
- No seasonality (insufficient data)
- Assumes stable consumption patterns
- Requires at least 3 data points per paper type

---

## Sample Output

### Query: "What faults occurred on MC002?"

**Assistant Response:**
```
Based on the production data, MC002 experienced multiple faults:

1. Material Jam (E002) - January 20, 2024 at 21:00
   - Duration: Approximately 30 minutes
   - Impact: Production stopped, OEE dropped to 0%
   - Context: Machine was running at normal temperature (60.7°C)

2. Additional E002 occurrences throughout the period
   - Total count: 6 instances in analyzed timeframe
   - Pattern: Mostly during evening shift

The material jam fault (E002) was the primary issue for MC002,
suggesting potential problems with the feeder system or material quality.
```

### Query: "Forecast for Flute 1700mm"

**Assistant Response:**
```
📊 Forecast for Flute 1700mm:

Month 1 (2026-02): 17,342 kg (range: 13,100 - 21,584 kg)
Month 2 (2026-03): 18,127 kg (range: 13,885 - 22,369 kg)
Month 3 (2026-04): 18,912 kg (range: 14,670 - 23,154 kg)

Historical Average: 16,895 kg/month
Model: Exponential Smoothing
Confidence: 95%

Forecast shows INCREASING trend compared to historical average.
Recommended safety stock: 22,694 kg
```

---

## Design Decisions

### Why ChromaDB Default Embeddings?
- **Problem:** PyTorch DLL issues on Windows
- **Solution:** ChromaDB's built-in ONNX embeddings
- **Trade-off:** Slightly lower quality vs. sentence-transformers
- **Benefit:** Zero setup friction, works everywhere

### Why Exponential Smoothing?
- **Problem:** Only 8 months of consumption data
- **Solution:** Simple, interpretable statistical model
- **Alternative:** Prophet (needs 2+ years for seasonality)
- **Benefit:** Fast, reliable, explainable predictions

### Why LM Studio?
- **Benefit:** Free, local, no API costs
- **Benefit:** Privacy - data stays on machine
- **Benefit:** OpenAI-compatible API
- **Trade-off:** Slower than cloud APIs
- **Alternative:** Could use OpenAI/Anthropic API

---

## Performance Notes

### Response Times
- **Vector search:** ~50-100ms
- **LLM generation:** 2-5 seconds (depends on model size)
- **Forecasting:** <1 second
- **Total query time:** 3-6 seconds average

### Resource Usage
- **RAM:** ~2-4GB (depends on LM Studio model)
- **Disk:** ~500MB (vector store + models)
- **CPU:** Moderate during LLM inference

### Scalability Considerations
- **Current:** 1,000 documents, sub-second retrieval
- **Recommended max:** 100,000 documents (ChromaDB)
- **Scale beyond:** Consider Pinecone, Weaviate, or FAISS
- **Production:** Add caching, batch processing, API rate limits



## Challenges Encountered & Solutions

During development, several technical challenges were encountered and resolved. This section documents the issues and solutions for transparency and future reference.

### 1. PyTorch DLL Initialization Error (Windows)

**Challenge:**
```
OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed. 
Error loading "torch\lib\c10.dll"
```

**Root Cause:**
- Windows systems require Visual C++ Redistributables for PyTorch
- sentence-transformers package has PyTorch as a dependency
- Missing or incompatible C++ runtime libraries

**Attempted Solutions:**
1. Installing Visual C++ 2015-2022 Redistributables
2. Reinstalling PyTorch with CPU-only version
3. Using different PyTorch versions

**Final Solution:**
Switched to **ChromaDB's built-in default embeddings** which use ONNX Runtime instead of PyTorch:
```python
self.embeddings = chromadb.utils.embedding_functions.DefaultEmbeddingFunction()
```

**Impact:**
- ✅ Eliminated all PyTorch dependencies
- ✅ Zero DLL errors on Windows
- ✅ Faster installation process
- ⚠️ Slightly different embedding quality (acceptable trade-off)

**Lesson Learned:** For Windows deployments, avoid PyTorch dependencies when simpler alternatives exist.

---

### 2. LangChain Version Conflicts

**Challenge:**
```
ERROR: Cannot install langchain==0.1.0 and langchain-community==0.0.13
ResolutionImpossible: conflicting dependencies on langsmith
```

**Root Cause:**
- `langchain` 0.1.0 requires `langsmith<0.1.0`
- `langchain-community` 0.0.13 requires `langsmith<0.1.0`
- `langchain-core` (auto-installed) requires `langsmith>=0.1.0`
- Incompatible version constraints

**Solution:**
Updated to compatible versions:
```
langchain==0.1.20
langchain-community==0.0.38
langchain-core==0.1.52
```

**Impact:**
- ✅ All packages install without conflicts
- ✅ No breaking changes in API usage
- ✅ Stable dependency tree

**Lesson Learned:** Always specify compatible version ranges for all dependencies, including transitive ones.

---

### 3. Virtual Environment Recreation

**Challenge:**
Multiple package installation attempts created an inconsistent environment state with conflicting dependencies.

**Solution:**
Complete virtual environment recreation:
```bash
deactivate
rmdir /s venv
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Impact:**
- ✅ Clean dependency state
- ✅ Eliminated cached package conflicts
- ✅ Reproducible installation

**Lesson Learned:** When facing persistent dependency issues, starting fresh is often faster than debugging.

---

### 4. Limited Inventory Data

**Challenge:**
Only Flute 1700mm has consumption history (8 data points). Other paper types (Test liner, White top) have no historical data.

**Design Decision:**
Kept original data structure as provided in assessment, rather than generating synthetic data.

**Implementation:**
System gracefully handles missing data:
```python
if paper_key not in self.models:
    return {
        "error": f"No model trained for {paper_type} {dimension}mm",
        "available_types": list(self.models.keys())
    }
```

**Impact:**
- ✅ Forecasting works for Flute 1700mm (demonstrates capability)
- ✅ System handles missing data gracefully (production-ready behavior)
- ✅ Clear error messages guide users
- ⚠️ Limited demonstration of multi-type forecasting

**Lesson Learned:** Work with provided data; document limitations clearly; handle edge cases gracefully.

---

### 5. Time Investment

**Estimated vs. Actual:**
- Initial estimate: 15-20 hours
- Actual time: ~25-30 hours (including troubleshooting)

**Value Gained:**
- Deep understanding of Windows-specific deployment challenges
- Experience with multiple embedding approaches
- Better appreciation for dependency management
- Production-ready error handling patterns

---

### 6. LM Studio Learning Curve

**Challenge:**
First time using LM Studio for local LLM hosting.

**Learning Points:**
- Understanding server startup process
- Configuring OpenAI-compatible API
- Model selection and resource requirements
- Timeout handling for slower responses

**Solution:**
Created robust connection testing and clear error messages:
```python
def _test_connection(self):
    """Test if LM Studio is running with helpful error messages"""
    try:
        response = requests.get(f"{self.base_url}/models", timeout=5)
        # ... detailed error handling
    except requests.exceptions.ConnectionError:
        print("Please make sure LM Studio is running...")
```

**Impact:**
- ✅ Clear setup instructions in README
- ✅ Helpful error messages for users
- ✅ Smooth development experience once configured

---

## Key Takeaways

### What Went Well
1. ✅ RAG pipeline implementation was straightforward with LangChain
2. ✅ ChromaDB's default embeddings solved Windows compatibility issues
3. ✅ Exponential smoothing worked well with limited data
4. ✅ Modular architecture made debugging easier
5. ✅ Clear separation of concerns (RAG vs. forecasting)

### What Could Be Improved
1. ⚠️ Earlier identification of PyTorch issues would have saved time
2. ⚠️ Initial dependency testing on clean environment
3. ⚠️ More comprehensive error handling from the start
4. ⚠️ Better time estimation accounting for environment setup

### Production Readiness Improvements Made
1. ✅ Comprehensive error handling throughout
2. ✅ Graceful degradation for missing data
3. ✅ Clear user feedback and error messages
4. ✅ Extensive documentation
5. ✅ Windows compatibility without external dependencies

---

## Troubleshooting

### LM Studio Connection Failed
```bash
# Check if server is running
curl http://localhost:1234/v1/models

# If fails:
# 1. Open LM Studio
# 2. Go to "Local Server" tab
# 3. Click "Start Server"
```

### Vector Store Build Errors
```bash
# Delete and rebuild
python vector_store.py build

# If still fails, check:
# - manufacturing_data.csv exists
# - Sufficient disk space
```

### Import Errors
```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

### Slow LLM Responses
- Use smaller model (e.g., 7B instead of 13B)
- Reduce max_tokens in config.yaml
- Enable GPU acceleration if available

---

## Testing

### Manual Testing
Run sample queries in `main.py` and verify:
- ✅ Accurate metric retrieval (OEE, speed, faults)
- ✅ Correct fault code interpretations
- ✅ Reasonable forecasts with confidence intervals
- ✅ Purchase recommendations match stock levels

### Automated Testing
```bash
python rag_pipeline.py test  # Run 9 predefined queries
```

Expected: All queries return relevant, data-backed responses

---

## Submission Checklist

- [x] Complete source code
- [x] requirements.txt with all dependencies
- [x] README.md with setup instructions
- [x] config.yaml with sensible defaults
- [x] Sample data (manufacturing_data.csv)
- [x] Working vector store (can be rebuilt)
- [x] Example queries and outputs
- [x] Architecture documentation

---

## Acknowledgments

- **LangChain** - RAG framework
- **ChromaDB** - Vector database
- **LM Studio** - Local LLM hosting
- **Statsmodels** - Time-series forecasting

---

**Built with ❤️ for manufacturing operations optimization**
            **TO MORE CODING.............**
