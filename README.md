# 🤖 RAG Question Answering System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Google Gemini](https://img.shields.io/badge/Powered%20by-Google%20Gemini-4285F4.svg)](https://ai.google.dev/)

An intelligent **Retrieval-Augmented Generation (RAG)** system that answers questions about AI research papers with accurate, grounded responses and source citations.

![RAG System Demo](demo.gif)
*Upload research papers and get accurate answers with source citations*

---

## 🌟 Features

- **📄 PDF Support:** Upload multiple research papers in PDF format
- **🧠 Semantic Search:** Uses embeddings for intelligent document retrieval
- **💬 Grounded Answers:** Responses backed by actual paper content
- **🔗 Source Citations:** Every answer includes traceable references
- **⚡ Fast Retrieval:** Semantic search in milliseconds
- **🎨 Beautiful UI:** Clean, intuitive Streamlit interface
- **📊 Real-time Stats:** Track processed documents and queries

---

## 🚀 Live Demo

**Try it now:** [Live Demo Link](#) *(Coming soon)*

Or run locally in 2 minutes! ⬇️

---

## 📋 Table of Contents

- [How It Works](#how-it-works)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [System Architecture](#system-architecture)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Examples](#examples)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🔍 How It Works

This RAG system follows a 5-step pipeline:

```
1. 📥 Load Documents → 2. ✂️ Chunk Text → 3. 🧠 Generate Embeddings 
                ↓
4. 🔍 Semantic Search ← User Query
                ↓
5. 💬 Generate Answer (with citations)
```

### Technical Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Embeddings** | Google Gemini Embedding-001 | 768-dim semantic vectors |
| **LLM** | Google Gemini 2.5 Flash | Fast, accurate answer generation |
| **Frontend** | Streamlit | Interactive web interface |
| **Vector Search** | NumPy + Cosine Similarity | Semantic retrieval |
| **PDF Processing** | PyPDF2 | Extract text from papers |

---

## 💻 Installation

### Prerequisites

- Python 3.10 or higher
- Google AI API Key ([Get one free](https://aistudio.google.com/app/apikey))

### Option 1: pip install (Recommended)

```bash
# Clone the repository
git clone https://github.com/azeemkhalipha/rag-qa-system.git
cd rag-qa-system

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Docker (One-command setup)

```bash
docker-compose up
```

Then open `http://localhost:8501` in your browser!

### Option 3: Google Colab (No installation needed)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/azeemkhalipha/rag-qa-system/blob/main/RAG_Capstone_Colab.ipynb)

---

## 🎯 Quick Start

### 1. Get Your API Key

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click "Create API Key"
3. Copy your key

### 2. Run the Application

```bash
# Start the Streamlit app
streamlit run streamlit_app.py
```

### 3. Upload & Query

1. Enter your API key in the sidebar
2. Upload 3-4 research papers (PDF)
3. Click "Process Documents"
4. Ask questions!

**That's it!** 🎉

---

## 📖 Usage

### Web Interface

```bash
streamlit run streamlit_app.py
```

**Features:**
- Drag-and-drop PDF upload
- Real-time processing status
- Interactive Q&A interface
- View retrieved chunks
- Question history

### Python API

```python
from rag_system import RAGQuestionAnswering

# Initialize system
rag = RAGQuestionAnswering(api_key="your_api_key")

# Load documents
rag.load_documents([
    "paper1.pdf",
    "paper2.pdf",
    "paper3.pdf"
])

# Ask questions
result = rag.answer("What is the Transformer architecture?")

print(result['answer'])
print(result['sources'])
```

### Command Line

```bash
python cli.py --question "What is multi-head attention?" --papers papers/*.pdf
```

---

## 🏗️ System Architecture

### Document Processing Pipeline

```python
# 1. Load PDF
text = DocumentLoader.load_pdf("paper.pdf")

# 2. Chunk with overlap
chunks = TextChunker(size=1000, overlap=200).chunk(text)

# 3. Generate embeddings
embeddings = EmbeddingGenerator.batch_generate(chunks)

# 4. Store in vector DB
VectorStore.add(chunks, embeddings)
```

### Query Pipeline

```python
# 1. Embed query
query_vec = embedding_gen.generate(query)

# 2. Semantic search
chunks = vector_store.search(query_vec, top_k=5)

# 3. Generate answer with LLM
answer = llm.generate(query, context=chunks)
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Chunk size: 1000 chars** | Balances context and precision |
| **20% overlap** | Prevents information loss at boundaries |
| **Top-k: 5** | Optimal context without noise |
| **Temperature: 0.3** | Factual, deterministic responses |
| **Max tokens: 1024** | Complete answers without verbosity |

---

## 🔧 Configuration

### Environment Variables

```bash
# .env file
GOOGLE_API_KEY=your_api_key_here
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=5
TEMPERATURE=0.3
MAX_TOKENS=1024
```

### config.yaml

```yaml
embedding:
  model: "models/gemini-embedding-001"
  dimensions: 768

generation:
  model: "models/gemini-2.5-flash"
  temperature: 0.3
  max_tokens: 1024

retrieval:
  top_k: 5
  similarity_threshold: 0.5

chunking:
  size: 1000
  overlap: 200
```

---

## 📚 Examples

### Example 1: Simple Question

```python
Q: "What is a Transformer?"

A: "The Transformer is a neural network architecture based solely on 
attention mechanisms [Source 1]. It consists of an encoder and decoder, 
each with multiple layers containing multi-head self-attention and 
feed-forward networks [Source 1]."

Sources:
  • Attention_Is_All_You_Need (Chunk 15)
```

### Example 2: Complex Question

```python
Q: "How does multi-head attention differ from single-head attention?"

A: "Multi-head attention projects queries, keys, and values h times with 
different learned linear projections [Source 1]. This allows the model to 
jointly attend to information from different representation subspaces at 
different positions [Source 2], whereas single-head attention would be 
limited to a single representation [Source 1]."

Sources:
  • Attention_Is_All_You_Need (Chunk 15)
  • Attention_Is_All_You_Need (Chunk 18)
```

### Example 3: Insufficient Information

```python
Q: "What is the training cost of GPT-4?"

A: "I don't have enough information in the provided context to answer 
this question. The uploaded papers do not contain information about 
GPT-4's training cost."
```

---

## ⚠️ Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Scale** | Optimized for 3-4 papers (~500 chunks) | Use FAISS for 10k+ documents |
| **Rate limits** | 100 embeddings/min (free tier) | Upgrade to paid tier |
| **English only** | Cannot process multilingual papers | Use multilingual models |
| **No OCR** | Cannot read scanned PDFs | Add OCR preprocessing |
| **No reranking** | ~85% retrieval precision | Add cross-encoder reranking |

See [LIMITATIONS.md](LIMITATIONS.md) for details.

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_retrieval.py

# With coverage
pytest --cov=rag_system tests/
```

---

## 📊 Performance Benchmarks

| Metric | Value |
|--------|-------|
| **Document Loading** | ~2s per paper |
| **Chunking** | ~0.5s per paper |
| **Embedding Generation** | ~0.65s per chunk |
| **Semantic Search** | <10ms for 500 chunks |
| **Answer Generation** | 2-3s |
| **Total (first query)** | ~3 mins (3 papers, 300 chunks) |
| **Subsequent queries** | 2-3s |

---

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone and setup
git clone https://github.com/azeemkhalipha/rag-qa-system.git
cd rag-qa-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest
```

### Roadmap

- [ ] Add FAISS for scalable vector search
- [ ] Implement reranking with cross-encoder
- [ ] Support for more file formats (DOCX, TXT, HTML)
- [ ] Multilingual support
- [ ] Conversation memory
- [ ] Export Q&A to PDF/DOCX
- [ ] REST API endpoint
- [ ] Docker deployment


## 📞 Contact

**Azeem Khalipha**

- 💼 LinkedIn: https://www.linkedin.com/in/azeemkhalipha/
- 🐙 GitHub: https://github.com/azeemkhalipha

---

## 🙏 Acknowledgments

- **Google Gemini** for powerful embedding and generation models
- **Streamlit** for the amazing web framework
- **Inspiration:** Modern RAG systems like Perplexity.ai

---


<div align="center">

**If this project helped you, please consider giving it a ⭐!**

</div>
