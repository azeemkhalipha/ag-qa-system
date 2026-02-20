"""
RAG Question Answering System - Web Interface (Theme-Adaptive)
===============================================================
Interactive web application for querying AI research papers using RAG.
Now with automatic dark/light theme adaptation!

Author: Azeem Khalipha
GitHub: https://github.com/azeemkhalipha
"""

import streamlit as st
import numpy as np
from typing import List, Dict
import time
import os
from google import genai
from google.genai import types
from PyPDF2 import PdfReader
import re

# Page configuration
st.set_page_config(
    page_title="RAG QA System - AI Research Papers",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# FIXED: Theme-adaptive CSS that works in both light and dark modes
st.markdown("""
<style>
    /* Main header - uses Streamlit's default text color */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 2rem;
        opacity: 0.8;
    }
    
    /* Chunk display boxes - adapts to theme */
    .chunk-box {
        background-color: var(--secondary-background-color);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid var(--background-color);
    }
    
    /* Answer box - uses theme colors */
    .answer-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E88E5;
        background-color: var(--secondary-background-color);
        margin: 1rem 0;
    }
    
    /* Source citations - adapts to theme */
    .source-box {
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.3rem 0;
        background-color: var(--secondary-background-color);
        border-left: 3px solid #FF9800;
    }
    
    /* Metric cards - theme adaptive */
    .metric-card {
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        background-color: var(--secondary-background-color);
        margin: 0.5rem 0;
    }
    
    /* Feature highlights - theme colors */
    .feature-highlight {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: var(--secondary-background-color);
        text-align: center;
        height: 100%;
    }
    
    /* Progress indicator */
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
    
    /* Remove extra padding */
    .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# RAG SYSTEM COMPONENTS (Same as before)
# ============================================================================

class DocumentLoader:
    """Load and extract text from PDF documents"""
    
    @staticmethod
    def load_pdf(uploaded_file) -> str:
        """Extract text from uploaded PDF"""
        try:
            pdf_reader = PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error loading PDF: {str(e)}")
            return ""


class TextChunker:
    """Split documents into overlapping chunks"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, doc_name: str) -> List[Dict]:
        """Split text into overlapping chunks"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < self.chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append({
                        'id': f"{doc_name}_chunk_{chunk_id}",
                        'document': doc_name,
                        'text': current_chunk.strip(),
                        'chunk_num': chunk_id
                    })
                    chunk_id += 1
                    
                    words = current_chunk.split()
                    overlap_text = " ".join(words[-int(self.overlap/5):])
                    current_chunk = overlap_text + " " + sentence + " "
                else:
                    current_chunk = sentence + " "
        
        if current_chunk.strip():
            chunks.append({
                'id': f"{doc_name}_chunk_{chunk_id}",
                'document': doc_name,
                'text': current_chunk.strip(),
                'chunk_num': chunk_id
            })
        
        return chunks


class EmbeddingGenerator:
    """Generate embeddings with rate limiting"""
    
    def __init__(self, client, model_name: str = "models/gemini-embedding-001"):
        self.client = client
        self.model_name = model_name
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for single text"""
        result = self.client.models.embed_content(
            model=self.model_name,
            contents=text
        )
        return np.array(result.embeddings[0].values)
    
    def batch_generate(self, texts: List[str], progress_bar=None) -> List[np.ndarray]:
        """Generate embeddings with progress tracking"""
        embeddings = []
        total = len(texts)
        
        for i, text in enumerate(texts):
            embedding = self.generate_embedding(text)
            embeddings.append(embedding)
            
            if progress_bar:
                progress_bar.progress((i + 1) / total)
            
            # Rate limiting
            if i < total - 1:
                time.sleep(0.65)
        
        return embeddings


class VectorStore:
    """In-memory vector store with cosine similarity search"""
    
    def __init__(self):
        self.chunks = []
        self.embeddings = []
    
    def add_chunks(self, chunks: List[Dict], embeddings: List[np.ndarray]):
        """Add chunks and embeddings"""
        self.chunks.extend(chunks)
        self.embeddings.extend(embeddings)
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        """Search for most similar chunks"""
        similarities = []
        for chunk, embedding in zip(self.chunks, self.embeddings):
            similarity = self.cosine_similarity(query_embedding, embedding)
            similarities.append((chunk, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class AnswerGenerator:
    """Generate answers using LLM"""
    
    def __init__(self, client, model: str = "models/gemini-2.5-flash"):
        self.client = client
        self.model = model
    
    def create_prompt(self, query: str, chunks: List[Dict]) -> str:
        """Create grounded prompt"""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}: {chunk['document']}, Chunk {chunk['chunk_num']}]\n"
                f"{chunk['text']}\n"
                f"[Relevance: {chunk['relevance_score']:.3f}]\n"
            )
        
        context = "\n".join(context_parts)
        
        return f"""You are a helpful research assistant answering questions based on AI research papers.

TASK: Answer the following question using ONLY the information provided in the context below.

INSTRUCTIONS:
1. Base your answer ONLY on the provided context
2. Cite sources using the format [Source X]
3. If insufficient information, say "I don't have enough information in the provided context to answer this question."
4. Be concise but comprehensive
5. Do not use training data - only provided context

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
    
    def generate_answer(self, query: str, chunks: List[Dict]) -> Dict:
        """Generate answer with citations"""
        prompt = self.create_prompt(query, chunks)
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=1024,
            )
        )
        
        return {
            'question': query,
            'answer': response.text,
            'sources': [f"{c['document']} (Chunk {c['chunk_num']})" for c in chunks],
            'retrieved_chunks': chunks
        }


# ============================================================================
# STREAMLIT APP (Updated with better theme handling)
# ============================================================================

def initialize_session_state():
    """Initialize session state variables"""
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'embedding_gen' not in st.session_state:
        st.session_state.embedding_gen = None
    if 'answer_gen' not in st.session_state:
        st.session_state.answer_gen = None
    if 'all_chunks' not in st.session_state:
        st.session_state.all_chunks = []
    if 'qa_history' not in st.session_state:
        st.session_state.qa_history = []


def main():
    """Main Streamlit application"""
    
    initialize_session_state()
    
    # Header - using div for theme adaptation
    st.markdown('<div class="main-header">🤖 RAG Question Answering System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask questions about AI research papers with intelligent retrieval and grounded answers</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key Setup with better visibility
        with st.expander("🔑 Get Your Free API Key", expanded=False):
            st.markdown("""
            **Quick Setup (30 seconds):**
            
            1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
            2. Click "Create API Key"
            3. Click "Create API key in new project"
            4. Copy and paste below
            
            💡 **Free tier:** Generous limits for testing!
            """)
        
        api_key = st.text_input(
            "Google AI API Key",
            type="password",
            help="Your key is only used in this session and never stored",
            placeholder="Paste your API key here..."
        )
        
        if api_key:
            try:
                client = genai.Client(api_key=api_key)
                st.success("✅ API Key Valid")
            except Exception as e:
                st.error(f"❌ Invalid API Key: {str(e)[:50]}")
                st.stop()
        else:
            st.warning("👆 Please enter your API key to continue")
            st.info("Don't have one? Click the dropdown above for instructions!")
            st.stop()
        
        st.divider()
        
        # Document upload
        st.header("📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF research papers",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload 3-4 AI research papers"
        )
        
        # Process documents button
        if uploaded_files and st.button("🚀 Process Documents", type="primary", use_container_width=True):
            process_documents(uploaded_files, client)
        
        st.divider()
        
        # System stats
        if st.session_state.documents_loaded:
            st.header("📊 System Stats")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", len(uploaded_files))
                st.metric("Chunks", len(st.session_state.all_chunks))
            with col2:
                st.metric("Embeddings", len(st.session_state.all_chunks))
                st.metric("Questions", len(st.session_state.qa_history))
        
        st.divider()
        
        # About section
        st.header("ℹ️ About")
        st.markdown("""
        **RAG System** for research papers
        
        **How it works:**
        1. 📥 Upload PDFs
        2. ✂️ Split into chunks
        3. 🧠 Generate embeddings
        4. 🔍 Semantic search
        5. 💬 LLM answers
        
        **Models:**
        - Embeddings: `gemini-embedding-001`
        - Generation: `gemini-2.5-flash`
        """)
        
        # Footer
        st.divider()
        st.caption("Built with Streamlit & Google Gemini")
    
    # Main content area
    if not st.session_state.documents_loaded:
        # Welcome screen with theme-adaptive cards
        st.info("👈 Upload research papers in the sidebar to get started!", icon="📚")
        
        st.subheader("📝 Example Questions:")
        examples = [
            "What are the key components of the Transformer architecture?",
            "How does multi-head attention work?",
            "What is the purpose of positional encoding?",
            "Explain few-shot learning in GPT-3",
            "What are the main components of a RAG model?"
        ]
        
        for i, q in enumerate(examples, 1):
            st.markdown(f"**{i}.** *{q}*")
        
        st.divider()
        
        # Features showcase - using columns for better spacing
        st.subheader("✨ Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="feature-highlight">
            <h3>🎯 Accurate</h3>
            <p>Grounded answers with source citations</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-highlight">
            <h3>⚡ Fast</h3>
            <p>Semantic search in milliseconds</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="feature-highlight">
            <h3>🔗 Transparent</h3>
            <p>Full source attribution</p>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        # Question answering interface
        st.subheader("💬 Ask Questions")
        
        # Question input
        user_question = st.text_input(
            "Enter your question:",
            placeholder="e.g., What is the Transformer architecture?",
            key="question_input"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)
        with col2:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.qa_history = []
                st.rerun()
        
        if ask_button and user_question:
            answer_question(user_question, client)
        
        # Display QA history
        if st.session_state.qa_history:
            st.divider()
            st.subheader("📜 Question History")
            
            for i, qa in enumerate(reversed(st.session_state.qa_history), 1):
                with st.expander(f"**Q{len(st.session_state.qa_history) - i + 1}:** {qa['question']}", expanded=(i==1)):
                    # Answer - using theme-adaptive box
                    st.markdown(f'<div class="answer-box"><strong>Answer:</strong><br><br>{qa["answer"]}</div>', unsafe_allow_html=True)
                    
                    # Sources
                    st.markdown("**📚 Sources:**")
                    for source in qa['sources']:
                        st.markdown(f'<div class="source-box">📄 {source}</div>', unsafe_allow_html=True)
                    
                    # Retrieved chunks
                    with st.expander("🔍 View Retrieved Chunks"):
                        for j, chunk in enumerate(qa['retrieved_chunks'], 1):
                            st.markdown(f"""
                            <div class="chunk-box">
                            <strong>[{j}] {chunk['document']} - Chunk {chunk['chunk_num']}</strong><br>
                            <em>Relevance: {chunk['relevance_score']:.3f}</em><br><br>
                            {chunk['text'][:300]}...
                            </div>
                            """, unsafe_allow_html=True)


def process_documents(uploaded_files, client):
    """Process uploaded documents"""
    
    with st.spinner("Processing documents... This may take a few minutes."):
        progress_bar = st.progress(0, text="Starting...")
        
        # Step 1: Load documents
        progress_bar.progress(0.1, text="📥 Loading documents...")
        documents = {}
        for i, file in enumerate(uploaded_files):
            doc_name = file.name.replace('.pdf', '')
            text = DocumentLoader.load_pdf(file)
            documents[doc_name] = text
            progress_bar.progress(0.1 + (i + 1) * 0.1 / len(uploaded_files), text=f"Loaded {i+1}/{len(uploaded_files)} documents")
        
        # Step 2: Chunk documents
        progress_bar.progress(0.2, text="✂️ Chunking documents...")
        chunker = TextChunker()
        all_chunks = []
        for i, (doc_name, text) in enumerate(documents.items()):
            chunks = chunker.chunk_text(text, doc_name)
            all_chunks.extend(chunks)
            progress_bar.progress(0.2 + (i + 1) * 0.1 / len(documents), text=f"Chunked {i+1}/{len(documents)} documents")
        
        # Step 3: Generate embeddings
        progress_bar.progress(0.3, text=f"🧠 Generating embeddings for {len(all_chunks)} chunks... (~{len(all_chunks) * 0.65 / 60:.1f} minutes)")
        embedding_gen = EmbeddingGenerator(client)
        
        chunk_texts = [chunk['text'] for chunk in all_chunks]
        
        embedding_progress = st.progress(0, text="Generating embeddings...")
        embeddings = embedding_gen.batch_generate(chunk_texts, embedding_progress)
        
        progress_bar.progress(0.9, text="🔗 Building vector index...")
        
        # Step 4: Build vector store
        vector_store = VectorStore()
        vector_store.add_chunks(all_chunks, embeddings)
        
        # Initialize other components
        answer_gen = AnswerGenerator(client)
        
        # Store in session state
        st.session_state.documents_loaded = True
        st.session_state.vector_store = vector_store
        st.session_state.embedding_gen = embedding_gen
        st.session_state.answer_gen = answer_gen
        st.session_state.all_chunks = all_chunks
        
        progress_bar.progress(1.0, text="✅ Complete!")
        embedding_progress.empty()
        
        st.success(f"✅ Successfully processed {len(documents)} documents with {len(all_chunks)} chunks!")
        time.sleep(1)
        progress_bar.empty()
        st.rerun()


def answer_question(question: str, client):
    """Answer a user question"""
    
    with st.spinner("🔍 Searching and generating answer..."):
        # Retrieve
        query_embedding = st.session_state.embedding_gen.generate_embedding(question)
        results = st.session_state.vector_store.search(query_embedding, top_k=5)
        
        retrieved_chunks = [
            {
                'chunk_id': chunk['id'],
                'document': chunk['document'],
                'chunk_num': chunk['chunk_num'],
                'text': chunk['text'],
                'relevance_score': float(score)
            }
            for chunk, score in results
        ]
        
        # Generate answer
        result = st.session_state.answer_gen.generate_answer(question, retrieved_chunks)
        
        # Add to history
        st.session_state.qa_history.append(result)
        
    st.success("✅ Answer generated!")
    st.rerun()


if __name__ == "__main__":
    main()
