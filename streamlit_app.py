"""
RAG Question Answering System - Web Interface (Clean Error Handling)
====================================================================
Interactive web application with user-friendly error messages.

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
import traceback

# Page configuration
st.set_page_config(
    page_title="RAG QA System - AI Research Papers",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme-adaptive CSS
st.markdown("""
<style>
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
    .chunk-box {
        background-color: var(--secondary-background-color);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid var(--background-color);
    }
    .answer-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E88E5;
        background-color: var(--secondary-background-color);
        margin: 1rem 0;
    }
    .source-box {
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.3rem 0;
        background-color: var(--secondary-background-color);
        border-left: 3px solid #FF9800;
    }
    .feature-highlight {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: var(--secondary-background-color);
        text-align: center;
        height: 100%;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# ERROR HANDLING UTILITIES
# ============================================================================

def get_user_friendly_error(error: Exception) -> tuple:
    """
    Convert technical errors to user-friendly messages
    Returns: (error_title, error_message, solution, show_details_link)
    """
    error_str = str(error)
    error_type = type(error).__name__
    
    # API Key errors
    if "401" in error_str or "Unauthorized" in error_str or "UNAUTHENTICATED" in error_str:
        return (
            "🔑 Invalid API Key",
            "The API key you provided is not valid or has expired.",
            "**How to fix:**\n1. Double-check your API key for typos\n2. Get a new API key from [Google AI Studio](https://aistudio.google.com/app/apikey)\n3. Make sure you copied the entire key",
            False
        )
    
    # Rate limit errors
    elif "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "rate" in error_str.lower():
        return (
            "⏱️ Rate Limit Reached",
            "You've hit the free tier limit of 100 requests per minute.",
            "**How to fix:**\n1. Wait 60 seconds and try again\n2. The system will automatically retry\n3. For faster processing, upgrade to a paid API tier",
            False
        )
    
    # Model not found errors
    elif "404" in error_str or "NOT_FOUND" in error_str or "not found" in error_str.lower():
        return (
            "🔍 Model Not Available",
            "The AI model specified is not available with your API key.",
            "**How to fix:**\n1. Check if your API key has access to Gemini models\n2. Try creating a new API key\n3. Make sure you're using a valid Google AI Studio key",
            False
        )
    
    # Network/connection errors
    elif "ConnectionError" in error_type or "Timeout" in error_type or "network" in error_str.lower():
        return (
            "🌐 Connection Error",
            "Unable to connect to Google's AI services.",
            "**How to fix:**\n1. Check your internet connection\n2. Try again in a few moments\n3. If the problem persists, Google's API may be temporarily down",
            False
        )
    
    # PDF processing errors
    elif "PDF" in error_str or "PdfReader" in error_str:
        return (
            "📄 PDF Processing Error",
            "Unable to read one of the uploaded PDF files.",
            "**How to fix:**\n1. Make sure the file is a valid PDF\n2. Try re-uploading the file\n3. If the PDF is scanned or image-based, it may not work (text-based PDFs only)",
            False
        )
    
    # Quota/billing errors
    elif "quota" in error_str.lower() or "billing" in error_str.lower():
        return (
            "💳 Quota Exceeded",
            "Your API quota has been exhausted.",
            "**How to fix:**\n1. Wait for your quota to reset (usually daily/monthly)\n2. Check your usage at [Google AI Studio](https://aistudio.google.com/)\n3. Consider upgrading to a paid tier for higher limits",
            False
        )
    
    # Generic API errors
    elif "ClientError" in error_type or "google.genai" in error_str:
        return (
            "⚠️ API Error",
            "The Google AI service returned an error.",
            "**How to fix:**\n1. Try again in a moment\n2. Check if your API key is still valid\n3. Make sure you have an active internet connection",
            True
        )
    
    # Generic errors
    else:
        return (
            "❌ Unexpected Error",
            "Something went wrong while processing your request.",
            "**How to fix:**\n1. Try refreshing the page\n2. Re-upload your documents\n3. If the problem persists, please report it",
            True
        )


def show_error(error: Exception, context: str = ""):
    """Display user-friendly error message"""
    title, message, solution, show_details = get_user_friendly_error(error)
    
    # Show main error
    st.error(f"**{title}**\n\n{message}")
    
    # Show solution in info box
    st.info(solution)
    
    # Optional: Show technical details in expander (for debugging)
    if show_details:
        with st.expander("🔧 Technical Details (for debugging)"):
            st.code(f"{type(error).__name__}: {str(error)}", language="python")
            if context:
                st.caption(f"Context: {context}")


# ============================================================================
# RAG SYSTEM COMPONENTS (with error handling)
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
            raise Exception(f"PDF Error: Unable to read {uploaded_file.name}. Make sure it's a valid text-based PDF.")


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
    """Generate embeddings with rate limiting and error handling"""
    
    def __init__(self, client, model_name: str = "models/gemini-embedding-001"):
        self.client = client
        self.model_name = model_name
        self.request_count = 0
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for single text with error handling"""
        try:
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=text
            )
            self.request_count += 1
            return np.array(result.embeddings[0].values)
        except Exception as e:
            # Check if it's a rate limit error
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                # Wait and retry once
                st.warning("⏱️ Rate limit reached. Waiting 60 seconds...")
                time.sleep(60)
                result = self.client.models.embed_content(
                    model=self.model_name,
                    contents=text
                )
                self.request_count += 1
                return np.array(result.embeddings[0].values)
            else:
                raise
    
    def batch_generate(self, texts: List[str], progress_bar=None) -> List[np.ndarray]:
        """Generate embeddings with progress tracking and error handling"""
        embeddings = []
        total = len(texts)
        
        for i, text in enumerate(texts):
            try:
                embedding = self.generate_embedding(text)
                embeddings.append(embedding)
                
                if progress_bar:
                    progress_bar.progress((i + 1) / total)
                
                # Rate limiting
                if i < total - 1:
                    time.sleep(0.65)
                    
            except Exception as e:
                # Show which chunk failed
                show_error(e, f"Processing chunk {i+1}/{total}")
                raise
        
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
    """Generate answers using LLM with error handling"""
    
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
        """Generate answer with citations and error handling"""
        try:
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
        except Exception as e:
            show_error(e, "Generating answer")
            raise


# ============================================================================
# STREAMLIT APP
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
    
    # Header
    st.markdown('<div class="main-header">🤖 RAG Question Answering System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask questions about AI research papers with intelligent retrieval and grounded answers</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key Setup
        with st.expander("🔑 Get Your Free API Key", expanded=False):
            st.markdown("""
            **Quick Setup (30 seconds):**
            
            1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
            2. Click "Create API Key"
            3. Click "Create API key in new project"
            4. Copy and paste below
            
            💡 **Free tier:** 100 requests/min
            """)
        
        api_key = st.text_input(
            "Google AI API Key",
            type="password",
            help="Your key is only used in this session",
            placeholder="Paste your API key here..."
        )
        
        if api_key:
            try:
                client = genai.Client(api_key=api_key)
                # Test the API key with a simple call
                st.success("✅ API Key Valid")
            except Exception as e:
                show_error(e, "Validating API key")
                st.stop()
        else:
            st.warning("👆 Please enter your API key to continue")
            st.info("Don't have one? Click the dropdown above!")
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
                st.metric("Documents", len(uploaded_files) if uploaded_files else 0)
                st.metric("Chunks", len(st.session_state.all_chunks))
            with col2:
                st.metric("Embeddings", len(st.session_state.all_chunks))
                st.metric("Questions", len(st.session_state.qa_history))
        
        st.divider()
        
        # About
        st.header("ℹ️ About")
        st.markdown("""
        **RAG System** for research papers
        
        **Models:**
        - Embeddings: `gemini-embedding-001`
        - Generation: `gemini-2.5-flash`
        """)
        
        st.divider()
        st.caption("Built with Streamlit & Google Gemini")
    
    # Main content area
    if not st.session_state.documents_loaded:
        # Welcome screen
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
        
        # Features
        st.subheader("✨ Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="feature-highlight">
            <h3>🎯 Accurate</h3>
            <p>Grounded answers with citations</p>
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
                    st.markdown(f'<div class="answer-box"><strong>Answer:</strong><br><br>{qa["answer"]}</div>', unsafe_allow_html=True)
                    
                    st.markdown("**📚 Sources:**")
                    for source in qa['sources']:
                        st.markdown(f'<div class="source-box">📄 {source}</div>', unsafe_allow_html=True)
                    
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
    """Process uploaded documents with error handling"""
    
    try:
        with st.spinner("Processing documents..."):
            progress_bar = st.progress(0, text="Starting...")
            
            # Step 1: Load documents
            progress_bar.progress(0.1, text="📥 Loading documents...")
            documents = {}
            for i, file in enumerate(uploaded_files):
                try:
                    doc_name = file.name.replace('.pdf', '')
                    text = DocumentLoader.load_pdf(file)
                    documents[doc_name] = text
                    progress_bar.progress(0.1 + (i + 1) * 0.1 / len(uploaded_files))
                except Exception as e:
                    show_error(e, f"Loading {file.name}")
                    progress_bar.empty()
                    return
            
            # Step 2: Chunk documents
            progress_bar.progress(0.2, text="✂️ Chunking documents...")
            chunker = TextChunker()
            all_chunks = []
            for doc_name, text in documents.items():
                chunks = chunker.chunk_text(text, doc_name)
                all_chunks.extend(chunks)
            
            # Step 3: Generate embeddings
            progress_bar.progress(0.3, text=f"🧠 Generating embeddings... (~{len(all_chunks) * 0.65 / 60:.1f} minutes)")
            embedding_gen = EmbeddingGenerator(client)
            
            chunk_texts = [chunk['text'] for chunk in all_chunks]
            embedding_progress = st.progress(0)
            
            try:
                embeddings = embedding_gen.batch_generate(chunk_texts, embedding_progress)
            except Exception as e:
                show_error(e, "Generating embeddings")
                progress_bar.empty()
                embedding_progress.empty()
                return
            
            # Step 4: Build vector store
            progress_bar.progress(0.9, text="🔗 Building vector index...")
            vector_store = VectorStore()
            vector_store.add_chunks(all_chunks, embeddings)
            
            answer_gen = AnswerGenerator(client)
            
            # Store in session state
            st.session_state.documents_loaded = True
            st.session_state.vector_store = vector_store
            st.session_state.embedding_gen = embedding_gen
            st.session_state.answer_gen = answer_gen
            st.session_state.all_chunks = all_chunks
            
            progress_bar.progress(1.0, text="✅ Complete!")
            st.success(f"✅ Processed {len(documents)} documents with {len(all_chunks)} chunks!")
            
            time.sleep(1)
            progress_bar.empty()
            embedding_progress.empty()
            st.rerun()
            
    except Exception as e:
        show_error(e, "Processing documents")


def answer_question(question: str, client):
    """Answer a user question with error handling"""
    
    try:
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
        
    except Exception as e:
        show_error(e, "Answering question")


if __name__ == "__main__":
    main()
