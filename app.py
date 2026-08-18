import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
import io
import math

# Page Configuration
st.set_page_config(
    page_title="DocuMind AI — Chat with PDFs",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App Styles
st.markdown("""
<style>
.main-title {
    font-size: 2.5rem;
    font-weight: 700;
    color: #1e3a8a;
    margin-bottom: 5px;
}
.sub-title {
    font-size: 1.1rem;
    color: #4b5563;
    margin-bottom: 25px;
}
.sidebar-section {
    font-weight: 600;
    margin-top: 15px;
    margin-bottom: 5px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Core RAG Algorithms (Pure-Python Vector Store)
# -------------------------------------------------------------

def extract_pdf_text(uploaded_file):
    """Extract text from an uploaded PDF file handle."""
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks of defined character size."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

def generate_embeddings(texts, api_key):
    """Fetch embeddings for a list of text chunks from Gemini API."""
    genai.configure(api_key=api_key)
    try:
        response = genai.embed_content(
            model="models/embedding-001",
            content=texts,
            task_type="retrieval_document"
        )
        return response['embedding']
    except Exception as e:
        st.error(f"Failed to generate embeddings: {e}")
        return None

def cosine_similarity(v1, v2):
    """Compute cosine similarity between two unit vectors (simple dot product)."""
    # Gemini embeddings are L2-normalized unit vectors, so dot product equals cosine similarity.
    return sum(x * y for x, y in zip(v1, v2))

def search_vector_store(query, database, api_key, top_k=3):
    """Embed query, search database using cosine similarity, return top K results."""
    genai.configure(api_key=api_key)
    try:
        query_response = genai.embed_content(
            model="models/embedding-001",
            content=query,
            task_type="retrieval_query"
        )
        query_vector = query_response['embedding']
    except Exception as e:
        st.error(f"Failed to embed query: {e}")
        return []

    # Score each chunk
    scored_chunks = []
    for doc in database:
        score = cosine_similarity(query_vector, doc['embedding'])
        scored_chunks.append((score, doc['text']))

    # Sort by score descending and return top K
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return scored_chunks[:top_k]

# -------------------------------------------------------------
# Streamlit Interface
# -------------------------------------------------------------

st.markdown('<div class="main-title">🤖 DocuMind AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Retrieval-Augmented Generation (RAG) PDF Chatbot</div>', unsafe_allow_html=True)

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "database" not in st.session_state:
    st.session_state.database = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None

# Sidebar Setup
with st.sidebar:
    st.markdown('<div class="sidebar-section">🔑 Authentication</div>', unsafe_allow_html=True)
    api_key = st.text_input("Gemini API Key", type="password", help="Get a free key from Google AI Studio")
    st.markdown("[Get Free Gemini API Key ↗](https://aistudio.google.com/)", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<div class="sidebar-section">📂 Upload PDF Document</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file and api_key:
        # Rebuild DB if a new file is uploaded
        if st.session_state.current_file != uploaded_file.name:
            with st.spinner("Parsing document & indexing embeddings..."):
                text = extract_pdf_text(uploaded_file)
                if not text.strip():
                    st.error("Uploaded PDF is empty or could not be read.")
                else:
                    chunks = chunk_text(text)
                    st.info(f"Split document into {len(chunks)} chunks.")
                    
                    # Batch generate embeddings
                    embeddings = generate_embeddings(chunks, api_key)
                    if embeddings:
                        db = []
                        for txt, emb in zip(chunks, embeddings):
                            db.append({"text": txt, "embedding": emb})
                        st.session_state.database = db
                        st.session_state.current_file = uploaded_file.name
                        st.session_state.messages = [] # Clear history on new document
                        st.success("Document indexed successfully! You can now chat.")
    elif not api_key:
        st.warning("Please enter your Gemini API Key in the sidebar to unlock document indexing and chat.")

# Chat Window
if not api_key:
    st.info("🔒 Secure Workspace: Enter your Gemini API key in the sidebar configuration to begin.")
elif not st.session_state.database:
    st.info("📄 Upload a PDF document in the sidebar to query its contents with AI.")
else:
    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if query := st.chat_input("Ask a question about your uploaded document..."):
        # Display user query
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        # Generate RAG response
        with st.chat_message("assistant"):
            with st.spinner("Scanning vector index & synthesizing response..."):
                # 1. Retrieve top context chunks
                results = search_vector_store(query, st.session_state.database, api_key, top_k=3)
                
                if not results:
                    response_text = "Sorry, I encountered an error while searching document embeddings."
                else:
                    # Construct context payload
                    context = "\n\n".join([f"--- Context Chunk ---\n{text}" for score, text in results])
                    
                    # 2. Formulate context-bound system prompt
                    prompt = (
                        "You are DocuMind AI, an expert research assistant. Answer the user's question "
                        "based ONLY on the extracted document context provided below. If the information is not "
                        "available in the context, state that you cannot find the answer in the document.\n\n"
                        f"Document Context:\n{context}\n\n"
                        f"User Question: {query}\n\n"
                        "Answer:"
                    )
                    
                    # 3. Request LLM generation
                    try:
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel("models/gemini-1.5-flash")
                        response = model.generate_content(prompt)
                        response_text = response.text
                    except Exception as e:
                        response_text = f"Failed to generate response: {e}"
                
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
