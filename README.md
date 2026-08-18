# DocuMind AI — RAG-Powered PDF Chatbot

DocuMind AI is a Retrieval-Augmented Generation (RAG) web application that enables users to upload PDF documents and hold context-aware interactive chats with them. 

Rather than uploading the entire document to the LLM (which is slow and expensive), the application uses **vector search** to find only the most relevant sections of the text, feeds them to Google Gemini as context, and generates precise, source-cited responses.

## Key Features
* **Zero-Dependency Vector Store**: Features a custom, pure-Python cosine similarity search engine. This avoids the compilation and binary installation issues commonly associated with complex C++ vector libraries on Windows (like ChromaDB or FAISS).
* **Gemini API Integration**: Leverages `text-embedding-004` to create high-dimensional document vectors and `gemini-1.5-flash` for rapid conversational reasoning.
* **Context-Bound Answers**: Restricts the AI's responses to only the uploaded document contents to eliminate hallucinations.
* **State Management**: Persists uploaded documents, chunk embeddings, and complete chat histories across user sessions.

## Tech Stack
* **Frontend UI**: Streamlit
* **AI Orchestration**: Google Generative AI SDK
* **PDF Parser**: PyPDF2
* **Language**: Python 3.10+

## How It Works (RAG Architecture)
1. **Document Loading**: The PDF is parsed, and its text is extracted recursively.
2. **Text Chunking**: The text is split into chunks of 1,000 characters with a 200-character sliding overlap.
3. **Embedding Generation**: Gemini's embedding model translates each text chunk into a 768-dimensional floating-point vector.
4. **Vector Search**: The user's query is embedded, and a pure-Python cosine similarity dot-product compares it against all chunk vectors.
5. **Context Augmentation**: The top-scoring text chunks are injected into the prompt context.
6. **Gemini Query**: The LLM reads the custom context-augmented prompt and responds.

## Setup & Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/nishanth702/DocuMind-AI.git
   cd DocuMind-AI
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Get a Free Gemini API Key**:
   Obtain an API key from [Google AI Studio](https://aistudio.google.com/).

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```
   Open `http://localhost:8501` in your browser.
